# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Modification points:
# 1. Adapted from static graph DSA implementation to pynative mode.
# 2. Removed parallel sharding (single card only).
# 3. Custom backward uses _Function (torch.autograd.Function style).
# 4. Uses pynative Linear and mint/ops instead of aclnn_ops.
#
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""DSA Lightning Indexer for pynative mode."""
# MindSpore ``Cell``/``_Function`` subclasses intentionally use operator-specific signatures.
# pylint: disable=arguments-differ,abstract-method
import copy
from dataclasses import dataclass
from typing import Union

import mindspore.common.dtype as mstype
from mindspore import nn, ops, mint
from mindspore.common._grad_function import _Function

from hyper_parallel.custom_ops.experimental import npu_dense_lightning_indexer_softmax_lse
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.pynative.base_models.common.embeddings.rope_utils import ApplyRotaryPosEmb
from mindformers.pynative.layers.identity_op import IdentityOp
from mindformers.pynative.transformers.experimental_attention_variant.utils import Hadamard


class _DSAIndexerFunction(_Function):
    """
    Custom autograd function to block gradient flow from main loss to indexer inputs.

    In DSAIndexer, the inputs (q, k, weights) are pre-computed by the indexer's linear
    layers. The main attention loss should NOT propagate gradients back through these
    inputs — the indexer is trained solely via DSAIndexerLoss. This function explicitly
    returns zero gradients for all inputs.
    """

    @staticmethod
    def forward(ctx, q, k, weights):
        """Forward pass: return inputs unchanged (identity function)."""
        ctx.q = q
        ctx.k = k
        ctx.weights = weights
        return q, k, weights

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_w):
        """Backward pass: return zero gradients to block gradient flow."""
        del grad_q, grad_k, grad_w
        return (mint.zeros_like(ctx.q),
                mint.zeros_like(ctx.k),
                mint.zeros_like(ctx.weights))


@dataclass
class DSAIndexerSubmodules:
    """
    Configuration class for specifying the submodules of a DSA Indexer.

    Args:
        linear_wq_b: Linear projection for query bottleneck expansion.
        linear_wk: Linear projection for key.
        k_norm: Layer normalization for key.
        linear_weights_proj: Linear projection for attention weights.
    """

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_wk: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None


class DSAIndexerComputeSparseIndices(nn.Cell):
    """Hookable DSA indexer boundary for sparse top-k and dense LSE stats.

    MF can attach one hook to this submodule and cover both:
    - ``ops.lightning_indexer`` for sparse top-k selection
    - ``npu_dense_lightning_indexer_softmax_lse`` for dense-stage statistics

    Both operators consume the same q/k/weights placements, so they belong to
    the same hookable boundary even though they are different kernels.
    """

    def __init__(self, input_layout: str, sparse_count: int, sparse_loss: bool):
        super().__init__()
        self.input_layout = input_layout
        self.sparse_count = sparse_count
        self.sparse_loss = sparse_loss

    def construct(self, query, key, weights, actual_seq_qlen=None, actual_seq_klen=None):
        """Compute sparse Top-K outputs or dense-stage softmax statistics."""
        query = ops.cast(query, mstype.bfloat16)
        key = ops.cast(key, mstype.bfloat16)
        if self.sparse_loss:
            topk_indices, index_scores = ops.lightning_indexer(
                query, key, weights,
                actual_seq_lengths_query=actual_seq_qlen,
                actual_seq_lengths_key=actual_seq_klen,
                layout_query=self.input_layout,
                layout_key=self.input_layout,
                sparse_count=self.sparse_count,
                return_value=True
            )
            softmax_max_index, softmax_sum_index = None, None
        else:
            softmax_max_index, softmax_sum_index = npu_dense_lightning_indexer_softmax_lse(
                query, key, weights,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_klen=actual_seq_klen,
                layout=self.input_layout,
            )
            topk_indices, index_scores = None, None

        return topk_indices, index_scores, softmax_max_index, softmax_sum_index


class DSAIndexer(nn.Cell):
    """
    DSA Lightning Indexer for pynative mode.

    Computes index scores to identify the top-k most relevant key-value pairs
    for each query in sparse attention. Single card only, no parallel sharding.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L431-L480

    Args:
        config (MLATransformerConfig): The configuration for the transformer model.
        submodules (DSAIndexerSubmodules): Indexer submodules specification.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: DSAIndexerSubmodules,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_layout = config.input_layout
        self.is_tnd = self.input_layout == "TND"
        self.hidden_size = config.hidden_size
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim
        self.q_lora_rank = config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size

        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk
        self.sparse_loss = config.dsa_indexer_use_sparse_loss
        self.softmax_scale: float = self.index_head_dim ** -0.5

        self.apply_rotary_emb_q = ApplyRotaryPosEmb(config)
        self.apply_rotary_emb_k = ApplyRotaryPosEmb(config)

        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            input_size=self.q_lora_rank,
            output_size=self.index_n_heads * self.index_head_dim,
            params_dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            init_method=config.init_method,
            bias=False,
        )

        self.linear_wk = build_module(
            submodules.linear_wk,
            input_size=self.hidden_size,
            output_size=self.index_head_dim,
            params_dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            init_method=config.init_method,
            bias=False,
        )

        k_norm_config = copy.copy(config)
        k_norm_config.normalization = "LayerNorm"
        self.k_norm = build_module(
            submodules.k_norm,
            dim=self.index_head_dim,
            eps=config.layernorm_epsilon,
            compute_dtype=config.layernorm_compute_dtype,
        )

        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            input_size=self.hidden_size,
            output_size=self.index_n_heads,
            params_dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            init_method=config.init_method,
            bias=False,
        )

        self.hadamard_q = Hadamard(self.index_head_dim, use_butterfly=True)
        self.hadamard_k = Hadamard(self.index_head_dim, use_butterfly=True)
        self.compute_sparse_indices = DSAIndexerComputeSparseIndices(
            input_layout=self.input_layout,
            sparse_count=self.index_topk,
            sparse_loss=self.sparse_loss,
        )
        self.key_handoff = IdentityOp()

    def get_qk_index(self, x, qr, rotary_pos_emb):
        """
        Pre-forward pass that computes indexer Q, K and weights.

        Args:
            x: Hidden states, shape (S, B, hidden_size).
            qr: Compressed query, shape (S, B, q_lora_rank).
            rotary_pos_emb: Rotary position embedding.

        Returns:
            q: Indexer query, shape (B, S, n_idx_heads, head_dim).
            k: Indexer key, shape (B, S, 1, head_dim).
            weights: Indexer weights, shape (B, S, n_idx_heads).
        """
        seqlen, bsz, _ = x.shape

        # Query path: linear -> reshape -> split -> RoPE -> concat -> Hadamard
        q = self.linear_wq_b(qr)
        q = mint.reshape(q, (seqlen, bsz, self.index_n_heads, self.index_head_dim))
        # Align with the graph-mode indexer: pe is the FIRST head-dim slice and
        # rope is applied without MLA de-interleave. Order matters because the
        # downstream hadamard mixes the concatenated dims.
        q_pe, q_nope = mint.split(
            q, [self.qk_pos_emb_head_dim, self.index_head_dim - self.qk_pos_emb_head_dim], dim=-1
        )
        q_pe = self.apply_rotary_emb_q(
            q_pe, rotary_pos_emb,
            rotary_interleaved=False,
            multi_latent_attention=False
        )
        q = mint.cat([q_pe, q_nope], dim=-1)

        # Key path: linear -> norm -> reshape -> split -> RoPE -> concat -> Hadamard
        k = self.linear_wk(x)
        k = self.k_norm(k)
        k = mint.reshape(k, (seqlen, bsz, 1, self.index_head_dim))
        k_pe, k_nope = mint.split(
            k, [self.qk_pos_emb_head_dim, self.index_head_dim - self.qk_pos_emb_head_dim], dim=-1
        )
        k_pe = self.apply_rotary_emb_k(
            k_pe, rotary_pos_emb,
            rotary_interleaved=False,
            multi_latent_attention=False
        )
        k = mint.cat([k_pe, k_nope], dim=-1)

        # Hadamard transform
        q = self.hadamard_q(q)
        k = self.hadamard_k(k)

        # Weights path: linear -> scale
        weights = self.linear_weights_proj(x)
        # Keep the indexer weights in the projection dtype.  A float32 Tensor
        # scale promotes BF16 weights to FP32 in MindSpore, while stock
        # Megatron and the graph-mode DSA path use a Python scalar and retain
        # BF16 for both the index-score forward and d_weights backward paths.
        weights = weights * ((self.index_n_heads ** -0.5) * self.softmax_scale)

        # Convert from SB* layout to BS* layout
        weights = mint.permute(weights, (1, 0, 2))
        q = mint.permute(q, (1, 0, 2, 3))
        k = mint.permute(k, (1, 0, 2, 3))

        if self.is_tnd:
            q = mint.reshape(q, (seqlen * bsz, self.index_n_heads, self.index_head_dim))
            k = mint.reshape(k, (seqlen * bsz, 1, self.index_head_dim))
            weights = mint.reshape(weights, (seqlen * bsz, self.index_n_heads))
        k = self.key_handoff(k)
        return q, k, weights

    def construct(self, q, k, weights, actual_seq_qlen=None, actual_seq_klen=None):
        """
        Forward pass: compute top-k indices using lightning_indexer.

        Gradient flow from the main attention loss to indexer inputs is blocked
        via _DSAIndexerFunction. Indexer parameters are updated solely via
        DSAIndexerLoss.

        Args:
            q: Indexer query, shape (B, S, n_idx_heads, head_dim).
            k: Indexer key, shape (B, S, 1, head_dim).
            weights: Indexer weights, shape (B, S, n_idx_heads).

        Returns:
            topk_indices: Top-k indices, shape (B, S, 1, topk).
            index_scores: Top-k scores, shape (B, S, 1, topk).
            softmax_max_index: Softmax max (dense stage only).
            softmax_sum_index: Softmax sum (dense stage only).
        """
        # Block gradient flow from main loss to indexer inputs via custom autograd Function.
        # Use _DSAIndexerFunction (a _Function) instead of ops.stop_gradient because
        # stop_gradient creates a hard break in the autograd graph that prevents
        # _DSAIndexerLossFunction from routing gradients back to indexer parameters.
        # _DSAIndexerFunction blocks the main-loss gradient path while keeping the
        # computation graph intact for the indexer-loss gradient route.
        q, k, weights = _DSAIndexerFunction.apply(q, k, weights)

        (
            topk_indices,
            index_scores,
            softmax_max_index,
            softmax_sum_index,
        ) = self.compute_sparse_indices(
            q, k, weights,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
        )
        if not self.sparse_loss:
            return softmax_max_index, softmax_sum_index
        return topk_indices, index_scores, softmax_max_index, softmax_sum_index
