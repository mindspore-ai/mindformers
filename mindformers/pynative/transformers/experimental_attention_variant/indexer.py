# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""DSA/CSA Lightning Indexer for pynative mode."""
from dataclasses import dataclass
from typing import Union

import mindspore.common.dtype as mstype
from mindspore import nn, ops, Tensor, mint, _no_grad
from mindspore.common._grad_function import _Function
from hyper_parallel.core.dtensor.dtensor import DTensor

try:
    from hyper_parallel.custom_ops.experimental import (
        npu_lightning_indexer_v2,
        npu_sparse_lightning_indexer_grad_kl_loss_v2
    )
except ImportError:
    npu_lightning_indexer_v2 = None
    npu_sparse_lightning_indexer_grad_kl_loss_v2 = None

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.pynative.transformers.experimental_attention_variant.utils import Hadamard
from mindformers.pynative.base_models.common.embeddings.rope_utils import ApplyRotaryPosEmb
from mindformers.pynative.layers.identity_op import IdentityOp


# Minimum finite BF16 value, matching MindSpeed's torch.finfo(query.dtype).min mask fill.
_BF16_MIN = -3.3895313892515355e38


@dataclass
class CSAIndexerSubmodules:
    """
    Configuration class for specifying the submodules of a CSA Indexer.

    Args:
        linear_wq_b: Linear projection for query bottleneck expansion.
        linear_weights_proj: Linear projection for attention weights.
        compressor: Compressor for key.
    """

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None
    compressor: Union[ModuleSpec, type] = None


class CSAIndexer(nn.Cell):
    """
    CSA Lightning Indexer for pynative mode.
    Computes index scores to select the most relevant compressed KV positions for each query.

    Args:
        config (MLATransformerConfig): The configuration for the transformer model.
        submodules (DSAIndexerSubmodules): Indexer submodules specification.
    """

    def __init__(
            self,
            config: MLATransformerConfig,
            submodules: CSAIndexerSubmodules,
            compress_ratio: int,
            rotary_pos_emb: nn.Cell = None
    ) -> None:
        super().__init__()
        self.config = config
        self.sparse_mode = 3
        self.compress_ratio = compress_ratio
        self.rotary_pos_emb = rotary_pos_emb

        self.input_layout = config.input_layout
        self.is_tnd = self.input_layout == "TND" and config.apply_dsa_kernel_fusion
        self.hidden_size = config.hidden_size
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim
        self.q_lora_rank = config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size
        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk
        self.sparse_loss = config.dsa_indexer_use_sparse_loss
        self.softmax_scale = self.index_head_dim ** -0.5
        self.compute_dtype = config.compute_dtype
        self.apply_dsa_kernel_fusion = config.apply_dsa_kernel_fusion
        if self.apply_dsa_kernel_fusion and npu_lightning_indexer_v2 is None:
            raise ValueError(
                "Fused DSA op 'npu_lightning_indexer_v2' is unavailable in this hyper_parallel build."
                "set apply_dsa_kernel_fusion=False to use the unfused small-op implement."
            )

        self.weight_scale = self.index_n_heads ** -0.5 * self.softmax_scale

        self.apply_rope = ApplyRotaryPosEmb(config)

        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            input_size=self.q_lora_rank,
            output_size=self.index_n_heads * self.index_head_dim,
            params_dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            init_method=config.init_method,
            bias=False,
        )
        self.compressor = build_module(
            submodules.compressor,
            config=self.config,
            compress_ratio=self.compress_ratio,
            head_dim=self.index_head_dim,
            rotate=True,
            rotary_pos_emb=rotary_pos_emb
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

        self.hadamard = Hadamard(self.index_head_dim)

        self.cast = ops.cast
        self.reshape = mint.reshape
        self.split = mint.split
        self.cat = mint.concat
        self.permute = mint.permute
        # operators for unfused implement
        self.bmm = mint.matmul
        self.relu = mint.nn.functional.relu
        self.unsqueeze = mint.unsqueeze
        self.sum = mint.sum
        self.topk = mint.topk
        self.clamp = mint.clamp

    def forward_before_topk(self, x, qr):
        """
        Pre-forward pass that computes indexer Q, K and weights.

        Args:
            x: Hidden states, shape (S, B, hidden_size).
            qr: Compressed query, shape (S, B, q_lora_rank).

        Returns:
            q: Indexer query, shape (B, S, n_idx_heads, head_dim).
            k: Indexer key, shape (B, S, 1, head_dim).
            weights: Indexer weights, shape (B, S, n_idx_heads).
        """
        seqlen, bsz, _ = x.shape
        freqs, _ = self.rotary_pos_emb(seqlen)

        # Query path: linear -> reshape -> split -> RoPE -> concat -> Hadamard
        q = self.linear_wq_b(qr)
        q = self.reshape(q, (seqlen, bsz, self.index_n_heads, self.index_head_dim))
        q_nope, q_pe = self.split(
            q, [self.index_head_dim - self.qk_pos_emb_head_dim, self.qk_pos_emb_head_dim], dim=-1
        )
        q_pe = self.apply_rope(
            q_pe, freqs, 1,
            rotary_interleaved=self.config.rotary_interleaved,
            multi_latent_attention=True,
            mla_output_remove_interleaving=True
        )
        q = self.cat([q_nope, q_pe], dim=-1)
        q = self.hadamard(q)
        k = self.compressor(x)

        # Weights path: linear -> scale
        weights = self.linear_weights_proj(x)
        weights = weights * self.weight_scale

        # Convert from SB* layout to BS* layout
        weights = self.permute(weights, (1, 0, 2))
        q = self.permute(q, (1, 0, 2, 3))
        k = self.permute(k, (1, 0, 2, 3))

        if self.is_tnd:
            q = self.reshape(q, (seqlen * bsz, self.index_n_heads, self.index_head_dim))
            k = self.reshape(k, (seqlen * bsz, 1, self.index_head_dim))
            weights = self.reshape(weights, (-1, self.index_n_heads))

        return q, k, weights

    def construct(self, q, k, weights, mask=None, actual_seq_qlen=None, actual_seq_klen=None):
        """compute top-k indices using lightning_indexer."""
        effective_topk = min(self.index_topk, int(k.shape[1]))
        if self.apply_dsa_kernel_fusion:
            with _no_grad():
                q = self.cast(q, self.compute_dtype)
                k = self.cast(k, self.compute_dtype)
                topk_indices, index_scores = npu_lightning_indexer_v2(
                    q, k, weights,
                    layout=self.input_layout,
                    cu_seq_lens_q=actual_seq_qlen,
                    cu_seq_lens_k=actual_seq_klen,
                    sparse_count=effective_topk,
                    cmp_ratio=self.compress_ratio,
                    sparse_mode=self.sparse_mode,
                )
        else:
            q = self.cast(q, mstype.float32)
            k = self.cast(k, mstype.float32)
            weights = self.cast(weights, mstype.float32)

            #   q: [b, sq, n, d] -> [b, sq*n, d]
            #   k: [b, sk, 1, d] -> [b, d, sk]
            #   bmm -> [b, sq*n, sk] -> [b, sq, n, sk]
            b, sq, n, d = q.shape
            sk = k.shape[1]
            q = self.reshape(q, (b, sq * n, d))
            k = self.permute(self.reshape(k, (b, sk, d)), (0, 2, 1))
            index_scores = self.bmm(q, k)
            index_scores = self.reshape(index_scores, (b, sq, n, sk))
            index_scores = self.relu(index_scores)

            # Weight each head by attention weights.
            # [b, sq, n, sk] * [b, sq, n, 1] -> [b, sq, n, sk]
            index_scores = index_scores * self.unsqueeze(weights, -1)

            # Sum across attention heads. [b, sq, n, sk] -> [b, sq, sk]
            index_scores = self.sum(index_scores, dim=2)
            if mask is not None:
                mask = self.cast(mask, mstype.float32)
                index_scores = index_scores + mask
            # Match MindSpeed's small-op contract: the second return value is
            # the selected top-k logits, not the full score matrix.  Besides
            # making the fused and unfused contracts consistent, this is
            # required by the DSV4 indexer KL formulation below.
            topk_scores, topk_indices = self.topk(index_scores, k=effective_topk, dim=-1)
            topk_indices = self.cast(topk_indices, mstype.int32)
            index_scores = topk_scores
        return topk_indices, index_scores


class _IndexerLossAutoScaler(_Function):
    """AutoScaler that triggers backward + scales the grad for indexer loss."""

    main_loss_backward_scale: Tensor = Tensor(1.0)

    @staticmethod
    def forward(ctx, output, indexer_loss):
        """Preserve `indexer_loss` on ctx; return `output` unchanged."""
        ctx.indexer_loss = indexer_loss
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Return (grad_output, ones_like(indexer_loss) * scale)."""
        indexer_loss = ctx.indexer_loss
        scale = _IndexerLossAutoScaler.main_loss_backward_scale
        indexer_loss = indexer_loss.to_local() if isinstance(indexer_loss, DTensor) else indexer_loss
        scaled_grad = mint.ones_like(indexer_loss) * scale
        return grad_output, scaled_grad

    @staticmethod
    def set_loss_scale(scale):
        """Set the backward scale (trainer/callback is responsible for reduce)."""
        _IndexerLossAutoScaler.main_loss_backward_scale = scale


class FusedCSAIndexerLoss(_Function):
    """Fused CSA Indexer loss"""

    @staticmethod
    def forward(ctx,
                query, key,
                query_index, key_index, weights, topk_indices,
                scale_value, layout, cmp_ratio, loss_coeff,
                softmax_max=None, softmax_sum=None,
                actual_seq_qlen=None, actual_seq_klen=None):
        """Forward pass: call fused op + stash indexer-input gradients on ctx."""
        d_query_index, d_key_index, d_weights, loss = npu_sparse_lightning_indexer_grad_kl_loss_v2(
            query, key, query_index, key_index, weights, topk_indices,
            softmax_max=softmax_max, softmax_sum=softmax_sum, scale_value=scale_value,
            actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen,
            layout=layout, sparse_mode=3,
            pre_tokens=9223372036854775807, next_tokens=9223372036854775807, cmp_ratio=cmp_ratio,
        )
        if layout == "TND":
            scale = query.shape[0]
        else:
            scale = query.shape[0] * query.shape[1]
        loss = loss * loss_coeff / scale
        ctx.d_query_index = d_query_index
        ctx.d_key_index = d_key_index
        ctx.d_weights = d_weights
        ctx.loss_coeff = loss_coeff
        ctx.scale = scale
        return loss[0] # the loss output has a shape (1,), so get its value.

    @staticmethod
    def backward(ctx, grad_loss):
        """Scale the three pre-computed indexer gradients by `grad_loss`."""
        grad_scale = grad_loss * ctx.loss_coeff / ctx.scale
        return (
            None, None,
            ctx.d_query_index * grad_scale,
            ctx.d_key_index * grad_scale,
            ctx.d_weights * grad_scale,
            None,
            None, None, None,
            None, None,
            None, None,
        )


class UnfusedCSAIndexerLoss(nn.Cell):
    """Indexer KL-divergence loss for the unfused CSA path."""

    def __init__(self, softmax_scale, loss_coeff=1.0, sparse_loss=True):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.loss_coeff = loss_coeff
        self.sparse_loss = sparse_loss

        self.post_head_sum = IdentityOp()

        self.permute = mint.permute
        self.matmul = mint.matmul
        self.softmax = mint.softmax
        self.sum = mint.sum
        self.mean = mint.mean
        self.log = mint.log
        self.reshape = mint.reshape
        self.unsqueeze = mint.unsqueeze
        self.full = mint.full
        self.triu = mint.triu
        self.scatter = mint.scatter
        self.where = mint.where
        self.any = mint.any
        self.max = mint.maximum
        self.cast = ops.cast

    def construct(
        self,
        index_scores,
        topk_indices,
        query,
        key,
        mask=None,
    ):
        """Compute the unfused indexer KL loss.

        Args:
            index_scores: ``[b, sq, topk]`` selected raw indexer logits.
            topk_indices: ``[b, sq, topk]`` selected compressed positions.
            query: ``[sq, b, np, v_head_dim]`` multi-head query (SBND).
            key: ``[sk, b, 1, v_head_dim]`` compressed KV (TND, single head).
            mask: optional ``[1, sq, sk]`` causal mask.  When ``None`` a
                standard upper-triangular causal mask is built.

        Returns:
            Scalar KL loss.
        """
        eps = 1.0e-9
        sq, b, _, _ = query.shape
        sk, _, _, _ = key.shape
        # MindSpeed computes the main-attention matmul in the incoming BF16
        # dtype, then requests an FP32 softmax.  Promoting the matmul itself to
        # FP32 shifts the KL loss by roughly 1e-4 on the alignment model.
        query = self.permute(query, (1, 2, 0, 3))   # [b, np, sq, d]
        key = self.permute(key, (1, 2, 3, 0))       # [b, 1, d, sk]
        attention_scores = self.matmul(query, key) * self.softmax_scale

        # Causal mask
        if mask is None:
            causal_mask = self.triu(
                self.full((sq, sk), float("-inf"), dtype=mstype.float32), diagonal=1
            )
            causal_mask = self.reshape(causal_mask, (1, 1, sq, sk))
        else:
            causal_mask = mask.to(mstype.float32)
            causal_mask = self.reshape(causal_mask, (-1, 1, sq, sk))

        max_valid_idx = sk - 1
        replacement = self.full(topk_indices.shape, max_valid_idx, dtype=topk_indices.dtype)
        clean_indices = self.where(topk_indices == -1, replacement, topk_indices)
        clean_indices = mint.clamp(clean_indices, 0, max_valid_idx).to(mstype.int64)

        # The main distribution used by MindSpeed is the sparse distribution:
        # its attention mask keeps exactly the selected compressed positions.
        # Build the same selection mask in the full compressed-key space.
        selection_mask = self.full((b, sq, sk), float("-inf"), dtype=mstype.float32)
        selection_mask = self.scatter(selection_mask, -1, clean_indices, 0.0)

        # MindSpeed applies a finite BF16-min mask before its FP32 softmax.
        # Using -inf changes the all-masked prefix rows from a uniform
        # distribution into NaNs and requires a non-reference special case.
        invalid = (causal_mask < 0) | (self.unsqueeze(selection_mask, 1) < 0)
        bf16_min = self.full(attention_scores.shape, _BF16_MIN,
                             dtype=attention_scores.dtype)
        attention_scores = self.where(invalid, bf16_min, attention_scores)
        attention_scores = self.softmax(self.cast(attention_scores, mstype.float32), dim=-1)
        attention_scores = self.sum(attention_scores, dim=1)   # [b, sq, sk] — partial under TP
        attention_scores = self.post_head_sum(attention_scores)

        # Exact DSV4 reference formulation: L1-normalize the full target,
        # gather only selected positions, renormalize those for log(p), but
        # weight the KL terms by their pre-selection-normalized target mass.
        attention_scores = attention_scores / self.max(
            self.sum(attention_scores, -1, keepdim=True), 1.0e-12
        )
        selected_target = mint.gather(attention_scores, -1, clean_indices)

        # Prefix queries can have no valid compressed key, so top-k returns an
        # all--inf score row. Convert only those masked entries to a finite
        # sentinel before softmax; otherwise the indexer loss and grad norm
        # become NaN even though the row carries no valid target mass.
        finite_min = self.full(
            index_scores.shape,
            _BF16_MIN,
            dtype=index_scores.dtype,
        )
        index_scores = self.where(mint.isfinite(index_scores), index_scores, finite_min)
        predicted = self.softmax(self.cast(index_scores, mstype.float32), dim=-1)
        target_sum = self.sum(selected_target, dim=-1, keepdim=True)
        normalized_target = selected_target / (target_sum + eps)
        log_target = self.log(self.max(normalized_target, eps))
        log_predicted = self.log(predicted + eps)
        kl_div = self.mean(self.sum((log_target - log_predicted) * selected_target, dim=-1))

        return kl_div * self.loss_coeff
