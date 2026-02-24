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
"""DSA Lightning Indexer."""
import copy
from dataclasses import dataclass
from typing import Union
from scipy.linalg import hadamard

import mindspore.common.dtype as mstype
from mindspore import nn, ops, Tensor
from mindspore.ops import operations as P
from mindspore.ops import auto_generate as aclnn_ops
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import MLATransformerConfig, TransformerConfig
from mindformers.parallel_core.training_graph.base_models.common.embeddings.rope_utils import ApplyRotaryPosEmb
from mindformers.parallel_core.training_graph.device_matrix import layout
from mindformers.parallel_core.training_graph.communication import get_dp_cp_id
from mindformers.parallel_core.training_graph.transformer.mask_generate import CausalEODMaskGenerate


class Hadamard(nn.Cell):
    """Hadamard Transform."""
    def __init__(self, config: TransformerConfig, for_k_emb=False):
        super().__init__()
        self.hadamard_mat = Tensor(hadamard(config.dsa_indexer_head_dim), mstype.float32)
        self.scale = Tensor([config.dsa_indexer_head_dim ** -0.5], mstype.float32)
        self.for_k_emb = for_k_emb
        self.bmm = aclnn_ops.BatchMatMul()
        self.cast = aclnn_ops.Cast()
        self.mul = aclnn_ops.Mul()
        self.shard()

    def construct(self, x):
        """do hadamard transform."""
        hadamard_mat = self.cast(self.hadamard_mat, x.dtype)
        scale = self.cast(self.scale, x.dtype)
        x = self.bmm(x, hadamard_mat)
        x = self.mul(x, scale)
        return x

    def shard(self):
        """Set parallel strategy."""
        bmm_shard = layout("cp", "dp", "None", "None") if self.for_k_emb else layout("cp", "dp", "tp", "None")
        self.bmm.shard((bmm_shard, layout("None", "None")))
        self.mul.shard((bmm_shard, layout("None")))


@dataclass
class DSAIndexerSubmodules:
    """
    Configuration class for specifying the submodules of an DSA Indexer.

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


class DSAIndexer(nn.Cell):
    """
    DSA Lightning Indexer for DeepSeek Sparse Attention.
    Computes index scores to identify the top-k most relevant key-value pairs for each query in sparse attention.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L431-L480

    Args:
        config (TransformerConfig): The configuration for the transformer model.
        submodules (DSAIndexerSubmodules): Indexer submodules specification.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: DSAIndexerSubmodules,
    ) -> None:
        """Initialize the indexer."""
        super().__init__()
        self.config = config
        self.cp = config.context_parallel_size
        self.input_layout = config.input_layout
        self.is_tnd = self.input_layout == "TND"
        self.hidden_size = config.hidden_size
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim
        self.q_lora_rank =  config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size

        _, cp_id = get_dp_cp_id(config)
        self.offset_id = cp_id

        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk
        self.sparse_loss = config.dsa_indexer_use_sparse_loss
        self.softmax_scale: float = self.index_head_dim ** -0.5

        self.apply_rotary_emb_q = ApplyRotaryPosEmb(config)
        self.apply_rotary_emb_k = ApplyRotaryPosEmb(config, for_k_pos_emb=True)
        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            input_size=self.q_lora_rank,
            output_size=self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        )

        self.linear_wk = build_module(
            submodules.linear_wk,
            self.hidden_size,
            self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        )

        k_norm_config = copy.copy(config)
        k_norm_config.normalization = "LayerNorm"
        self.k_norm = build_module(
            submodules.k_norm,
            config=k_norm_config,
            dim=self.index_head_dim,
            eps=config.layernorm_epsilon,
        )

        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        )

        self.eod_mask_generator = CausalEODMaskGenerate(config)
        # preprocess operators
        self.split_q = aclnn_ops.SplitWithSize()
        self.split_k = aclnn_ops.SplitWithSize()
        self.cat_q = aclnn_ops.Concat(-1).add_prim_attr("self_define_shard", True)
        self.cat_k = aclnn_ops.Concat(-1).add_prim_attr("self_define_shard", True)
        self.cast = aclnn_ops.Cast()
        self.reshape = aclnn_ops.Reshape()
        self.transpose_q = aclnn_ops.Transpose()
        self.transpose_k = aclnn_ops.Transpose()
        self.transpose_w = aclnn_ops.Transpose()
        # sparse indexer operators
        self.clamp= aclnn_ops.ClampScalar()
        self.bmm = aclnn_ops.BatchMatMul()
        self.relu = aclnn_ops.ReLU()
        self.roll = aclnn_ops.Roll(1)
        self.cumsum = aclnn_ops.CumsumExt()
        self.compute_sparse_indices = P.Morph(
            self._compute_sparse_indices_forward,
            self.indexer_infer_shape,
            lambda *args: (mstype.int32, args[0])
        ).add_prim_attr("self_define_shard", True)
        # dense indexer operators
        self.mul = aclnn_ops.Mul()
        self.mul_scalar = aclnn_ops.Mul()
        self.sum = aclnn_ops.SumExt().add_prim_attr("self_define_shard", True)
        self.add = aclnn_ops.AddExt()
        self.topk = aclnn_ops.TopkExt().add_prim_attr("self_define_shard", True)
        self.slice = aclnn_ops.SliceExt().add_prim_attr("self_define_shard", True)
        self.hadamard_q = Hadamard(config)
        self.hadamard_k = Hadamard(config, for_k_emb=True)

        self.inf = Tensor([-10000], mstype.float32)
        if _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
            self.shard()

    def indexer_infer_shape(self, *args):
        """indexer infer shape"""
        q_shape = args[0]
        k_shape = args[1]
        infer_shape = q_shape[:-2] + [k_shape[-2], self.index_topk]
        return infer_shape, infer_shape

    def compute_dense_indices(self, q, k, weights, mask, actual_seq_len=None):
        """Compute full index score and topk indices with unfused operators in dense warmup stage."""
        bsz, seqlen_q = q.shape[:2]
        seqlen_k = k.shape[-1]
        if self.is_tnd:
            mask = self.eod_mask_generator(self.reshape(actual_seq_len, (bsz, -1)))
        # Compute attention scores: q @ k^T
        # [batch, seqlen_q, index_n_heads, index_head_dim] @ [batch, 1, index_head_dim, seqlen_k]
        #   -> [batch, seqlen_q, index_n_heads, seqlen_k]
        q = self.cast(q, mstype.float32)
        k = self.cast(k, mstype.float32)
        index_scores = self.bmm(q, k)

        # Apply ReLU activation.
        index_scores = self.relu(index_scores)

        # Weight each head by attention weights.
        # [batch, seqlen_q, index_n_heads, seqlen_k] * [batch, seqlen_q, index_n_heads, 1]
        #   -> [batch, seqlen_q, index_n_heads, seqlen_k]
        index_scores = self.mul(index_scores, weights)

        # Sum across attention heads.
        # [batch, seqlen_q, index_n_heads, seqlen_k] -> [batch, seqlen_q, 1, seqlen_k]
        mask = self.cast(self.reshape(mask, (bsz, seqlen_q, 1, -1)), mstype.float32)
        index_scores = self.sum(index_scores, 2, True)
        index_scores = self.add(index_scores, self.mul_scalar(mask, self.inf))

        topk_indices = self.cast(self.topk(index_scores, self.index_topk, -1)[1], mstype.int32)
        mask_sliced = self.cast(self.slice(mask, 3, 0, self.index_topk, 1) * -seqlen_k, mstype.int32)
        topk_indices = self.clamp(self.add(topk_indices, mask_sliced), -1)
        if self.is_tnd:
            topk_indices = self.reshape(topk_indices, (bsz * seqlen_q, 1, -1))
            index_scores = self.reshape(index_scores, (bsz * seqlen_q, 1, -1))
        return topk_indices, index_scores

    def _compute_sparse_indices_forward(self, q, k, weights, actual_seq_qlen=None, actual_seq_klen=None):
        """Compute topk index score and indices with fused operator in sparse training stage."""
        if self.is_tnd:
            slice_tq = q.shape[0]
            slice_tk = k.shape[0]
            offset_q = slice_tq * self.offset_id
            # process actual_seq_len for fused operator to auto-generate mask in individual shard
            new_actual_seq_qlen = self.cast(self.clamp(actual_seq_qlen - offset_q, 0, slice_tq), mstype.int32)
            new_actual_seq_klen = actual_seq_klen - self.relu(actual_seq_qlen - offset_q) + new_actual_seq_qlen
            prev_seq_klen = self.roll(new_actual_seq_klen)
            prev_seq_klen[0] = 0
            new_actual_seq_klen = self.cumsum(self.relu(new_actual_seq_klen - prev_seq_klen), 0)
            new_actual_seq_klen[-1] = slice_tk
            actual_seq_qlen = new_actual_seq_qlen
            actual_seq_klen = new_actual_seq_klen

        return ops.lightning_indexer(
            q, k, weights,
            actual_seq_lengths_query=actual_seq_qlen,
            actual_seq_lengths_key=actual_seq_klen,
            layout_query=self.input_layout,
            layout_key=self.input_layout,
            sparse_count=self.index_topk,
            return_value=True
        )

    def compute_index_scores(self, q, k, weights, mask, actual_seq_qlen=None, actual_seq_klen=None):
        """Compute topk index score and indices"""
        if self.sparse_loss:
            return self.compute_sparse_indices(q, k, weights, actual_seq_qlen, actual_seq_klen)
        return self.compute_dense_indices(q, k, weights, mask, actual_seq_qlen)

    def construct(self, x, qr, mask, rotary_pos_emb, actual_seq_qlen=None, actual_seq_klen=None):
        """Forward pass for DSA Indexer that returns both index scores and top-k indices."""
        # =========================================
        # Get sequence length and batch size
        # =========================================
        seqlen, bsz, _ = x.shape

        # =========================================
        # q linear and apply rope to q
        # =========================================
        # [seqlen, batch, q_lora_rank] -> [seqlen, batch, index_n_heads * index_head_dim]
        q, _ = self.linear_wq_b(qr)
        # [seqlen, batch, index_n_heads * index_head_dim] -> [seqlen, batch, index_n_heads, index_head_dim]
        q = self.reshape(q, (seqlen, bsz, self.index_n_heads, self.index_head_dim))
        q_nope, q_pe = self.split_q(
            q, [self.index_head_dim - self.qk_pos_emb_head_dim, self.qk_pos_emb_head_dim], dim=-1
        )
        q_pe = self.apply_rotary_emb_q(
            q_pe,
            rotary_pos_emb,
            rotary_interleaved=self.config.rotary_interleaved,
            multi_latent_attention=self.config.multi_latent_attention
        )
        # [seqlen, batch, *, index_head_dim]
        q = self.cat_q([q_nope, q_pe])

        # =========================================
        # k linear and apply rope to k
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, 1, index_head_dim]
        k, _ = self.linear_wk(x)
        k = self.k_norm(k)
        # [seqlen, batch, index_head_dim] -> [seqlen, batch, 1, index_head_dim]
        k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
        # [seqlen, batch, 1, index_head_dim] -> [seqlen, batch, index_head_dim]
        k_nope, k_pe = self.split_k(
            k, [self.index_head_dim - self.qk_pos_emb_head_dim, self.qk_pos_emb_head_dim], dim=-1
        )
        k_pe = self.apply_rotary_emb_k(
            k_pe,
            rotary_pos_emb,
            rotary_interleaved=self.config.rotary_interleaved,
            multi_latent_attention=self.config.multi_latent_attention
        )
        # [seqlen, batch, *, index_head_dim]
        k = self.cat_k([k_nope, k_pe])

        # =========================================
        # Rotate activation
        # =========================================
        q = self.hadamard_q(q)
        k = self.hadamard_k(k)

        # =========================================
        # Compute index scores
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_n_heads]
        weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.index_n_heads ** -0.5) * self.softmax_scale

        # convert q, k, weights from SB* layout to BS* or T* layout
        weights = self.transpose_w(weights, (1, 0, 2))
        q = self.transpose_q(q, (1, 0, 2, 3))
        if self.sparse_loss:
            k = self.transpose_k(k, (1, 0, 2, 3))
            if self.is_tnd:
                q = self.reshape(q, (seqlen * bsz, self.index_n_heads, self.index_head_dim))
                k = self.reshape(k, (seqlen * bsz, 1, self.index_head_dim))
                weights = self.reshape(weights, (seqlen * bsz, self.index_n_heads))
        else:
            # Currently, unfused bsnd indexer computation method will be applied to both tnd layout and bsnd layout
            # during dense warmup stage.
            k = self.transpose_k(k, (1, 2, 3, 0))
            k = self.reshape(k, (bsz, 1, self.index_head_dim, seqlen))
            weights = self.reshape(weights, (bsz, seqlen, self.index_n_heads, 1))

        topk_indices, index_scores = self.compute_index_scores(q, k, weights, mask, actual_seq_qlen, actual_seq_klen)
        return index_scores, topk_indices

    def shard(self):
        """Set parallel strategy."""
        # before convert_layout
        q_shard = layout("cp", "dp", "tp", "None")
        k_shard = layout("cp", "dp", "None", "None")
        w_shard = layout("cp", "dp", "None")
        self.split_q.shard((q_shard,))
        self.split_k.shard((k_shard,))
        self.cat_q.shard(((q_shard, q_shard),), (q_shard,))
        self.cat_k.shard(((k_shard, k_shard),), (k_shard,))
        self.transpose_q.shard((q_shard,))
        self.transpose_k.shard((k_shard,))
        self.transpose_w.shard((w_shard,))

        # after convert_layout
        if self.sparse_loss and self.is_tnd:
            q_shard = layout("dp_cp", "None", "None")
            k_shard = layout("dp", "None", "None")
            w_shard = layout("dp_cp", "None")
            actual_seq_shard = layout("dp")
            indexer_shard = layout("dp_cp", "None", "None")
            self.compute_sparse_indices.shard(
                (q_shard, k_shard, w_shard, actual_seq_shard, actual_seq_shard), (indexer_shard, indexer_shard)
            )
        elif self.sparse_loss:
            q_shard = layout("dp", "None", "None", "None")
            k_shard = layout("dp", "None", "None", "None")
            w_shard = layout("dp", "None", "None")
            indexer_shard = layout("dp", "None", "None", "None")
            self.compute_sparse_indices.shard((q_shard, k_shard, w_shard), (indexer_shard, indexer_shard))
        else:
            q_shard = layout("dp", "cp", "tp", "None")
            k_shard = layout("dp", "None", "None", "None")
            w_shard = layout("dp", "cp", "tp", "None")
            qk_shard = layout("dp", "cp", "tp", "None")
            sum_shard = layout("dp", "cp", "None", "None")
            score_shard = layout("dp", "cp", "None", "None")
            self.bmm.shard((q_shard, k_shard), (qk_shard,))
            self.relu.shard((qk_shard,))
            self.mul.shard((qk_shard, w_shard))
            self.sum.shard((sum_shard,), (sum_shard,))
            self.mul_scalar.shard((score_shard, layout("None")))
            self.add.shard((score_shard, score_shard))
            self.topk.shard((score_shard,), (score_shard, score_shard))
            self.slice.shard((score_shard,), (score_shard,))
            self.clamp.shard((score_shard,))
