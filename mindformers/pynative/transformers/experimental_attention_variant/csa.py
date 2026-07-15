# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright 2026 Huawei Technologies Co., Ltd
#
# This file is derived from Megatron-LM and adapted for MindSpore.
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
"""CompressedSparseAttention"""
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

from mindspore import nn, Tensor, mint, ops
from mindspore import dtype as mstype
from mindspore.common._grad_function import _Function
from mindspore.common.parameter import Parameter

try:
    from hyper_parallel.custom_ops.experimental import (
        npu_sparse_flash_mla,
        npu_sparse_flash_mla_grad,
        npu_sparse_lightning_indexer_kl_loss_grad,
    )
except ImportError:
    npu_sparse_flash_mla = None
    npu_sparse_flash_mla_grad = None
    npu_sparse_lightning_indexer_kl_loss_grad = None

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.pynative.transformers.experimental_attention_variant.indexer import (
    _IndexerLossAutoScaler,
    UnfusedCSAIndexerLoss,
)
from mindformers.pynative.transformers.experimental_attention_variant.utils import  save_to_indexer_losses_tracker


def _prepare_sparse_flash_mla(
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        cmp_ratio,
):
    """Prepare compressed indices and residual for SparseFlashMla."""
    ori_length = ori_kv.shape[1]
    has_cmp = cmp_kv is not None
    if has_cmp and cmp_ratio <= 0:
        raise ValueError(f"cmp_ratio must be positive when cmp_kv is present, but got {cmp_ratio}.")
    kernel_cmp_ratio = cmp_ratio if has_cmp else 1
    use_sparse_indices = has_cmp and cmp_ratio == 4 and cmp_sparse_indices is not None
    sparse_indices = mint.unsqueeze(cmp_sparse_indices, dim=2) if use_sparse_indices else None
    # MindSpeed materializes a zero residual tensor even when no compressed-KV
    # branch is present (``cmp_ratio == 0`` is normalized to one first).  The
    # SparseFlashMLA forward is bit-identical with ``None``, but its backward
    # takes a slightly different kernel path and changes a few dKV elements.
    cmp_residual = Tensor(
        [int(ori_length) % kernel_cmp_ratio],
        dtype=mstype.int32,
    )
    return sparse_indices, cmp_residual, kernel_cmp_ratio


class FusedSparseFlashMla(_Function):
    """SparseFlashMla custom autograd for BSND DSV4 layers without indexer loss."""

    @staticmethod
    def forward(
            ctx,
            query,
            ori_kv,
            cmp_kv,
            cmp_sparse_indices,
            sinks,
            softmax_scale,
            cmp_ratio,
            ori_mask_mode,
            cmp_mask_mode,
            ori_win_left,
            ori_win_right,
    ):
        """Run SparseFlashMla forward and retain inputs for its custom backward."""
        sparse_indices, cmp_residual, kernel_cmp_ratio = _prepare_sparse_flash_mla(
            ori_kv, cmp_kv, cmp_sparse_indices, cmp_ratio,
        )
        output, softmax_lse = npu_sparse_flash_mla(
            query,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=sparse_indices,
            cmp_residual_kv=cmp_residual,
            sinks=sinks,
            softmax_scale=softmax_scale,
            cmp_ratio=kernel_cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout="BSND",
            return_softmax_lse=True,
        )
        ctx.query = query
        ctx.ori_kv = ori_kv
        ctx.cmp_kv = cmp_kv
        ctx.sparse_indices = sparse_indices
        ctx.cmp_residual = cmp_residual
        ctx.sinks = sinks
        ctx.output = output
        ctx.softmax_lse = softmax_lse
        ctx.softmax_scale = softmax_scale
        ctx.cmp_ratio = kernel_cmp_ratio
        ctx.ori_mask_mode = ori_mask_mode
        ctx.cmp_mask_mode = cmp_mask_mode
        ctx.ori_win_left = ori_win_left
        ctx.ori_win_right = ori_win_right
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Run SparseFlashMlaGrad and return gradients for differentiable inputs."""
        d_query, d_ori_kv, d_cmp_kv, d_sinks, _, _ = npu_sparse_flash_mla_grad(
            ctx.query,
            grad_output.contiguous(),
            ctx.output,
            ctx.softmax_lse,
            ori_kv=ctx.ori_kv,
            cmp_kv=ctx.cmp_kv,
            cmp_sparse_indices=ctx.sparse_indices,
            cmp_residual_kv=ctx.cmp_residual,
            sinks=ctx.sinks,
            softmax_scale=ctx.softmax_scale,
            cmp_ratio=ctx.cmp_ratio,
            ori_mask_mode=ctx.ori_mask_mode,
            cmp_mask_mode=ctx.cmp_mask_mode,
            ori_win_left=ctx.ori_win_left,
            ori_win_right=ctx.ori_win_right,
            layout="BSND",
        )
        return (
            d_query, d_ori_kv, d_cmp_kv if ctx.cmp_kv is not None else None,
            None, d_sinks, None, None, None, None, None, None,
        )


def _compute_fused_indexer_loss(attn_softmax_l1, indexer_softmax, eps=1.0e-9):
    """Match MindSpeed's source-level fused DSV4 indexer loss formula."""
    norm = mint.sum(mint.abs(attn_softmax_l1), dim=-1, keepdim=True)
    target = attn_softmax_l1 / mint.clamp(norm, min=1.0e-12)
    target_sum = mint.sum(target, dim=-1, keepdim=True)
    normalized_target = target / (target_sum + eps)
    log_target = mint.log(mint.clamp(normalized_target, min=eps))
    log_predicted = mint.log(indexer_softmax + eps)
    return mint.mean(mint.sum((log_target - log_predicted) * target, dim=-1))


class FusedSparseFlashMlaWithIndexerLoss(_Function):
    """MindSpeed-equivalent SparseFlashMla and indexer-loss combined autograd."""

    @staticmethod
    def forward(
            ctx,
            query,
            ori_kv,
            cmp_kv,
            cmp_sparse_indices,
            query_index,
            key_index,
            weights,
            sinks,
            softmax_scale,
            cmp_ratio,
            ori_mask_mode,
            cmp_mask_mode,
            ori_win_left,
            ori_win_right,
            loss_coeff,
            layer_number,
            num_layers,
    ):
        """Run fused attention and save everything required by the combined backward."""
        sparse_indices, cmp_residual, kernel_cmp_ratio = _prepare_sparse_flash_mla(
            ori_kv, cmp_kv, cmp_sparse_indices, cmp_ratio,
        )
        output, softmax_lse = npu_sparse_flash_mla(
            query,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=sparse_indices,
            cmp_residual_kv=cmp_residual,
            sinks=sinks,
            softmax_scale=softmax_scale,
            cmp_ratio=kernel_cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout="BSND",
            return_softmax_lse=True,
        )
        ctx.query = query
        ctx.ori_kv = ori_kv
        ctx.cmp_kv = cmp_kv
        ctx.sparse_indices = sparse_indices
        ctx.query_index = query_index
        ctx.key_index = key_index
        ctx.weights = weights
        ctx.cmp_residual = cmp_residual
        ctx.sinks = sinks
        ctx.output = output
        ctx.softmax_lse = softmax_lse
        ctx.softmax_scale = softmax_scale
        ctx.cmp_ratio = kernel_cmp_ratio
        ctx.ori_mask_mode = ori_mask_mode
        ctx.cmp_mask_mode = cmp_mask_mode
        ctx.ori_win_left = ori_win_left
        ctx.ori_win_right = ori_win_right
        ctx.loss_coeff = loss_coeff
        ctx.layer_number = layer_number
        ctx.num_layers = num_layers
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Fuse SparseFlashMlaGrad with SparseLightningIndexerKLLossGrad."""
        d_query, d_ori_kv, d_cmp_kv, d_sinks, _, cmp_softmax_l1 = npu_sparse_flash_mla_grad(
            ctx.query,
            grad_output.contiguous(),
            ctx.output,
            ctx.softmax_lse,
            ori_kv=ctx.ori_kv,
            cmp_kv=ctx.cmp_kv,
            cmp_sparse_indices=ctx.sparse_indices,
            cmp_residual_kv=ctx.cmp_residual,
            sinks=ctx.sinks,
            softmax_scale=ctx.softmax_scale,
            cmp_ratio=ctx.cmp_ratio,
            ori_mask_mode=ctx.ori_mask_mode,
            cmp_mask_mode=ctx.cmp_mask_mode,
            ori_win_left=ctx.ori_win_left,
            ori_win_right=ctx.ori_win_right,
            layout="BSND",
        )
        batch_size, query_length = ctx.query.shape[:2]
        d_query_index, d_key_index, d_weights, indexer_softmax = (
            npu_sparse_lightning_indexer_kl_loss_grad(
                ctx.query_index,
                ctx.key_index,
                ctx.weights,
                ctx.sparse_indices,
                cmp_softmax_l1,
                cmp_residual_k=ctx.cmp_residual,
                layout="BSND",
                mask_mode=ctx.cmp_mask_mode,
                cmp_ratio=ctx.cmp_ratio,
            )
        )
        grad_scale = (
            _IndexerLossAutoScaler.main_loss_backward_scale
            * ctx.loss_coeff
            / (batch_size * query_length)
        )
        d_query_index = d_query_index * grad_scale
        d_key_index = d_key_index * grad_scale
        d_weights = d_weights * grad_scale
        indexer_loss = _compute_fused_indexer_loss(cmp_softmax_l1, indexer_softmax)
        save_to_indexer_losses_tracker(
            indexer_loss * ctx.loss_coeff,
            ctx.layer_number,
            ctx.num_layers,
        )
        return (
            d_query,
            d_ori_kv,
            d_cmp_kv if ctx.cmp_kv is not None else None,
            None,
            d_query_index,
            d_key_index,
            d_weights,
            d_sinks,
            None, None, None, None, None, None, None, None, None,
        )


def parse_cu_seqlens(actual_seq_len, t_global):
    """Reduce a padded cumulative doc-end vector to host cu_seqlens."""
    if actual_seq_len is None:
        return None
    ends = np.unique(actual_seq_len.reshape(-1).asnumpy()).tolist()
    ends = [int(e) for e in ends if 0 < int(e) <= t_global]
    if not ends or ends[-1] != t_global:
        ends.append(t_global)
    return ends if len(ends) > 1 else None


def get_doc_window_topk_idxs(window_size, batch_size, seqlen, cu_seqlens,
                             pos_offset=0, ori_halo_len=0):
    """Doc-aware sliding-window indices (no-pad TND, RFC nopad-tnd §2.3)."""
    ends = list(cu_seqlens)
    starts = [0] + ends[:-1]

    def doc_start(p):
        for st, en in zip(starts, ends):
            if st <= p < en:
                return st
        return p  # tail pad: degenerate one-token doc

    idx = np.full((seqlen, window_size), -1, np.int32)
    for i in range(seqlen):
        p = pos_offset + i
        lo = max(doc_start(p), p - window_size + 1)
        n = p - lo + 1
        idx[i, :n] = np.arange(lo, p + 1, dtype=np.int32) - pos_offset + ori_halo_len
    t = Tensor(idx, mstype.int32)
    return mint.unsqueeze(t, 0).broadcast_to((batch_size, seqlen, window_size))


def build_grid_valid_blocks(ratio, cu_seqlens):
    """Valid global grid blocks (END-based rule, RFC nopad-tnd §2.3)."""
    ends = list(cu_seqlens)
    starts = [0] + ends[:-1]
    nblocks = ends[-1] // ratio
    valid = np.zeros(nblocks, bool)
    block_doc = np.full(nblocks, -1, np.int32)
    for g in range(nblocks):
        g0 = g * ratio
        for d, (st, en) in enumerate(zip(starts, ends)):
            if st <= g0 < en:
                block_doc[g] = d
                valid[g] = g0 + ratio <= en
                break
    return valid, block_doc


def get_doc_compress_topk_idxs(ratio, batch_size, seqlen, offset, cu_seqlens,
                               pos_offset=0, compact=False, n_compressed=None):
    """Doc-aware dense compressed indices (no-pad TND, global-grid path G)."""
    valid, block_doc = build_grid_valid_blocks(ratio, cu_seqlens)
    ends = list(cu_seqlens)
    starts = [0] + ends[:-1]

    def doc_of(p):
        for d, (st, en) in enumerate(zip(starts, ends)):
            if st <= p < en:
                return d
        return -1

    nblocks = len(valid)
    width = n_compressed if n_compressed is not None else nblocks
    slot = np.cumsum(valid) - 1 if compact else np.arange(nblocks)
    idx = np.full((seqlen, width), -1, np.int32)
    anchors = (np.arange(nblocks, dtype=np.int32) + 1) * ratio
    for i in range(seqlen):
        p = pos_offset + i
        vis = valid & (block_doc == doc_of(p)) & (anchors <= p + 1)
        idx[i, :nblocks][vis] = slot[vis].astype(np.int32) + offset
    t = Tensor(idx, mstype.int32)
    return mint.unsqueeze(t, 0).broadcast_to((batch_size, seqlen, width))


def build_doc_block_mask(ratio, seqlen, cu_seqlens, pos_offset=0, n_compressed=None,
                         compact=False):
    """Doc-aware additive mask over compressed blocks (no-pad TND, path G)."""
    valid, block_doc = build_grid_valid_blocks(ratio, cu_seqlens)
    nblocks = len(valid)
    n_compressed = nblocks if n_compressed is None else n_compressed
    ends = list(cu_seqlens)
    starts = [0] + ends[:-1]

    def doc_of(p):
        for d, (st, en) in enumerate(zip(starts, ends)):
            if st <= p < en:
                return d
        return -1

    anchors = (np.arange(nblocks, dtype=np.int64) + 1) * ratio
    grid = np.full((seqlen, nblocks), -np.inf, np.float32)
    for i in range(seqlen):
        p = pos_offset + i
        vis = valid & (block_doc == doc_of(p)) & (anchors <= p + 1)
        grid[i][vis] = 0.0
    mask = np.full((seqlen, n_compressed), -np.inf, np.float32)
    if compact:
        # CP select-and-pad space: only valid blocks survive, compacted in
        # global order; pad tail stays -inf.
        nvalid = int(valid.sum())
        mask[:, :nvalid] = grid[:, valid]
    else:
        mask[:, :nblocks] = grid
    return Tensor(mask, mstype.float32)


def get_compress_topk_idxs(ratio, batch_size, seqlen, offset):
    """Dense compressed-position indices for the naive CSA path."""
    n_compressed = seqlen // ratio
    # matrix: [seqlen, n_compressed], each row = arange(n_compressed).
    matrix = mint.tile(
        mint.unsqueeze(mint.arange(n_compressed, dtype=mstype.int32), 0), (seqlen, 1)
    )
    # mask: matrix >= ((arange(1, seqlen+1)) // ratio)[:, None].
    visible = mint.unsqueeze(
        mint.arange(1, seqlen + 1, dtype=mstype.int32) // ratio, 1
    )
    mask = matrix >= visible
    neg_one = mint.full(matrix.shape, -1, dtype=mstype.int32)
    matrix = mint.where(mask, neg_one, matrix + offset)
    # Broadcast over batch.
    matrix = mint.unsqueeze(matrix, 0).broadcast_to((batch_size, seqlen, n_compressed))
    return matrix


def get_window_topk_idxs(window_size, batch_size, seqlen):
    """Sliding-window KV indices for the naive CSA path."""
    # base: [seqlen, 1]; offsets: [window_size].
    base = mint.unsqueeze(mint.arange(seqlen, dtype=mstype.int32), 1)
    offsets = mint.arange(window_size, dtype=mstype.int32)
    # matrix = clamp(base - window_size + 1, min=0) + offsets.
    matrix = mint.clamp(base - window_size + 1, min=0) + offsets
    # Mask positions that point past the query.
    neg_one = mint.full(matrix.shape, -1, dtype=mstype.int32)
    matrix = mint.where(matrix > base, neg_one, matrix)
    # Broadcast over batch: [seqlen, window_size] -> [batch, seqlen, window_size].
    matrix = mint.unsqueeze(matrix, 0).broadcast_to((batch_size, seqlen, window_size))
    return matrix


def unfused_compressed_sparse_attn(query, kv_full, attn_sink, topk_indices, softmax_scale):
    """Differentiable sparse attention with shared (MQA-style) KV and a sink."""
    sq, b, n, d = query.shape
    if hasattr(attn_sink, "to_local"):
        attn_sink = attn_sink.to_local()

    # --- Gather KV at topk positions
    # kv_full: [sk, b, 1, d] -> [b, 1, sk, d]
    kv_t = mint.permute(kv_full, (1, 2, 0, 3))

    safe_indices = ops.cast(mint.clamp(topk_indices, min=0), mstype.int64)  # [b, sq, topk]
    topk = safe_indices.shape[-1]

    # Flatten KV to [b*sk, d] and gather via flat index lookup.
    # This avoids materializing kv_expanded [b, sq, sk, d] (up to 40 GB fp32 at sq=4096, ratio=4)
    # and safe_indices_exp [b, sq, topk, d].
    # Uses only fundamental tensor indexing — no deprecated APIs.
    sk = kv_full.shape[0]
    kv_flat = mint.reshape(kv_t, (b * sk, d))                     # [b*sk, d]
    batch_offset = mint.arange(b, dtype=mstype.int64).unsqueeze(1).unsqueeze(2) * sk
    flat_indices = mint.reshape(safe_indices + batch_offset, (-1,))
    kv_gathered = mint.reshape(kv_flat[flat_indices], (b, sq, topk, d))

    # --- Attention scores
    # query: [sq, b, n, d] -> [b, n, sq, d]
    q = ops.cast(mint.permute(query, (1, 2, 0, 3)), mstype.float32)
    kv_g = ops.cast(kv_gathered, mstype.float32)  # [b, sq, topk, d]

    # [b, n, sq, topk] = bnsh,bskh->bnsk
    topk = kv_g.shape[2]
    q_bm = mint.reshape(mint.permute(q, (0, 2, 1, 3)), (b * sq, n, d))  # [b*sq, n, d]
    kv_bm = mint.reshape(mint.permute(kv_g, (0, 1, 3, 2)), (b * sq, d, topk))  # [b*sq, d, topk]
    scores = mint.reshape(mint.bmm(q_bm, kv_bm), (b, sq, n, topk))  # [b, sq, n, topk]
    scores = mint.permute(scores, (0, 2, 1, 3)) * softmax_scale  # [b, n, sq, topk]

    # --- Mask invalid (topk_indices < 0)
    # invalid: [b, sq, topk] -> [b, 1, sq, topk]
    invalid_mask = mint.unsqueeze(topk_indices < 0, 1)
    scores = scores + ops.cast(invalid_mask, scores.dtype) * -1e30

    # --- Softmax with attention sink ---
    # attn_sink shape [n] -> [1, n, 1, 1]
    sink = ops.cast(mint.reshape(attn_sink, (1, n, 1, 1)), mstype.float32)
    # scores_max [b, n, sq, 1]; combine with sink for stable softmax
    scores_max, _ = mint.max(scores, dim=-1, keepdim=True)
    scores_max = mint.maximum(scores_max, sink)

    exp_scores = mint.exp(scores - scores_max)        # [b, n, sq, topk]
    exp_sink = mint.exp(sink - scores_max)            # [1, n, 1, 1]
    sum_exp = mint.sum(exp_scores, dim=-1, keepdim=True) + exp_sink
    attn_weights = exp_scores / sum_exp               # [b, n, sq, topk]

    # --- Weighted sum ---
    aw_bm = mint.reshape(mint.permute(attn_weights, (0, 2, 1, 3)), (b * sq, n, topk))  # [b*sq, n, topk]
    kvo_bm = mint.reshape(kv_g, (b * sq, topk, d))  # [b*sq, topk, d]
    output = mint.reshape(mint.bmm(aw_bm, kvo_bm), (b, sq, n, d))  # [b, sq, n, d]
    output = mint.permute(output, (0, 2, 1, 3))
    output = ops.cast(output, query.dtype)

    # [b, n, sq, d] -> [sq, b, n, d]
    output = mint.permute(output, (2, 0, 1, 3))
    return output


@dataclass
class CompressedSparseAttentionSubmodules:
    """Submodule specs for the DSv4 hybrid CompressedSparseAttention."""

    compressor: Union[ModuleSpec, type] = None
    indexer: Union[ModuleSpec, type] = None


class CompressedSparseAttention(nn.Cell):
    """
    DeepSeek-V4 hybrid core attention — window + compressed + attention sink.

    Args:
        config: MLATransformerConfig.
        submodules: CompressedSparseAttentionSubmodules with ``compressor`` / ``indexer`` specs.
        layer_number: Number which indicates the index of this transformer layer in the whole transformer block.
        softmax_scale: Optional softmax-scale override; defaults to ``config.v_head_dim ** -0.5``.
        rotary_pos_emb: Optional RoPE Cell for compressor and indexer. Not consumed directly here.
        compress_ratio: Per-layer compress ratio from ``config.csa_compress_ratios[layer_number-1]``.
    """

    def __init__(
            self,
            config: MLATransformerConfig,
            submodules: CompressedSparseAttentionSubmodules,
            layer_number: int,
            softmax_scale: Optional[float] = None,
            rotary_pos_emb: Optional[nn.Cell] = None,
            compress_ratio: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.compress_ratio = compress_ratio
        self.rotary_pos_emb = rotary_pos_emb
        self.layout = config.input_layout
        self.is_tnd = config.input_layout == "TND"
        self.dsa_indexer_loss_coeff = config.dsa_indexer_loss_coeff
        self.sparse_loss = config.dsa_indexer_use_sparse_loss

        self.softmax_scale = softmax_scale or config.v_head_dim ** -0.5
        self.ori_mask_mode = 4
        self.cmp_mask_mode = 3
        self.ori_win_left = config.csa_window_size - 1
        self.ori_win_right = 0
        self.window_size = config.csa_window_size
        self.apply_dsa_kernel_fusion = config.apply_dsa_kernel_fusion
        required_fused_ops = (
            npu_sparse_flash_mla,
            npu_sparse_flash_mla_grad,
            npu_sparse_lightning_indexer_kl_loss_grad,
        )
        if self.apply_dsa_kernel_fusion and any(op is None for op in required_fused_ops):
            raise ValueError(
                "DSV4 fused SparseFlashMla/indexer-loss ops are unavailable in this hyper_parallel build."
                "set apply_dsa_kernel_fusion=False to use the unfused small-op implement."
            )
        if self.apply_dsa_kernel_fusion and self.layout != "BSND":
            raise ValueError("The DSV4 fused precision path currently supports BSND layout only.")

        # --- attn_sink learnable Parameter (fp32, per-head) ----------
        self.attn_sink = Parameter(
            mint.zeros(config.num_attention_heads, dtype=mstype.float32),
            name="attn_sink",
        )

        if compress_ratio > 0 and submodules.compressor is not None:
            self.enable_compress = True
            self.compressor = build_module(
                submodules.compressor,
                config=config,
                compress_ratio=self.compress_ratio,
                head_dim=config.v_head_dim,
                rotate=False,
                rotary_pos_emb=self.rotary_pos_emb,
            )
        else:
            self.enable_compress = False
            self.compressor = None

        if compress_ratio == 4 and not config.csa_dense_mode and submodules.indexer is not None:
            self.enable_indexer = True
            self.indexer = build_module(
                submodules.indexer,
                config=config,
                compress_ratio=self.compress_ratio,
                rotary_pos_emb=self.rotary_pos_emb,
            )
            self.indexer_loss_auto_scaler = _IndexerLossAutoScaler()
            self.unfused_indexer_loss = UnfusedCSAIndexerLoss(
                softmax_scale=self.softmax_scale,
                loss_coeff=config.dsa_indexer_loss_coeff,
                sparse_loss=config.dsa_indexer_use_sparse_loss,
            )
        else:
            self.enable_indexer = False
            self.indexer = None
            self.indexer_loss_auto_scaler = None
            self.unfused_indexer_loss = None

        # Alias the non-trivial mint ops used in construct/forward per the
        # fine-grained-recompute convention (RFC §3.1 #9).
        self.squeeze = mint.squeeze
        self.reshape = mint.reshape
        self.unsqueeze = mint.unsqueeze
        self.cat = mint.cat
        self.permute = mint.permute
        self.where = mint.where

    def reset_parameter(self):
        """Zero ``attn_sink`` after meta-device ``to_empty()`` (B2)."""
        self.attn_sink.zero_()

    def bsnd_to_tnd(self, x):
        b, s, n, d = x.shape
        return self.reshape(x, (b * s, n, d))

    def construct(self, query, key, x, qr, actual_seq_len=None):
        """Core attention forward.
        Args:
            query: ``[sq, b, np, v_head_dim]`` multi-head query.
            key: ``[sq, b, 1, v_head_dim]`` shared single-head key.
            x: optional ``[sq, b, hidden_size]`` for indexer.
            qr: optional ``[sq, b, q_lora_rank]`` for indexer.

        Returns:
            ``[sq, b, np, v_head_dim]`` core attention output.
        """
        if self.apply_dsa_kernel_fusion:
            return self._construct_fused(query, key, x, qr, actual_seq_len)
        return self._construct_naive(query, key, x, qr, actual_seq_len)

    def _construct_fused(self, query, key, x, qr, actual_seq_len=None):
        """NPU fused operators forward."""
        s, b, n, d = query.shape
        topk_indices = None
        if self.enable_indexer:
            x_detach = ops.stop_gradient(x)
            qr_detach = ops.stop_gradient(qr)
            query_index, key_index, weights = self.indexer.forward_before_topk(x_detach, qr_detach)
            topk_indices, _ = self.indexer(query_index, key_index, weights,
                                           actual_seq_qlen=actual_seq_len,
                                           actual_seq_klen=actual_seq_len)
        # --- Build compressed KV (ratio > 1 only) -------------------
        compressed_kv = self.compressor(x) if self.enable_compress else None
        # SBND/3D seq-first -> BSND 4D (batch-first, explicit single-head axis).
        query = self.permute(query, (1, 0, 2, 3))   # [b, sq, np, vd]
        key = self.permute(key, (1, 0, 2, 3))       # [b, sq, 1, vd]
        if compressed_kv is not None:
            compressed_kv = self.permute(compressed_kv, (1, 0, 2, 3))
        if self.is_tnd:
            query = self.bsnd_to_tnd(query)
            key = self.bsnd_to_tnd(key)
            compressed_kv = self.bsnd_to_tnd(compressed_kv)
            topk_indices = self.bsnd_to_tnd(topk_indices)
        attn_sink = self.attn_sink
        if hasattr(attn_sink, "to_local"):
            attn_sink = attn_sink.to_local()
        # SparseFlashMla requires its sinks input in fp32.
        attn_sink = ops.cast(attn_sink, mstype.float32)
        if self.enable_indexer and self.training:
            output = FusedSparseFlashMlaWithIndexerLoss.apply(
                query,
                key,
                compressed_kv,
                topk_indices,
                ops.cast(query_index, mstype.bfloat16),
                ops.cast(key_index, mstype.bfloat16),
                ops.cast(weights, mstype.float32),
                attn_sink,
                self.softmax_scale,
                self.compress_ratio,
                self.ori_mask_mode,
                self.cmp_mask_mode,
                self.ori_win_left,
                self.ori_win_right,
                self.dsa_indexer_loss_coeff,
                self.layer_number,
                self.config.num_layers + (self.config.mtp_num_layers or 0),
            )
        else:
            output = FusedSparseFlashMla.apply(
                query,
                key,
                compressed_kv,
                topk_indices,
                attn_sink,
                self.softmax_scale,
                self.compress_ratio,
                self.ori_mask_mode,
                self.cmp_mask_mode,
                self.ori_win_left,
                self.ori_win_right,
            )
        # npu_sparse_attn_shared_kv returns BSND/TND [b, sq, np, vd]/[b * sq, np, vd];
        # reshape to [b, sq, np * vd]
        # then transpose back to [sq, b, np, vd]
        output = self.reshape(output, (b, s, n, d))
        output = self.permute(output, (1, 0, 2, 3))

        return output

    # pylint: disable=too-many-locals
    def _construct_naive(self, query, key, x, qr, actual_seq_len=None) -> Tensor:
        """
        Naive small-op forward (no NPU fused operators).

        Ratio modes:
          * ratio==0   : kv_full = ori_kv; topk_idxs = window_idxs.
          * ratio==4   : kv_full = cat([ori_kv, compressed_kv]); compressed top-k indices; window + compress idxs.
          * ratio==128 : kv_full = cat([ori_kv, compressed_kv]);
                         dense compress top-k indices via ``get_compress_topk_idxs``; window + dense.
        """
        sq, b, _, _  = query.shape
        ratio = self.compress_ratio
        cu_seqlens = parse_cu_seqlens(actual_seq_len, sq * b) # `None` will be returned if actual_seq_len is `None`.
        # --- Step 1: Compression -> concatenated KV space -----------
        compressed_kv = self.compressor(x) if self.enable_compress else None
        if compressed_kv is not None:
            kv_full = self.cat([key, compressed_kv], dim=0)
            n_compressed = int(compressed_kv.shape[0])
        else:
            kv_full = key
            n_compressed = 0
        offset = sq  # compressed indices start after the original positions.

        # --- Step 2: Sliding-window indices (into ori_kv) -----------
        if cu_seqlens is not None:
            window_idxs = get_doc_window_topk_idxs(self.window_size, b, sq, cu_seqlens)
        else:
            window_idxs = get_window_topk_idxs(self.window_size, b, sq)

        # --- Step 3: Compressed indices + (ratio==4) KL loss --------
        indexer_loss = None
        if ratio > 1 and n_compressed > 0:
            if self.enable_indexer:
                x_detach = ops.stop_gradient(x)
                qr_detach = ops.stop_gradient(qr)
                query_index, key_index, weights = self.indexer.forward_before_topk(x_detach, qr_detach)

                # causal_mask [b, sq, n_compressed]: -inf where the compressed
                # slot is in the future for that query.
                positions = self.unsqueeze(
                    mint.arange(1, sq + 1, dtype=mstype.int32), 1
                )  # [sq, 1]
                if cu_seqlens is not None:
                    causal_mask = build_doc_block_mask(ratio, sq, cu_seqlens)
                else:
                    cm = self.unsqueeze(
                        mint.arange(n_compressed, dtype=mstype.int32), 0
                    ).broadcast_to((sq, n_compressed))
                    future = cm >= (positions // ratio)
                    neg_inf = mint.full((sq, n_compressed), float("-inf"),
                                        dtype=mstype.float32)
                    zeros = mint.zeros((sq, n_compressed), dtype=mstype.float32)
                    causal_mask = self.where(future, neg_inf, zeros)
                causal_mask = self.reshape(causal_mask, (1, sq, n_compressed))

                topk_indices_compressed, index_scores = self.indexer(
                    query_index, key_index, weights, mask=causal_mask
                )
                # Indexer KL loss (small-op) — training only.
                if self.training:
                    indexer_loss = self.unfused_indexer_loss(
                        index_scores,
                        topk_indices_compressed,
                        ops.stop_gradient(query),
                        ops.stop_gradient(compressed_kv),
                        mask=causal_mask,
                    )
                    if self.dsa_indexer_loss_coeff > 0:
                        save_to_indexer_losses_tracker(
                            indexer_loss,
                            self.layer_number,
                            self.config.num_layers + (self.config.mtp_num_layers or 0),
                        )

                # Map compressed top-k into the concatenated KV space + drop future entries.
                if cu_seqlens is not None:
                    valid = mint.isfinite(index_scores)
                else:
                    n_valid_per_pos = positions // ratio  # [sq, 1]
                    valid = topk_indices_compressed < self.unsqueeze(n_valid_per_pos, 0)
                neg_one = mint.full(topk_indices_compressed.shape, -1, dtype=mstype.int32)
                compress_topk_idxs = self.where(valid, topk_indices_compressed + offset, neg_one)
            elif cu_seqlens is not None:
                compress_topk_idxs = get_doc_compress_topk_idxs(ratio, b, sq, offset, cu_seqlens)
            else:
                compress_topk_idxs = get_compress_topk_idxs(ratio, b, sq, offset)

            topk_idxs = self.cat([window_idxs, compress_topk_idxs], dim=-1)
        else:
            topk_idxs = window_idxs

        # --- Step 4: Differentiable sparse attention ----------------
        output = unfused_compressed_sparse_attn(
            query, kv_full, self.attn_sink, topk_idxs, self.softmax_scale
        )

        # --- Step 5: Attach KL loss (training, ratio==4) ------------
        if indexer_loss is not None and self.training:
            output = self.indexer_loss_auto_scaler.apply(output, indexer_loss)

        return output
