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
"""Context parallel wrappers for DeepSeek-V4 CSA / HCA in PyNative mode.

This module implements CSA/HCA CP as an explicit local-Tensor runtime. It does
not rely on HyperParallel DTensor distributed-op dispatch at the CSA fused-op
boundaries: the wrapper prepares the CP inputs, calls the single-rank kernels,
and returns the local output shard for the local query shard.

Supported scope in this first version:
  * padded SBND only; TND / no-pad multi-document input is rejected explicitly;
  * local query is never all-gathered, so CP still reduces query-side compute;
  * fused SparseFlashMla receives original KV in a prefix-length frame;
    values before the real sliding-window halo are zero-padded, so the kernel
    can infer the local-query offset from the ori_kv length;
  * compressed KV / compressed indexer KV are all-gathered and then sliced to
    the causal compressed prefix visible to the current CP rank;
  * fused Op-1/Op-2 are used when their CP input semantics are expressible;
    fused Op-3 is required for training the indexer KL loss on the fused CP path.
"""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from mindspore import Tensor, _no_grad, mint, nn, ops
from mindspore.common import dtype as mstype
from mindspore.ops import communication as comm

from hyper_parallel import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from mindformers.pynative.distributed.style import ParallelStyle
from mindformers.pynative.transformers.experimental_attention_variant.csa import (
    FusedSparseFlashMla,
    FusedSparseFlashMlaWithIndexerLoss,
    npu_sparse_flash_mla,
    npu_sparse_flash_mla_grad,
    npu_sparse_lightning_indexer_kl_loss_grad,
    unfused_compressed_sparse_attn,
)
from mindformers.pynative.transformers.experimental_attention_variant.indexer import npu_lightning_indexer
from mindformers.pynative.transformers.experimental_attention_variant.utils import (
    save_to_indexer_losses_tracker,
)

__all__ = [
    "CPRotaryEmbeddingProxy",
    "CompressedSparseAttentionContextParallel",
    "DSv4HybridAttentionContextParallel",
]


def _expect_local_tensor(value: Any, name: str) -> Tensor:
    """Validate that CSA CP receives local Tensor inputs, not DTensor layouts."""
    if isinstance(value, DTensor):
        raise ValueError(
            f"CSA context parallel currently expects local Tensor for {name}; "
            "DTensor input belongs to TP/CP composed layout support and is not implemented."
        )
    return value


def _ag_seq(tensor: Tensor, group_name: str) -> Tensor:
    """All-gather a sequence-major tensor over the CP group with autograd support."""
    out, _ = comm.all_gather_into_tensor(
        output_tensor=None,
        input_tensor=tensor.contiguous(),
        group=group_name,
    )
    return out


def build_select_pad_gather_index(
        cu_seqlens_padded: Sequence[int],
        cu_seqlens_real: Sequence[int],
        cp_size: int,
        ratio: int,
        overlap: bool = True,
) -> List[int]:
    """Build the static gather index for compressed-KV select-and-pad."""
    total_padded = int(cu_seqlens_padded[-1])
    local_seq = total_padded // cp_size
    slots_per_rank = local_seq // ratio
    out_slots = cp_size * slots_per_rank
    gathered_per_rank = slots_per_rank + 1 if overlap else slots_per_rank
    skip = 1 if overlap else 0
    sentinel = cp_size * gathered_per_rank

    def real_end(doc_id: int) -> int:
        return int(cu_seqlens_real[doc_id + 1])

    def doc_of(position: int) -> int:
        for doc_id in range(len(cu_seqlens_padded) - 1):
            if cu_seqlens_padded[doc_id] <= position < cu_seqlens_padded[doc_id + 1]:
                return doc_id
        return -1

    gather_index = []
    for global_block in range(out_slots):
        block_start = global_block * ratio
        doc_id = doc_of(block_start)
        if doc_id >= 0 and block_start + ratio <= real_end(doc_id):
            rank_id, slot_id = divmod(global_block, slots_per_rank)
            gather_index.append(rank_id * gathered_per_rank + skip + slot_id)
    gather_index += [sentinel] * (out_slots - len(gather_index))
    return gather_index


def select_and_pad_smallop(compressed_gathered: Tensor, gather_index: Sequence[int]) -> Tensor:
    """Select globally valid compressed slots and pad invalid tail positions with zero."""
    zero_row = mint.zeros((1,) + tuple(compressed_gathered.shape[1:]), dtype=compressed_gathered.dtype)
    compressed_padded = mint.cat([compressed_gathered, zero_row], dim=0)
    if not isinstance(gather_index, Tensor):
        gather_index = Tensor(np.asarray(gather_index, np.int32), mstype.int32)
    return mint.index_select(compressed_padded, 0, gather_index)


def assemble_window_halo(
        ori_bnd: Tensor,
        ori_kv: Tensor,
        cp_rank: int,
        halo_width: int,
        window_size: int,
) -> Tuple[Tensor, int]:
    """Prepend this rank's left sliding-window halo to local original KV."""
    local_seq = int(ori_kv.shape[0])
    global_start = cp_rank * local_seq
    needed = min(window_size - 1, global_start)
    if cp_rank == 0 or needed == 0:
        return ori_kv, 0
    halo = ori_bnd[cp_rank * halo_width - needed:cp_rank * halo_width]
    return mint.cat([halo, ori_kv], dim=0), needed



def assemble_prefix_padded_ori_kv(
        ori_bnd: Optional[Tensor],
        ori_kv: Tensor,
        cp_rank: int,
        halo_width: int,
        window_size: int,
) -> Tensor:
    """Build prefix-frame original KV for fused SparseFlashMla.

    SparseFlashMla uses the original-KV sequence length to infer the
    local-query global offset.  CP therefore cannot pass only ``halo + local``
    KV to the fused op.  Instead, this helper returns a prefix-length frame:
    positions before the real sliding-window halo are zero-padded, and the
    suffix contains exactly the real halo plus local original KV values.
    """
    local_seq = int(ori_kv.shape[0])
    global_start = cp_rank * local_seq
    real_start = max(0, global_start - window_size + 1)
    needed = global_start - real_start

    if cp_rank > 0 and needed > 0:
        if ori_bnd is None:
            raise ValueError("ori_bnd is required to build prefix-padded original KV for non-zero CP rank.")
        halo = ori_bnd[cp_rank * halo_width - needed:cp_rank * halo_width]
        real_ori = mint.cat([halo, ori_kv], dim=0)
    else:
        real_ori = ori_kv

    if real_start == 0:
        return real_ori
    pad = mint.zeros((real_start,) + tuple(ori_kv.shape[1:]), dtype=ori_kv.dtype)
    return mint.cat([pad, real_ori], dim=0)


def build_global_window_topk_idxs(
        window_size: int,
        batch_size: int,
        local_seq: int,
        pos_offset: int,
        ori_halo_len: int,
) -> Tensor:
    """Build local-query sliding-window indices in the local ``halo + ori`` frame."""
    index = np.full((batch_size, local_seq, window_size), -1, dtype=np.int32)
    for query_idx in range(local_seq):
        global_pos = pos_offset + query_idx
        first = max(0, global_pos - window_size + 1)
        values = np.arange(first, global_pos + 1, dtype=np.int32) - pos_offset + ori_halo_len
        index[:, query_idx, :values.shape[0]] = values
    return Tensor(index, mstype.int32)


def build_global_compress_topk_idxs(
        ratio: int,
        batch_size: int,
        local_seq: int,
        pos_offset: int,
        index_offset: int,
        n_compressed: int,
) -> Tensor:
    """Build dense compressed-KV visibility indices for local queries."""
    matrix = mint.tile(
        mint.unsqueeze(mint.arange(n_compressed, dtype=mstype.int32), 0),
        (local_seq, 1),
    )
    visible = mint.unsqueeze(
        mint.arange(pos_offset + 1, pos_offset + local_seq + 1, dtype=mstype.int32) // ratio,
        1,
    )
    invalid = matrix >= visible
    neg_one = mint.full(matrix.shape, -1, dtype=mstype.int32)
    matrix = mint.where(invalid, neg_one, matrix + index_offset)
    return mint.unsqueeze(matrix, 0).broadcast_to((batch_size, local_seq, n_compressed))


def build_global_compressed_causal_mask(
        ratio: int,
        batch_size: int,
        local_seq: int,
        pos_offset: int,
        n_compressed: int,
) -> Tensor:
    """Build ``[B, S_local, N_cmp]`` mask for global compressed-KV visibility."""
    compressed_pos = mint.unsqueeze(
        mint.arange(n_compressed, dtype=mstype.int32),
        0,
    ).broadcast_to((local_seq, n_compressed))
    query_visible = mint.unsqueeze(
        mint.arange(pos_offset + 1, pos_offset + local_seq + 1, dtype=mstype.int32) // ratio,
        1,
    )
    future = compressed_pos >= query_visible
    neg_inf = mint.full((local_seq, n_compressed), float("-inf"), dtype=mstype.float32)
    zeros = mint.zeros((local_seq, n_compressed), dtype=mstype.float32)
    mask = mint.where(future, neg_inf, zeros)
    return mint.unsqueeze(mask, 0).broadcast_to((batch_size, local_seq, n_compressed))


def stage1_compress_with_halo(
        compressor: nn.Cell,
        local_x: Tensor,
        halo_x: Optional[Tensor],
        cp_rank: int,
        overlap: bool = True,
) -> Tensor:
    """Compress local hidden states with the left boundary halo needed by CSA."""
    ratio = compressor.compress_ratio
    slots_per_rank = int(local_x.shape[0]) // ratio
    if not overlap:
        return compressor(local_x, rope_pos_offset=cp_rank * slots_per_rank * ratio)
    if cp_rank == 0:
        blocks = compressor(local_x, rope_pos_offset=0)
        pad = mint.zeros((1,) + tuple(blocks.shape[1:]), dtype=blocks.dtype)
        return mint.cat([pad, blocks], dim=0)
    extended_x = mint.cat([halo_x, local_x], dim=0)
    rope_offset = (cp_rank * slots_per_rank - 1) * ratio
    return compressor(extended_x, rope_pos_offset=rope_offset)


def _gather_global_compressed(
        compressor: nn.Cell,
        x: Tensor,
        halo_x: Optional[Tensor],
        cp_size: int,
        cp_rank: int,
        cp_group: str,
        overlap: bool,
) -> Tensor:
    """Build global compressed KV from local compressed slots."""
    local_slots = stage1_compress_with_halo(compressor, x, halo_x, cp_rank, overlap=overlap)
    gathered = _ag_seq(local_slots, cp_group)
    cu_seqlens = [0, int(x.shape[0]) * cp_size]
    gather_index = build_select_pad_gather_index(
        cu_seqlens,
        cu_seqlens,
        cp_size,
        compressor.compress_ratio,
        overlap=overlap,
    )
    return select_and_pad_smallop(gathered, gather_index)


def _indexer_query_and_weights(indexer: nn.Cell, x: Tensor, qr: Tensor, query_pos_offset: int):
    """Compute indexer query and weights with query RoPE at global positions."""
    query_index, _, weights = indexer.forward_before_topk(
        x,
        qr,
        query_pos_offset=query_pos_offset,
    )
    return query_index, weights


def _run_global_indexer_topk(
        indexer: nn.Cell,
        query_index: Tensor,
        key_index_global: Tensor,
        weights: Tensor,
        sparse_count: int,
        mask: Tensor,
):
    """Run a small-op indexer top-k over the global compressed indexer key."""
    query_index = indexer.cast(query_index, mstype.float32)
    key_index_global = indexer.cast(key_index_global, mstype.float32)
    weights = indexer.cast(weights, mstype.float32)

    batch_size, seq_len, n_heads, head_dim = query_index.shape
    n_compressed = int(key_index_global.shape[1])
    query_index = indexer.reshape(query_index, (batch_size, seq_len * n_heads, head_dim))
    key_index_global = indexer.permute(
        indexer.reshape(key_index_global, (batch_size, n_compressed, head_dim)),
        (0, 2, 1),
    )
    index_scores = indexer.bmm(query_index, key_index_global)
    index_scores = indexer.reshape(index_scores, (batch_size, seq_len, n_heads, n_compressed))
    index_scores = indexer.relu(index_scores)
    index_scores = index_scores * indexer.unsqueeze(weights, -1)
    index_scores = indexer.sum(index_scores, dim=2)
    index_scores = index_scores + indexer.cast(mask, mstype.float32)
    topk_scores, topk_indices = indexer.topk(index_scores, k=sparse_count, dim=-1)
    return indexer.cast(topk_indices, mstype.int32), topk_scores


def _attach_indexer_loss(
        cell: nn.Cell,
        output: Tensor,
        index_scores: Optional[Tensor],
        topk_indices_compressed: Optional[Tensor],
        query: Tensor,
        compressed_kv: Tensor,
        causal_mask: Tensor,
) -> Tensor:
    """Attach the unfused indexer KL loss to the output graph when training."""
    if (
            index_scores is None
            or not cell.training
            or cell.indexer_loss_auto_scaler is None
            or getattr(cell, "unfused_indexer_loss", None) is None
    ):
        return output
    indexer_loss = cell.unfused_indexer_loss(
        index_scores,
        topk_indices_compressed,
        ops.stop_gradient(query),
        ops.stop_gradient(compressed_kv),
        mask=causal_mask,
    )
    if cell.dsa_indexer_loss_coeff > 0:
        save_to_indexer_losses_tracker(
            indexer_loss,
            cell.layer_number,
            cell.config.num_layers + (cell.config.mtp_num_layers or 0),
        )
    return cell.indexer_loss_auto_scaler.apply(output, indexer_loss)


def _compressed_prefix_len(local_seq: int, cp_rank: int, ratio: int) -> int:
    """Return compressed prefix length visible to the current CP rank."""
    if ratio <= 1:
        return 0
    if local_seq % ratio != 0:
        raise ValueError(
            f"CSA fused context parallel requires local sequence length divisible by compress_ratio, "
            f"got S={local_seq}, ratio={ratio}."
        )
    return (cp_rank + 1) * local_seq // ratio


def _slice_compressed_prefix(compressed: Tensor, local_seq: int, cp_rank: int, ratio: int) -> Tensor:
    """Slice global compressed sequence to this rank's causal prefix."""
    prefix_len = _compressed_prefix_len(local_seq, cp_rank, ratio)
    if prefix_len <= 0 or prefix_len > int(compressed.shape[0]):
        raise ValueError(
            f"Invalid CSA compressed prefix length {prefix_len}; "
            f"global compressed length is {int(compressed.shape[0])}."
        )
    return compressed[:prefix_len]


def _sbnd_to_bsnd(tensor: Optional[Tensor]) -> Optional[Tensor]:
    """Convert sequence-major SBND tensor to fused-op BSND layout."""
    if tensor is None:
        return None
    return mint.permute(tensor, (1, 0, 2, 3))


def _call_fused_sparse_flash_mla(
        cell: nn.Cell,
        query: Tensor,
        ori_kv: Tensor,
        cmp_kv: Optional[Tensor],
        cmp_sparse_indices: Optional[Tensor],
        query_index: Optional[Tensor] = None,
        key_index_prefix: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
) -> Tensor:
    """Call fused SparseFlashMla on local query and prepared CP KV inputs.

    Args use sequence-major tensors before this helper:
      * query: [S_local, B, N, D]
      * ori_kv: [S_ori_prefix_frame, B, 1, D]
      * cmp_kv: [S_cmp_prefix, B, 1, D] or None
      * cmp_sparse_indices: [B, S_local, K] or None

    The helper converts tensor operands to BSND and reuses the single-rank
    fused autograd Function, so SparseFlashMla backward and the new
    SparseLightningIndexerKLLossGrad path stay consistent with non-CP fused CSA.
    """
    if npu_sparse_flash_mla is None or npu_sparse_flash_mla_grad is None:
        raise ValueError(
            "Fused CSA CP requires 'npu_sparse_flash_mla' and "
            "'npu_sparse_flash_mla_grad', but at least one op is unavailable."
        )

    seq_len, batch_size, n_heads, head_dim = query.shape
    attn_sink = cell.attn_sink
    if hasattr(attn_sink, "to_local"):
        attn_sink = attn_sink.to_local()
    attn_sink = ops.cast(attn_sink, mstype.float32)

    query_bsnd = _sbnd_to_bsnd(query)
    ori_kv_bsnd = _sbnd_to_bsnd(ori_kv)
    cmp_kv_bsnd = _sbnd_to_bsnd(cmp_kv)

    if query_index is not None:
        if npu_sparse_lightning_indexer_kl_loss_grad is None:
            raise ValueError(
                "Fused CSA CP training with compress_ratio=4 requires "
                "'npu_sparse_lightning_indexer_kl_loss_grad', but the op is unavailable."
            )
        output = FusedSparseFlashMlaWithIndexerLoss.apply(
            query_bsnd,
            ori_kv_bsnd,
            cmp_kv_bsnd,
            cmp_sparse_indices,
            query_index,
            _sbnd_to_bsnd(key_index_prefix),
            weights,
            attn_sink,
            cell.softmax_scale,
            cell.compress_ratio,
            cell.ori_mask_mode,
            cell.cmp_mask_mode,
            cell.ori_win_left,
            cell.ori_win_right,
            cell.dsa_indexer_loss_coeff,
            cell.layer_number,
            cell.config.num_layers + (cell.config.mtp_num_layers or 0),
        )
    else:
        output = FusedSparseFlashMla.apply(
            query_bsnd,
            ori_kv_bsnd,
            cmp_kv_bsnd,
            cmp_sparse_indices,
            attn_sink,
            cell.softmax_scale,
            cell.compress_ratio,
            cell.ori_mask_mode,
            cell.cmp_mask_mode,
            cell.ori_win_left,
            cell.ori_win_right,
        )
    output = mint.reshape(output, (batch_size, seq_len, n_heads, head_dim))
    return mint.permute(output, (1, 0, 2, 3))


def _run_fused_prefix_indexer_topk(
        indexer: nn.Cell,
        query_index: Tensor,
        key_index_prefix: Tensor,
        weights: Tensor,
) -> Tensor:
    """Run fused indexer top-k against a compressed causal prefix.

    ``key_index_prefix`` is the prefix slice of the global compressed indexer
    key. Returned indices are local to that prefix, which is exactly the index
    space expected later by fused shared-KV attention and fused indexer loss.
    """
    if npu_lightning_indexer is None:
        raise ValueError(
            "Fused DSA op 'npu_lightning_indexer' is unavailable in this hyper_parallel build."
        )
    key_index_prefix = _sbnd_to_bsnd(key_index_prefix)
    effective_topk = min(indexer.index_topk, int(key_index_prefix.shape[1]))
    if effective_topk <= 0:
        raise ValueError("CSA fused context parallel requires a non-empty compressed indexer prefix.")
    with _no_grad():
        cmp_residual_k = Tensor(
            [int(key_index_prefix.shape[1]) % indexer.compress_ratio],
            dtype=mstype.int32,
        )
        topk_indices, _ = npu_lightning_indexer(
            indexer.cast(query_index, indexer.compute_dtype),
            indexer.cast(key_index_prefix, indexer.compute_dtype),
            indexer.cast(weights, mstype.float32),
            effective_topk,
            cmp_residual_k=cmp_residual_k,
            layout=indexer.input_layout,
            sparse_mode=indexer.sparse_mode,
            cmp_ratio=indexer.compress_ratio,
            return_value=True,
        )
        if len(topk_indices.shape) == 4 and int(topk_indices.shape[2]) == 1:
            topk_indices = mint.squeeze(topk_indices, dim=2)
    return topk_indices


def two_stage_cp_naive_attention(
        cell: nn.Cell,
        query: Tensor,
        ori_kv: Tensor,
        x: Tensor,
        qr: Tensor,
        cp_size: int,
        cp_rank: int,
        cp_group: str,
) -> Tensor:
    """Run two-stage CSA/HCA CP with small-op attention.

    This path mirrors the single-card naive CSA semantics while keeping only
    the query shard local. Stage 1 prepares cross-rank operands: a sliding-window
    halo for original KV, and global compressed KV / compressed indexer KV via
    local compression, CP all-gather and select-and-pad. Stage 2 builds the
    local query visibility indices and calls the differentiable small-op shared
    KV attention. The output is the local O shard for ``query``.
    """
    if getattr(cell, "is_tnd", False):
        raise ValueError("CSA context parallel currently supports padded SBND only, but got input_layout='TND'.")

    ratio = int(cell.compress_ratio)
    local_seq = int(query.shape[0])
    batch_size = int(query.shape[1])
    pos_offset = cp_rank * local_seq
    overlap = ratio == 4
    ties = []

    if ratio > 1 and local_seq % ratio != 0:
        raise ValueError(
            f"CSA context parallel requires local sequence length divisible by compress_ratio, "
            f"got S={local_seq}, ratio={ratio}."
        )

    # --- Stage 1a: build the boundary halo for ratio==4 overlap compression ---
    halo_x = None
    if overlap:
        x_bnd = _ag_seq(x[-ratio:], cp_group)
        ties.append(x_bnd)
        if cp_rank > 0:
            halo_x = x_bnd[(cp_rank - 1) * ratio:cp_rank * ratio]

    # --- Stage 1b: prepend the original-KV sliding-window halo ---
    window_size = int(cell.window_size)
    ori_ext = ori_kv
    ori_halo_len = 0
    if window_size > 1:
        halo_width = min(window_size - 1, local_seq)
        ori_bnd = _ag_seq(ori_kv[-halo_width:], cp_group)
        ties.append(ori_bnd)
        ori_ext, ori_halo_len = assemble_window_halo(ori_bnd, ori_kv, cp_rank, halo_width, window_size)

    # --- Stage 1c: gather the global compressed KV used by local queries ---
    compressed_kv = None
    indexer_scores = None
    topk_indices_compressed = None
    causal_mask = None
    if ratio > 1 and cell.enable_compress:
        compressed_kv = _gather_global_compressed(
            cell.compressor,
            x,
            halo_x,
            cp_size,
            cp_rank,
            cp_group,
            overlap,
        )

    if compressed_kv is not None:
        kv_full = mint.cat([ori_ext, compressed_kv], dim=0)
        n_compressed = int(compressed_kv.shape[0])
    else:
        kv_full = ori_ext
        n_compressed = 0

    # --- Stage 2a: build local-query visibility indices in the CP frame ---
    topk_idxs = build_global_window_topk_idxs(
        window_size,
        batch_size,
        local_seq,
        pos_offset,
        ori_halo_len,
    )
    index_offset = int(ori_ext.shape[0])

    if ratio > 1 and n_compressed > 0:
        if cell.enable_indexer:
            x_detach = ops.stop_gradient(x)
            qr_detach = ops.stop_gradient(qr)
            halo_x_detach = ops.stop_gradient(halo_x) if halo_x is not None else None
            cmp_idx_global = _gather_global_compressed(
                cell.indexer.compressor,
                x_detach,
                halo_x_detach,
                cp_size,
                cp_rank,
                cp_group,
                overlap,
            )
            query_index, weights = _indexer_query_and_weights(
                cell.indexer,
                x_detach,
                qr_detach,
                pos_offset,
            )
            key_index = cell.indexer.permute(cmp_idx_global, (1, 0, 2, 3))
            causal_mask = build_global_compressed_causal_mask(
                ratio,
                batch_size,
                local_seq,
                pos_offset,
                n_compressed,
            )
            effective_topk = min(cell.indexer.index_topk, n_compressed)
            topk_indices_compressed, indexer_scores = _run_global_indexer_topk(
                cell.indexer,
                query_index,
                key_index,
                weights,
                effective_topk,
                causal_mask,
            )
            visible = mint.unsqueeze(
                mint.arange(pos_offset + 1, pos_offset + local_seq + 1, dtype=mstype.int32) // ratio,
                1,
            )
            valid = topk_indices_compressed < cell.unsqueeze(visible, 0)
            neg_one = mint.full(topk_indices_compressed.shape, -1, dtype=mstype.int32)
            compress_topk_idxs = cell.where(valid, topk_indices_compressed + index_offset, neg_one)
        else:
            compress_topk_idxs = build_global_compress_topk_idxs(
                ratio,
                batch_size,
                local_seq,
                pos_offset,
                index_offset,
                n_compressed,
            )
        topk_idxs = cell.cat([topk_idxs, compress_topk_idxs], dim=-1)

    # --- Stage 2b: run differentiable small-op shared-KV attention ---
    output = unfused_compressed_sparse_attn(query, kv_full, cell.attn_sink, topk_idxs, cell.softmax_scale)

    # --- Attach indexer KL loss to the output graph when ratio==4 trains the indexer ---
    if causal_mask is not None:
        output = _attach_indexer_loss(
            cell,
            output,
            indexer_scores,
            topk_indices_compressed,
            query,
            compressed_kv,
            causal_mask,
        )
    # --- Tie gathered tensors to keep collective backward symmetric on all ranks ---
    for tensor in ties:
        output = output + ops.cast(tensor.sum(), output.dtype) * 0.0
    return output


def two_stage_cp_fused_attention(
        cell: nn.Cell,
        query: Tensor,
        ori_kv: Tensor,
        x: Tensor,
        qr: Tensor,
        cp_size: int,
        cp_rank: int,
        cp_group: str,
) -> Tensor:
    """Run two-stage CSA/HCA CP using fused prefix operators when supported.

    Ratio-specific fused policy:
      * ratio==0: pure sliding-window attention. There is no compressed branch,
        so Op-2 band masking is aligned in the local ``left_halo + local_kv``
        frame and no compressed-prefix offset is needed.
      * ratio==4: gather compressed KV and compressed indexer KV, slice both to
        this rank's compressed causal prefix, run fused Op-1 to produce prefix-
        local indices, then run fused Op-2. Training uses fused Op-3 for the
        indexer KL loss.
      * ratio==128: there is no dynamic indexer. The fused attention consumes
        the compressed prefix directly with ``cmp_sparse_indices=None``.

    Op-1/Op-2 are required for the fused forward path and missing kernels raise
    immediately. Op-3 is required when the fused CP path needs to train the
    indexer KL loss.
    """
    if getattr(cell, "is_tnd", False):
        raise ValueError("CSA fused context parallel currently supports padded SBND only, but got input_layout='TND'.")

    ratio = int(cell.compress_ratio)
    if npu_sparse_flash_mla is None or npu_sparse_flash_mla_grad is None:
        raise ValueError(
            "Fused CSA CP requires 'npu_sparse_flash_mla' and "
            "'npu_sparse_flash_mla_grad', but at least one op is unavailable."
        )
    if ratio not in (0, 4, 128):
        raise ValueError(
            f"CSA fused context parallel supports only compress_ratio in (0, 4, 128), but got {ratio}."
        )

    local_seq = int(query.shape[0])
    pos_offset = cp_rank * local_seq
    overlap = ratio == 4
    ties = []

    # --- Pure window: Op-2 runs on local query plus prefix-framed original KV ---
    if ratio == 0:
        window_size = int(cell.window_size)
        ori_ext = ori_kv
        if window_size > 1:
            halo_width = min(window_size - 1, local_seq)
            ori_bnd = _ag_seq(ori_kv[-halo_width:], cp_group)
            ties.append(ori_bnd)
            ori_ext = assemble_prefix_padded_ori_kv(ori_bnd, ori_kv, cp_rank, halo_width, window_size)
        output = _call_fused_sparse_flash_mla(cell, query, ori_ext, None, None)
        for tensor in ties:
            output = output + ops.cast(tensor.sum(), output.dtype) * 0.0
        return output

    if local_seq < ratio or local_seq % ratio != 0:
        raise ValueError(
            f"CSA fused context parallel requires local sequence length to be >= and divisible by "
            f"compress_ratio, got S={local_seq}, ratio={ratio}."
        )
    if not cell.enable_compress:
        raise ValueError(
            f"CSA fused context parallel with compress_ratio={ratio} requires "
            "cell.enable_compress=True and a valid compressor, but compression is disabled."
        )

    # --- Stage 1a: build compression halo for ratio==4 overlap blocks ---
    halo_x = None
    if overlap:
        x_bnd = _ag_seq(x[-ratio:], cp_group)
        ties.append(x_bnd)
        if cp_rank > 0:
            halo_x = x_bnd[(cp_rank - 1) * ratio:cp_rank * ratio]

    # --- Stage 1b: prepare original KV in prefix frame for fused Op-2 ---
    window_size = int(cell.window_size)
    ori_ext = ori_kv
    if window_size > 1:
        halo_width = min(window_size - 1, local_seq)
        ori_bnd = _ag_seq(ori_kv[-halo_width:], cp_group)
        ties.append(ori_bnd)
        ori_ext = assemble_prefix_padded_ori_kv(ori_bnd, ori_kv, cp_rank, halo_width, window_size)

    # --- Stage 1c: gather compressed KV and slice this rank's causal prefix ---
    compressed_global = _gather_global_compressed(
        cell.compressor,
        x,
        halo_x,
        cp_size,
        cp_rank,
        cp_group,
        overlap,
    )
    compressed_prefix = _slice_compressed_prefix(compressed_global, local_seq, cp_rank, ratio)

    cmp_sparse_indices = None
    query_index = None
    weights = None
    cmp_idx_prefix = None
    if ratio == 4 and not cell.enable_indexer:
        raise ValueError(
            "CSA fused context parallel with compress_ratio=4 requires "
            "cell.enable_indexer=True and a valid indexer."
        )
    # --- Fused Op-1: build prefix-local compressed indices for ratio==4 ---
    if ratio == 4 and cell.enable_indexer:
        if npu_lightning_indexer is None:
            raise ValueError(
                "Fused CSA CP ratio=4 requires 'npu_lightning_indexer', but the op is unavailable."
            )
        x_detach = ops.stop_gradient(x)
        qr_detach = ops.stop_gradient(qr)
        halo_x_detach = ops.stop_gradient(halo_x) if halo_x is not None else None
        cmp_idx_global = _gather_global_compressed(
            cell.indexer.compressor,
            x_detach,
            halo_x_detach,
            cp_size,
            cp_rank,
            cp_group,
            overlap,
        )
        cmp_idx_prefix = _slice_compressed_prefix(cmp_idx_global, local_seq, cp_rank, ratio)
        query_index, weights = _indexer_query_and_weights(
            cell.indexer,
            x_detach,
            qr_detach,
            pos_offset,
        )
        cmp_sparse_indices = _run_fused_prefix_indexer_topk(
            cell.indexer,
            query_index,
            cmp_idx_prefix,
            weights,
        )

    # --- Fused Op-2: local-query SparseFlashMla over ori halo and compressed prefix ---
    use_fused_indexer_loss = (
        cell.training
        and cell.indexer_loss_auto_scaler is not None
        and cmp_sparse_indices is not None
    )
    output = _call_fused_sparse_flash_mla(
        cell,
        query,
        ori_ext,
        compressed_prefix,
        cmp_sparse_indices,
        query_index if use_fused_indexer_loss else None,
        cmp_idx_prefix if use_fused_indexer_loss else None,
        weights if use_fused_indexer_loss else None,
    )
    # --- Tie gathered tensors to keep collective backward symmetric on all ranks ---
    for tensor in ties:
        output = output + ops.cast(tensor.sum(), output.dtype) * 0.0
    return output


class CPRotaryEmbeddingProxy(nn.Cell):
    """Proxy that supplies a CP global-position offset to a single-card RoPE cell."""

    def __init__(self, base_rope: nn.Cell, cp_rank: int, cp_size: int) -> None:
        super().__init__()
        self.base_rope = base_rope
        self.cp_rank = int(cp_rank)
        self.cp_size = int(cp_size)

    def construct(self, max_seq_len: int, offset: int = 0, position_ids: Optional[Tensor] = None) -> Any:
        """Generate RoPE frequencies with CP offset when the caller did not provide positions."""
        if self.cp_size <= 1 or position_ids is not None or offset != 0:
            return self.base_rope(max_seq_len, offset=offset, position_ids=position_ids)
        return self.base_rope(max_seq_len, offset=self.cp_rank * int(max_seq_len), position_ids=position_ids)


class CompressedSparseAttentionContextParallel(ParallelStyle):
    """Two-stage CP wrapper for PR8386 ``CompressedSparseAttention`` cells."""

    def __init__(self, cp_mesh: DeviceMesh) -> None:
        super().__init__()
        self.cp_size = int(cp_mesh.size())
        self.cp_rank = int(cp_mesh.get_local_rank())
        self.cp_group = cp_mesh.get_group()

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh = None) -> nn.Cell:
        """Replace ``module.construct`` with local-query CSA CP dispatch."""
        del device_mesh
        if self.cp_size <= 1:
            return module
        if getattr(module, "_mf_csa_cp_wrapped", False):
            return module

        cp_size = self.cp_size
        cp_rank = self.cp_rank
        cp_group = self.cp_group

        def _cp_construct(
                query,
                key,
                x,
                qr,
                actual_seq_len=None,
                actual_seq_qlen=None,
                actual_seq_kvlen=None,
        ):
            if actual_seq_len is not None or actual_seq_qlen is not None or actual_seq_kvlen is not None:
                raise ValueError("CSA context parallel currently does not support TND/no-pad actual_seq_len.")
            query = _expect_local_tensor(query, "query")
            key = _expect_local_tensor(key, "key")
            x = _expect_local_tensor(x, "x")
            qr = _expect_local_tensor(qr, "qr")
            if module.apply_dsa_kernel_fusion:
                return two_stage_cp_fused_attention(module, query, key, x, qr, cp_size, cp_rank, cp_group)
            return two_stage_cp_naive_attention(module, query, key, x, qr, cp_size, cp_rank, cp_group)

        module.construct = _cp_construct
        setattr(module, "_mf_csa_cp_wrapped", True)
        return module


class DSv4HybridAttentionContextParallel(ParallelStyle):
    """Apply CP to a DSv4 hybrid attention module without editing its single-card code."""

    def __init__(self, cp_mesh: DeviceMesh) -> None:
        super().__init__()
        self.cp_mesh = cp_mesh
        self.cp_size = int(cp_mesh.size())
        self.cp_rank = int(cp_mesh.get_local_rank())

    def _wrap_parent_rope(self, module: nn.Cell) -> None:
        """Wrap only the parent hybrid RoPE used by main Q/K and inverse RoPE."""
        rotary_pos_emb = getattr(module, "rotary_pos_emb", None)
        if rotary_pos_emb is None or isinstance(rotary_pos_emb, CPRotaryEmbeddingProxy):
            return
        module.rotary_pos_emb = CPRotaryEmbeddingProxy(rotary_pos_emb, self.cp_rank, self.cp_size)

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh = None) -> nn.Cell:
        """Apply hybrid-level RoPE offset and CSA core CP."""
        del device_mesh
        if self.cp_size <= 1:
            return module
        if not hasattr(module, "core_attention"):
            raise ValueError("DSv4HybridAttentionContextParallel expects module.core_attention to exist.")
        self._wrap_parent_rope(module)
        CompressedSparseAttentionContextParallel(self.cp_mesh)._apply(module.core_attention, self.cp_mesh)
        return module
