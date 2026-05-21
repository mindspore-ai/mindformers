# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Muon optimizer for pynative mode.

This module provides the Muon optimizer implementation for pynative mode training.
Unlike the static graph version, this version:
- Uses Python for loops instead of hyper_map + MultitypeFuncGraph
- Uses direct reshape operations instead of Morph operator
"""

import copy as cp
from fnmatch import fnmatch

from mindspore import mint
from mindspore.common import dtype as mstype
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import functional as F, operations as P
from mindspore.ops.function import comm_func
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.common.tensor import Tensor
from mindspore.communication import get_rank, get_group_size
from mindspore.mint.distributed import broadcast, irecv, isend

from hyper_parallel import DTensor
from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.layout import _infer_slice_area_by_rank
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.core.dtensor.redistribute_infer import RedistributionOperatorInfer
from hyper_parallel.platform import get_platform

from mindformers.core import context as core_context
from mindformers.tools.logger import logger
from mindformers.pynative.optimizer.adamw import _run_adamw_opt


_HP_PLATFORM = get_platform()
_FULL_TENSOR_OP_CACHE = {}
_ALL_CONCAT_GROUP_CACHE = {}


def _create_state_parameter(old_param, prefix, init='zeros'):
    """Create optimizer state parameter with the same shape and dtype as the original parameter."""
    param = old_param.clone(init)
    param.name = prefix + "." + old_param.name
    return param


def _to_local(tensor):
    """Return the local tensor for a DTensor, otherwise pass through."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _shape_numel(shape):
    """Return the number of elements represented by ``shape``."""
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return int(numel)


class _ShapeOnlyTensor:
    """Tiny shape carrier for schema callables used during cost estimation."""

    def __init__(self, shape):
        self.shape = tuple(shape)


def _eval_static_tuple(spec, param_name, shape):
    """Evaluate a static schema tuple, falling back to a shape-only tensor for callables."""
    if callable(spec):
        return spec(param_name, _ShapeOnlyTensor(shape))
    return spec


def _match_muon_schema_rule(param_name, muon_split_fn):
    """Return the model Muon schema rule for ``param_name`` when available."""
    for rule in getattr(muon_split_fn, "_muon_schema", ()):
        if any(fnmatch(param_name, pattern) for pattern in rule["patterns"]):
            return rule
    return None


def _estimate_periodic_shapes(shape, rule, param_name):
    """Estimate output piece shapes for a periodic row split."""
    if len(shape) != 2:
        return [shape]
    part_a, part_b = _eval_static_tuple(rule["parts"], param_name, shape)[:2]
    part_a = int(part_a)
    part_b = int(part_b)
    rows, cols = int(shape[0]), int(shape[1])
    total = part_a + part_b
    if total <= 0 or rows % total != 0:
        return [shape]
    groups = rows // total
    return [(groups * part_a, cols), (groups * part_b, cols)]


def _estimate_reshape_concat_shapes(shape, rule, param_name):
    """Estimate piece shapes for reshape+concat Muon rules."""
    _, hidden_size, total_intermediate = _eval_static_tuple(rule["reshape"], param_name, shape)
    hidden_size = int(hidden_size)
    total_intermediate = int(total_intermediate)
    if hidden_size <= 0 or total_intermediate <= 0:
        return [shape]
    batch = max(1, _shape_numel(shape) // (hidden_size * total_intermediate))
    half_intermediate = total_intermediate // 2
    return [
        (batch, hidden_size, half_intermediate),
        (batch, hidden_size, total_intermediate - half_intermediate),
    ]


def _estimate_reshape_only_shapes(shape, rule, param_name):
    """Estimate piece shapes for reshape-only Muon rules."""
    _, intermediate_size, hidden_size = _eval_static_tuple(rule["reshape"], param_name, shape)
    intermediate_size = int(intermediate_size)
    hidden_size = int(hidden_size)
    if intermediate_size <= 0 or hidden_size <= 0:
        return [shape]
    batch = max(1, _shape_numel(shape) // (intermediate_size * hidden_size))
    return [(batch, intermediate_size, hidden_size)]


def _estimate_block_split_shapes(shape, rule, param_name):
    """Estimate piece shapes for block-split Muon rules."""
    if len(shape) != 2:
        return [shape]
    cols = int(shape[1])
    piece_shapes = []
    for block in _eval_static_tuple(rule["blocks"], param_name, shape):
        if isinstance(block, int):
            piece_shapes.append((int(block), cols))
            continue
        part_a, part_b, num_blocks = block
        piece_shapes.append((int(part_a) * int(num_blocks), cols))
        piece_shapes.append((int(part_b) * int(num_blocks), cols))
    return piece_shapes or [shape]


def _estimate_muon_piece_shapes(param_name, shape, muon_split_fn):
    """Estimate the Newton-Schulz input piece shapes created by model split rules."""
    shape = tuple(int(dim) for dim in shape)
    rule = _match_muon_schema_rule(param_name, muon_split_fn)
    if rule is None:
        return [shape]

    kind = rule["kind"]
    try:
        if kind == "periodic":
            return _estimate_periodic_shapes(shape, rule, param_name)
        if kind == "reshape_concat":
            return _estimate_reshape_concat_shapes(shape, rule, param_name)
        if kind == "reshape_only":
            return _estimate_reshape_only_shapes(shape, rule, param_name)
        if kind == "block_split":
            return _estimate_block_split_shapes(shape, rule, param_name)
    except (TypeError, ValueError, ZeroDivisionError):
        return [shape]
    return [shape]


def _estimate_ns_work_for_shape(shape, ns_steps):
    """Estimate Newton-Schulz compute work for one 2D or 3D input piece."""
    if len(shape) == 2:
        batch = 1
        dim_a, dim_b = int(shape[-2]), int(shape[-1])
    elif len(shape) == 3:
        batch = int(shape[0])
        dim_a, dim_b = int(shape[-2]), int(shape[-1])
    else:
        return _shape_numel(shape)

    short_dim = min(dim_a, dim_b)
    long_dim = max(dim_a, dim_b)
    per_step = 2 * long_dim * short_dim * short_dim + short_dim * short_dim * short_dim
    return int(batch * (int(ns_steps) * per_step + dim_a * dim_b))


def _estimate_muon_work(param_name, shape, muon_split_fn, ns_steps):
    """Estimate total Muon work for load-balanced rank assignment."""
    piece_shapes = _estimate_muon_piece_shapes(param_name, shape, muon_split_fn)
    return sum(_estimate_ns_work_for_shape(piece_shape, ns_steps) for piece_shape in piece_shapes)


def _format_work_load_summary(rank_loads, rank_counts):
    """Format rank load statistics for one-line optimizer init logging."""
    return ", ".join(
        f"r{rank}:count={rank_counts[rank]},work={rank_loads[rank]:.3e}"
        for rank in range(len(rank_loads))
    )


class _AsyncAllConcatTensor:
    """Pending async all-gather that matches DTensor all_concat reconstruction."""

    def __init__(self, output, handle, concat_size, concat_dim):
        self.output = output
        self.handle = handle
        self.concat_size = int(concat_size)
        self.concat_dim = int(concat_dim)

    def wait(self):
        """Wait for the gather and return the reconstructed full tensor."""
        if self.handle is not None:
            self.handle.wait()
        if self.concat_dim == 0:
            return self.output
        output_tensors = P.Split(output_num=self.concat_size)(self.output)
        return mint.concat(output_tensors, self.concat_dim)


class _PendingP2PGather:
    """P2P gather of a sharded 2D weight to ``assigned_rank``; only the owner
    receives data so total HCCS traffic is ``N`` times less than all-gather."""

    __slots__ = ("full_tensor", "_handles", "_shards", "_concat_dim", "_keep_alive",
                 "_prealloc_full", "_deferred")

    def __init__(self, full_tensor=None, handles=(), shards=None, concat_dim=0,
                 keep_alive=(), prealloc_full=None, deferred=None):
        self.full_tensor = full_tensor
        self._handles = list(handles)
        self._shards = shards
        self._concat_dim = int(concat_dim)
        self._keep_alive = list(keep_alive)
        self._prealloc_full = prealloc_full
        # N-D shard slices are non-contiguous, so irecv lands in a scratch buffer
        # and is copied into ``prealloc_full`` here in wait() as ``(slice, buf)``.
        self._deferred = deferred

    def wait(self):
        """Block on outstanding handles and return the full tensor (owner) or None."""
        for handle in self._handles:
            if handle is not None:
                handle.wait()
        self._handles = []
        if self.full_tensor is not None:
            return self.full_tensor
        if self._prealloc_full is not None:
            if self._deferred:
                for slice_spec, buf in self._deferred:
                    self._prealloc_full[slice_spec] = buf
                self._deferred = None
            self._keep_alive = []
            return self._prealloc_full
        if self._shards is None:
            self._keep_alive = []
            return None
        shards = self._shards
        self._shards = None
        self._keep_alive = []
        if len(shards) == 1:
            return shards[0]
        return mint.cat(shards, dim=self._concat_dim)


def _infer_full_tensor_ops(dtensor, rank_id):
    """Infer redistribution ops from the current DTensor layout to fully replicated."""
    from_layout = dtensor.layout
    replicated_layout = cp.deepcopy(from_layout)
    replicated_placements = [Replicate()] * len(replicated_layout.mesh_shape)
    replicated_layout.set_placements(replicated_placements)
    replicated_layout.placement_to_tensor_map(len(dtensor.to_local().shape))
    replicated_layout.reset_partial()

    inferrer = RedistributionOperatorInfer(
        dev_mat=from_layout.mesh_shape,
        in_tensor_map=list(from_layout.tensor_map),
        out_tensor_map=list(replicated_layout.tensor_map),
    )
    return inferrer.infer_ops_list(rank_id, from_layout.rank_list)


def _placements_cache_key(placements):
    """Build a cache key for DTensor placements."""
    return tuple(repr(placement) for placement in placements)


def _get_full_tensor_ops(local_tensor, device_mesh, placements, rank_id):
    """Return cached redistribution ops for a stable DTensor layout."""
    cache_key = (
        id(device_mesh),
        _placements_cache_key(placements),
        tuple(int(dim) for dim in local_tensor.shape),
        int(rank_id),
    )
    if cache_key not in _FULL_TENSOR_OP_CACHE:
        dtensor = DTensor.from_local(local_tensor, device_mesh, placements)
        _FULL_TENSOR_OP_CACHE[cache_key] = _infer_full_tensor_ops(dtensor, rank_id)
    return _FULL_TENSOR_OP_CACHE[cache_key]


def _get_all_concat_group(rank_list):
    """Return cached communication group for an all_concat rank list."""
    cache_key = tuple(int(rank) for rank in rank_list)
    if cache_key not in _ALL_CONCAT_GROUP_CACHE:
        _ALL_CONCAT_GROUP_CACHE[cache_key] = _HP_PLATFORM.create_group(list(cache_key))
    return _ALL_CONCAT_GROUP_CACHE[cache_key]


def _slice_full_tensor_for_layout_rank(full_tensor, layout, rank):
    """Slice ``full_tensor`` exactly as ``distribute_tensor(...).to_local()`` would for ``rank``."""
    rank_list = tuple(int(item) for item in layout.rank_list)
    inner_rank_id = rank_list.index(int(rank))
    slice_area = _infer_slice_area_by_rank(
        layout.mesh_shape,
        layout.tensor_map,
        inner_rank_id,
        full_tensor.shape,
    )
    slice_spec = tuple(slice(begin, end) for begin, end in slice_area)
    return full_tensor[slice_spec].clone()


def _list_full_tensor_local_shards(full_tensor, layout):
    """Return every rank's local shard in DTensor layout rank order."""
    rank_list = tuple(int(rank) for rank in layout.rank_list)
    return [
        _slice_full_tensor_for_layout_rank(full_tensor, layout, rank)
        for rank in rank_list
    ]


def _list_full_tensor_local_shards_cached(
    full_tensor, rank_list_tuple, mesh_shape_tuple, tensor_map_list,
):
    """Cached-tuple variant: returns shards in ``rank_list_tuple`` order.

    Skips the per-call ``layout.rank_list`` / ``mesh_shape`` / ``tensor_map``
    property accesses that :func:`_list_full_tensor_local_shards` would do.
    """
    full_shape = full_tensor.shape
    shards = []
    for inner_rank_id in range(len(rank_list_tuple)):
        slice_area = _infer_slice_area_by_rank(
            mesh_shape_tuple, tensor_map_list, inner_rank_id, full_shape)
        slice_spec = tuple(slice(int(begin), int(end)) for begin, end in slice_area)
        shards.append(full_tensor[slice_spec].clone())
    return shards


def _start_full_tensor_async(local_tensor, device_mesh, placements, rank_id):
    """Try to start an async DTensor full-tensor gather for simple all_concat layouts."""
    try:
        op_list = _get_full_tensor_ops(local_tensor, device_mesh, placements, rank_id)
    except (AttributeError, TypeError, ValueError):
        return None
    if not op_list:
        return _AsyncAllConcatTensor(local_tensor, None, 1, 0)
    if len(op_list) != 1 or op_list[0][0] != "all_concat":
        return None

    concat_dim, concat_size, rank_list = op_list[0][1]
    if int(concat_size) <= 1:
        return _AsyncAllConcatTensor(local_tensor, None, concat_size, concat_dim)
    group = _get_all_concat_group(rank_list)
    output, handle = comm_func.all_gather_into_tensor(
        None, local_tensor, group=group, async_op=True)
    return _AsyncAllConcatTensor(output, handle, concat_size, concat_dim)


def _start_full_tensor_p2p_gather_async(
    local_tensor, device_mesh, placements, rank_id, assigned_rank, layout=None,
    rank_list_tuple=None, mesh_shape_tuple=None, tensor_map_list=None,
    full_shape_tuple=None,
):
    """P2P gather: non-owner ranks isend their shard, owner irecv assembles. Returns
    None when the layout needs reductions (Partial) or doesn't tile one shard per rank,
    so the caller falls back to all-gather."""
    try:
        op_list = _get_full_tensor_ops(local_tensor, device_mesh, placements, rank_id)
    except (AttributeError, TypeError, ValueError):
        return None
    if not op_list:
        return _PendingP2PGather(full_tensor=local_tensor)
    if any(op[0] != "all_concat" for op in op_list):
        return None
    if len(op_list) > 1:
        return _start_full_tensor_p2p_gather_multi(
            local_tensor, rank_id, int(assigned_rank), layout,
            rank_list_tuple=rank_list_tuple,
            mesh_shape_tuple=mesh_shape_tuple,
            tensor_map_list=tensor_map_list,
            full_shape_tuple=full_shape_tuple,
        )

    concat_dim, concat_size, rank_list = op_list[0][1]
    concat_size = int(concat_size)
    rank_list = tuple(int(r) for r in rank_list)

    if concat_size <= 1:
        return _PendingP2PGather(full_tensor=local_tensor)

    # Layout subsets: ranks outside ``rank_list`` are routed around via
    # ``skip_redist_comm`` in Phase 0, so we only require owner + current to be in.
    assigned_rank = int(assigned_rank)
    if assigned_rank not in rank_list or int(rank_id) not in rank_list:
        return None

    if int(rank_id) != assigned_rank:
        # ns_inputs_local from _prepare_muon_input is already contiguous —
        # skip the redundant ViewCopy.
        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()
        handle = isend(local_tensor, dst=assigned_rank)
        return _PendingP2PGather(handles=[handle], keep_alive=[local_tensor])

    # Owner. When concat_dim == 0 pre-allocate one contiguous full buffer and
    # irecv directly into its row slices, avoiding the post-recv ``mint.cat``.
    can_prealloc = concat_dim == 0
    if can_prealloc:
        local_shape = local_tensor.shape
        full_shape = list(local_shape)
        full_shape[0] *= concat_size
        prealloc_full = mint.empty(tuple(full_shape), dtype=local_tensor.dtype)
        local_rank_list = list(rank_list)
        my_pos = local_rank_list.index(int(rank_id))
        shard_rows = int(local_shape[0])
        start_row = int(my_pos) * shard_rows
        # HCCL irecv requires contiguous destination slices.
        test_slice = prealloc_full[0:shard_rows]
        if not test_slice.is_contiguous():
            can_prealloc = False
        else:
            prealloc_full[start_row:start_row + shard_rows].copy_(local_tensor)
            handles = []
            for pos, src_rank in enumerate(local_rank_list):
                if pos == my_pos:
                    continue
                start = pos * shard_rows
                buf_slice = prealloc_full[start:start + shard_rows]
                handles.append(irecv(buf_slice, src=src_rank))
            return _PendingP2PGather(
                handles=handles,
                prealloc_full=prealloc_full,
            )
    shards = []
    handles = []
    for src_rank in rank_list:
        if int(src_rank) == int(rank_id):
            shards.append(local_tensor)
            continue
        buf = mint.empty_like(local_tensor)
        shards.append(buf)
        handles.append(irecv(buf, src=int(src_rank)))
    return _PendingP2PGather(
        handles=handles,
        shards=shards,
        concat_dim=int(concat_dim),
    )


def _start_full_tensor_p2p_gather_multi(
    local_tensor, rank_id, assigned_rank, layout,
    rank_list_tuple=None, mesh_shape_tuple=None, tensor_map_list=None,
    full_shape_tuple=None,
):
    """P2P gather for N-D sharded layouts (e.g. DP × TP). Owner slice-places each
    incoming shard into a pre-allocated full tensor via :func:`_infer_slice_area_by_rank`
    — the inverse of ``distribute_tensor(...).to_local()`` — matching the all-gather
    path bit-for-bit. The ``*_tuple`` / ``*_list`` kwargs carry frozen layout metadata
    cached at init so the hot path skips ``layout.*`` property access."""
    if rank_list_tuple is not None and mesh_shape_tuple is not None \
            and tensor_map_list is not None and full_shape_tuple is not None:
        rank_list = rank_list_tuple
        mesh_shape = mesh_shape_tuple
        tensor_map = tensor_map_list
        full_shape = full_shape_tuple
    else:
        if layout is None:
            return None
        try:
            rank_list = tuple(int(r) for r in layout.rank_list)
            mesh_shape = tuple(int(d) for d in layout.mesh_shape)
            tensor_map = list(layout.tensor_map)
            full_shape = tuple(int(d) for d in layout.get_global_shape(local_tensor.shape))
        except (AttributeError, TypeError, ValueError):
            return None

    mesh_numel = 1
    for dim in mesh_shape:
        mesh_numel *= int(dim)
    # Require one unique shard per rank for the gather assembly.
    if mesh_numel != len(rank_list):
        return None

    rank_id = int(rank_id)
    assigned_rank = int(assigned_rank)
    if assigned_rank not in rank_list or rank_id not in rank_list:
        return None

    if rank_id != assigned_rank:
        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()
        handle = isend(local_tensor, dst=assigned_rank)
        return _PendingP2PGather(handles=[handle], keep_alive=[local_tensor])

    # Owner: N-D slices are non-contiguous, so placement is deferred to wait().
    prealloc_full = mint.empty(full_shape, dtype=local_tensor.dtype)
    handles = []
    keep_alive = []
    deferred = []
    for inner_rank_id, src_rank in enumerate(rank_list):
        slice_area = _infer_slice_area_by_rank(
            mesh_shape, tensor_map, inner_rank_id, full_shape)
        slice_spec = tuple(slice(int(begin), int(end)) for begin, end in slice_area)
        if int(src_rank) == rank_id:
            prealloc_full[slice_spec] = local_tensor
            continue
        buf = mint.empty_like(local_tensor)
        handles.append(irecv(buf, src=int(src_rank)))
        keep_alive.append(buf)
        deferred.append((slice_spec, buf))
    return _PendingP2PGather(
        prealloc_full=prealloc_full,
        handles=handles,
        deferred=deferred,
        keep_alive=keep_alive,
    )


def newton_schulz(x, dim_a, dim_b, eps, ns_steps, ns_coefficients, matmul_op, addmm_op):
    """Apply Newton-Schulz iteration."""
    a, b, c = ns_coefficients

    if dim_a > dim_b:
        x = x.mT
    # Ensure spectral norm is at most 1
    x = mint.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=eps)
    # Perform the NS iterations
    for _ in range(ns_steps):
        # a_mat = x @ x.T
        a_mat = matmul_op(x, x.mT)

        # b_mat = b * a_mat + c * (a_mat @ a_mat)
        b_mat = addmm_op(a_mat, a_mat, a_mat, beta=b, alpha=c)

        # x = a * x + (b_mat @ x)
        x = addmm_op(x, b_mat, x, beta=a, alpha=1)
    if dim_a > dim_b:
        x = x.mT
    return x


def _apply_muon_ns(
    ns_inputs, muon_split_fn, muon_merge_fn, param_name,
    eps, ns_steps, ns_coefficients, matmul_op, addmm_op,
    lr, matched_adamw_rms
):
    """Run Newton-Schulz on the full tensor: split → NS per piece → merge → scale."""
    op_cast = P.Cast()
    ns_inputs_list = muon_split_fn(param_name, ns_inputs)
    x_list = []

    for ns_inputs_item in ns_inputs_list:
        dim_a, dim_b = ns_inputs_item.shape[-2:]
        x = newton_schulz(
            ns_inputs_item, dim_a, dim_b, eps, ns_steps, ns_coefficients, matmul_op, addmm_op)
        adjusted_lr = lr * mint.sqrt(op_cast(max(dim_a, dim_b), mstype.float32)) * \
            matched_adamw_rms
        x = adjusted_lr * x
        x_list.append(x)
    return muon_merge_fn(param_name, x_list).contiguous()


def _apply_muon_ns_batched(
    full_tensors, param_names, lrs,
    muon_split_fn, muon_merge_fn,
    eps, ns_steps, ns_coefficients, matched_adamw_rms,
):
    """Run Newton-Schulz on ``K`` same-shape weights in one bmm-batched pass.
    Stacks pieces across weights on a new batch dim so ``K * 15``-ish kernel
    launches collapse into ``15``-ish per piece. Per-weight ``lr`` may differ."""
    op_cast = P.Cast()
    n_weights = len(full_tensors)
    if n_weights == 1:
        return [_apply_muon_ns(
            full_tensors[0], muon_split_fn, muon_merge_fn, param_names[0],
            eps, ns_steps, ns_coefficients, mint.mm, mint.addmm,
            lrs[0], matched_adamw_rms)]

    # Split each weight into pieces — same rule → matching shapes.
    pieces_per_weight = [
        muon_split_fn(name, t) for name, t in zip(param_names, full_tensors)
    ]
    n_pieces = len(pieces_per_weight[0])

    # ``lrs`` are usually a shared Python float (one lr per param group).  Check
    # cheaply; only build a per-weight scale tensor when they actually differ.
    lrs_all_same = all(lr is lrs[0] or lr == lrs[0] for lr in lrs)

    out_pieces_per_weight = [[] for _ in range(n_weights)]
    for piece_idx in range(n_pieces):
        per_weight_pieces = [pieces_per_weight[w][piece_idx] for w in range(n_weights)]
        ref_shape = per_weight_pieces[0].shape
        dim_a, dim_b = int(ref_shape[-2]), int(ref_shape[-1])

        # Stack across weights on a new leading batch dim.  Piece may be 2D (most
        # rules) — stack → 3D, bmm-friendly.  Piece may be 3D (reshape_concat /
        # reshape_only) — stack → 4D, flatten ``(K * B)`` for bmm then unflatten.
        stack = mint.stack(per_weight_pieces, dim=0)
        if stack.dim() == 3:
            x_batched = newton_schulz(
                stack, dim_a, dim_b, eps, ns_steps, ns_coefficients,
                mint.bmm, mint.baddbmm)
        elif stack.dim() == 4:
            k_dim, b_dim, m_dim, n_dim = stack.shape
            x_flat = newton_schulz(
                stack.reshape(k_dim * b_dim, m_dim, n_dim),
                m_dim, n_dim, eps, ns_steps, ns_coefficients,
                mint.bmm, mint.baddbmm)
            x_batched = x_flat.reshape(k_dim, b_dim, m_dim, n_dim)
        else:
            raise ValueError(
                f"_apply_muon_ns_batched: unexpected piece rank {stack.dim()} "
                f"(piece shape {tuple(ref_shape)})"
            )

        base_scale = mint.sqrt(
            op_cast(max(dim_a, dim_b), mstype.float32)) * matched_adamw_rms
        if lrs_all_same:
            x_batched = (lrs[0] * base_scale) * x_batched
        else:
            # Broadcast (K,) per-weight scale over the batched output.
            scale_vec = Tensor(
                [float(lr) for lr in lrs], dtype=mstype.float32) * base_scale
            x_batched = x_batched * scale_vec.reshape(
                (n_weights,) + (1,) * (x_batched.dim() - 1))

        # Unstack into per-weight pieces.  ``x_batched[k]`` is a view; merge_fn
        # may reshape/cat so a contiguous owner is fine downstream.
        for k in range(n_weights):
            out_pieces_per_weight[k].append(x_batched[k])

    return [
        muon_merge_fn(name, pieces).contiguous()
        for name, pieces in zip(param_names, out_pieces_per_weight)
    ]


def _apply_muon_update(
    gradient, muon_m, momentum, use_nesterov, param, lr, weight_decay,
    matched_adamw_rms, muon_split_fn, muon_merge_fn, param_name,
    eps, ns_steps, ns_coefficients):
    """Apply Muon optimizer update (original sequential allgather strategy).

    Works for both DTensor (multi-card) and regular Tensor (single-card) gradients.
    When ``gradient`` is a 2D DTensor, the full tensor is gathered to run Newton-Schulz
    and the result is redistributed back to the original sharding before applying the
    update. All other cases (regular Tensor, or non-2D DTensor) operate on the local
    tensor directly and skip the gather/redistribute entirely.
    """
    op_cast = P.Cast()
    ndim = len(gradient.shape)

    if ndim == 2:
        matmul_op = mint.mm
        addmm_op = mint.addmm
    elif ndim == 3:
        matmul_op = mint.bmm
        addmm_op = mint.baddbmm
    else:
        raise ValueError(f"newton_schulz only supports 2D or 3D gradient, got shape={gradient.shape}")

    needs_dtensor_redist = isinstance(gradient, DTensor) and len(gradient.shape) == 2
    device_mesh = gradient.device_mesh if needs_dtensor_redist else None
    placements = gradient.placements if needs_dtensor_redist else None

    gradient = _to_local(gradient)
    muon_m = _to_local(muon_m)

    m_fp32 = op_cast(muon_m, mstype.float32)
    gradient_fp32 = op_cast(gradient, mstype.float32)
    next_m = m_fp32 * momentum + gradient_fp32

    if use_nesterov:
        gradient_fp32 = gradient_fp32 + next_m * momentum
    else:
        gradient_fp32 = next_m

    ns_inputs = op_cast(gradient_fp32, mstype.bfloat16)
    if needs_dtensor_redist:
        ns_inputs = DTensor.from_local(ns_inputs, device_mesh, placements).full_tensor()

    x_ret = _apply_muon_ns(
        ns_inputs, muon_split_fn, muon_merge_fn, param_name,
        eps, ns_steps, ns_coefficients, matmul_op, addmm_op,
        lr, matched_adamw_rms)

    if needs_dtensor_redist:
        x_ret = distribute_tensor(x_ret, device_mesh, placements).to_local()

    with SkipDTensorDispatch():
        param_fp32 = op_cast(param, mstype.float32) * (1 - lr * weight_decay)
        next_param = param_fp32 - x_ret.reshape(param_fp32.shape)
        param.copy_(op_cast(next_param, F.dtype(param)))
        muon_m.copy_(op_cast(next_m, F.dtype(muon_m)))
    return op_cast(next_param, F.dtype(param))


# ---------------------------------------------------------------------------
#  Batched muon update for ``allgather_deredundency`` strategy.
#
#  The key insight: instead of processing weights one-by-one (which leaves
#  non-assigned ranks idle while the assigned rank computes Newton-Schulz),
#  we batch all muon weights into four phases:
#
#    Phase 0 – Prepare        : momentum update (local, no comm)
#    Phase 1 – Gather         : 2D sharded inputs are P2P-gathered to the
#                               assigned rank (non-owner ranks only ``isend``
#                               their shard and never materialize the full
#                               tensor); falls back to an all-gather for
#                               layouts that do not cover the full worker
#                               group.
#    Phase 2 – Compute        : local 3D/non-DTensor weights run NS on every
#                               rank; 2D DTensors run NS only on the assigned
#                               rank after the full tensor is available
#    Phase 3 – Shard send     : send each rank only the local 2D shard it needs,
#                               with a full broadcast fallback
#    Phase 4 – Apply          : write param / momentum
#
#  This removes redundant NS for sharded 2D weights while preserving the
#  allgather-mode local path for 3D weights such as expert batches.
# ---------------------------------------------------------------------------

def _prepare_muon_input_compute(gradient, muon_m, momentum, use_nesterov):
    """Phase 0 hot path: compute momentum-updated gradient (local, no comm).

    Returns ``(ns_inputs_local, next_m_fp32)``.  This deliberately does *not*
    touch ``gradient.layout`` / ``gradient.device_mesh`` / ``gradient.shape``
    — those go through hyper_parallel's Layout machinery and dominate
    Python-side overhead.  Mesh metadata is fetched once at init via
    :meth:`Muon._recompute_muon_assigned_ranks` and threaded in via
    ``muon_metas``.

    Uses ``mint.add(input, other, alpha=momentum)`` to fuse the
    ``input + momentum * other`` pattern (was ``other * momentum + input``,
    a separate Mul + Add) — saves one kernel launch per term per weight.
    """
    op_cast = P.Cast()
    gradient_local = _to_local(gradient)
    muon_m_local = _to_local(muon_m)

    m_fp32 = op_cast(muon_m_local, mstype.float32)
    grad_fp32 = op_cast(gradient_local, mstype.float32)
    # next_m = momentum * m + grad
    next_m = mint.add(grad_fp32, m_fp32, alpha=momentum)

    if use_nesterov:
        # ns_input = grad + momentum * next_m
        ns_input = mint.add(grad_fp32, next_m, alpha=momentum)
    else:
        ns_input = next_m

    ns_inputs_local = op_cast(ns_input, mstype.bfloat16)
    return ns_inputs_local, next_m


def _prepare_muon_input(gradient, muon_m, momentum, use_nesterov):
    """Phase 0 slow path: compute momentum-updated gradient *and* extract
    DTensor metadata via ``gradient.layout`` / ``device_mesh`` / ``placements``.

    Kept for the legacy code path (``muon_metas=None``) where metadata has
    not been pre-frozen.  Returns
    ``(ns_inputs_local, next_m_fp32, needs_redist, dev_mesh, placements, layout)``.
    """
    ndim = len(gradient.shape)
    if ndim not in (2, 3):
        raise ValueError(
            f"newton_schulz only supports 2D or 3D gradient, got shape={gradient.shape}")

    needs_redist = isinstance(gradient, DTensor) and ndim == 2
    dev_mesh = gradient.device_mesh if needs_redist else None
    plmts = gradient.placements if needs_redist else None
    layout = gradient.layout if needs_redist else None

    ns_inputs_local, next_m = _prepare_muon_input_compute(
        gradient, muon_m, momentum, use_nesterov)
    return ns_inputs_local, next_m, needs_redist, dev_mesh, plmts, layout


def _run_muon_batched(
    muon_gradients, muon_params, muon_m_ms,
    muon_lrs, muon_wds,
    muon_param_names, muon_assigned_ranks,
    muon_momentum, use_nesterov,
    matched_adamw_rms, muon_split_fn, muon_merge_fn,
    eps, ns_steps, ns_coefficients,
    rank_id,
    overlap_callback=None,
    muon_metas=None,
):
    """Batched Muon update with parallel Newton-Schulz across ranks.

    Only called when ``comm_strategy == "allgather_deredundency"``.

    ``muon_metas`` is a parallel array (same length as ``muon_gradients``)
    of frozen per-param DTensor metadata produced by
    :meth:`Muon._recompute_muon_assigned_ranks`.  When present, Phase 0 reads
    rank_list / mesh_shape / tensor_map / full_shape / dev_mesh / placements
    from it instead of touching ``gradient.layout`` every step.  When
    absent (legacy call path) the per-step layout introspection is used.

    Returns a list of updated parameters (same length as input lists).
    """
    op_cast = P.Cast()
    n_weights = len(muon_gradients)

    def _materialize_full_tensor(info):
        """Wait for this weight's async allgather only when its full tensor is needed."""
        pending_full = info.pop('pending_full_tensor', None)
        if pending_full is not None:
            info['ns_inputs_full'] = pending_full.wait()
        return info['ns_inputs_full']

    def _apply_prepared_update(info):
        """Apply one prepared Muon result to parameter and momentum state."""
        x_ret = info['x_ret']
        if info['needs_redist'] and not info.get('x_ret_is_local', False):
            x_ret = distribute_tensor(
                x_ret, info['dev_mesh'], info['placements']).to_local()

        param = info['param']
        muon_m = info['muon_m']
        with SkipDTensorDispatch():
            param_fp32 = op_cast(param, mstype.float32) * (1 - info['lr'] * info['wd'])
            next_param = param_fp32 - x_ret.reshape(param_fp32.shape)
            param.copy_(op_cast(next_param, F.dtype(param)))
            muon_m.copy_(op_cast(info['next_m'], F.dtype(muon_m)))
        return param

    def _try_start_local_shard_scatter(info, x_ret_full):
        """Use P2P sends so each rank receives only its local shard.

        Layout subsets are fine: ranks outside ``rank_list`` are skipped
        upstream via ``info['skip_redist_comm']`` and never enter this
        function, so we only need the owner and the current rank to be in
        ``rank_list``.  If the gating below fails the caller falls back to
        a world broadcast — that fallback assumes every rank in the world
        group participates, which is only safe when the layout itself covers
        the whole group, hence the optional broadcast-fallback gate added at
        the call site.

        Reads ``info['rank_list_tuple']`` / ``mesh_shape_tuple`` / ``tensor_map_list``
        when available so the per-step ``layout`` property accesses are avoided.
        """
        rank_list = info.get('rank_list_tuple')
        if rank_list is None:
            layout = info['layout']
            rank_list = tuple(int(rank) for rank in layout.rank_list)
        assigned_rank = int(info['assigned_rank'])
        if assigned_rank not in rank_list or int(rank_id) not in rank_list:
            return False

        if len(rank_list) == 1:
            info['x_ret'] = x_ret_full
            info['x_ret_is_local'] = True
            return True

        if int(rank_id) == assigned_rank:
            mesh_shape = info.get('mesh_shape_tuple')
            tensor_map = info.get('tensor_map_list')
            if mesh_shape is not None and tensor_map is not None:
                shards = _list_full_tensor_local_shards_cached(
                    x_ret_full, rank_list, mesh_shape, tensor_map)
            else:
                shards = _list_full_tensor_local_shards(x_ret_full, info['layout'])
            p2p_handles = []
            p2p_tensors = []
            local_output = None
            for dst_rank, shard in zip(rank_list, shards):
                if dst_rank == int(rank_id):
                    local_output = shard
                else:
                    p2p_tensors.append(shard)
                    p2p_handles.append(isend(shard, dst=dst_rank))
            info['x_ret'] = local_output
            info['p2p_tensors'] = p2p_tensors
            info['p2p_handles'] = p2p_handles
        else:
            local_output = mint.empty_like(info['ns_inputs_local'])
            info['x_ret'] = local_output
            info['p2p_handles'] = [irecv(local_output, src=assigned_rank)]

        info['x_ret_is_local'] = True
        return True

    # ------------------------------------------------------------------
    # Phase 0 — Prepare (all local, no communication)
    # ------------------------------------------------------------------
    # Per-weight: momentum update, cast to bf16, read frozen mesh metadata,
    # and decide whether this rank participates in this weight's redist.
    prepared = []
    for i in range(n_weights):
        meta = muon_metas[i] if muon_metas is not None else None

        if meta is not None:
            # Fast path: metadata frozen at first construct; skip per-step
            # DTensor introspection entirely.
            ns_inputs_local, next_m = _prepare_muon_input_compute(
                muon_gradients[i], muon_m_ms[i], muon_momentum, use_nesterov)
            ndim = meta['ndim']
            needs_redist = meta['needs_redist']
            dev_mesh = meta['dev_mesh']
            plmts = meta['placements']
            layout = meta['layout']
            rank_list_tuple = meta['rank_list_tuple']
            mesh_shape_tuple = meta['mesh_shape_tuple']
            tensor_map_list = meta['tensor_map_list']
            full_shape_tuple = meta['full_shape_tuple']
            layout_covers_world = meta['layout_covers_world']
            skip_redist_comm = meta['skip_redist_comm']
            group_sig = meta.get('group_sig')
        else:
            # Slow path: derive everything from gradient.layout (legacy).
            ns_inputs_local, next_m, needs_redist, dev_mesh, plmts, layout = \
                _prepare_muon_input(
                    muon_gradients[i], muon_m_ms[i], muon_momentum, use_nesterov)
            ndim = len(muon_gradients[i].shape)
            rank_list_tuple = None
            mesh_shape_tuple = None
            tensor_map_list = None
            full_shape_tuple = None
            group_sig = None
            # A 2D DTensor whose layout does not include this rank has no local
            # shard here; subsequent phases must not enter its gather/scatter
            # collective or they would either hang on a broadcast that nobody
            # sources or burn a no-op allgather on a sub-group we are outside of.
            skip_redist_comm = False
            layout_covers_world = False
            if needs_redist and layout is not None:
                rank_list_tuple = tuple(int(r) for r in layout.rank_list)
                if int(rank_id) not in rank_list_tuple:
                    skip_redist_comm = True
                layout_covers_world = len(rank_list_tuple) == get_group_size()

        if ndim == 2:
            matmul_op = mint.mm
            addmm_op = mint.addmm
        else:
            matmul_op = mint.bmm
            addmm_op = mint.baddbmm

        prepared.append({
            'ns_inputs_local': ns_inputs_local,
            'next_m': next_m,
            'needs_redist': needs_redist,
            'dev_mesh': dev_mesh,
            'placements': plmts,
            'layout': layout,
            'rank_list_tuple': rank_list_tuple,
            'mesh_shape_tuple': mesh_shape_tuple,
            'tensor_map_list': tensor_map_list,
            'full_shape_tuple': full_shape_tuple,
            'matmul_op': matmul_op,
            'addmm_op': addmm_op,
            'param': muon_params[i],
            'muon_m': muon_m_ms[i],
            'lr': muon_lrs[i],
            'wd': muon_wds[i],
            'param_name': muon_param_names[i],
            'assigned_rank': muon_assigned_ranks[i],
            'skip_redist_comm': skip_redist_comm,
            'layout_covers_world': layout_covers_world,
            'group_sig': group_sig,
        })

    # ------------------------------------------------------------------
    # Phase 1 — Gather inputs to the assigned rank
    # ------------------------------------------------------------------
    # P2P gather is preferred: each non-owner rank only ``isend`` its shard,
    # so non-owners never receive the (N-1)*shard bytes they would discard.
    # Layout subsets are supported — ranks outside the layout skip the weight
    # entirely via ``skip_redist_comm`` so no broadcast / collective is left
    # dangling.  Analysis failures fall back to the sync ``DTensor.full_tensor()``.
    #
    # We iterate in param order (the same order on every rank) — HCCL's
    # auto-tagging for tag-less isend/irecv depends on the rank-global P2P
    # launch sequence matching between peers, so any per-rank reordering
    # (e.g. owner-first) trips ``HcomRecv ret:4`` mid-training.
    for info in prepared:
        if not info['needs_redist']:
            info['ns_inputs_full'] = info['ns_inputs_local']
            continue
        if info['skip_redist_comm']:
            # This rank holds no shard of this weight; nothing to gather.
            info['ns_inputs_full'] = info['ns_inputs_local']
            continue
        pending_full = _start_full_tensor_p2p_gather_async(
            info['ns_inputs_local'],
            info['dev_mesh'],
            info['placements'],
            rank_id,
            info['assigned_rank'],
            info['layout'],
            rank_list_tuple=info['rank_list_tuple'],
            mesh_shape_tuple=info['mesh_shape_tuple'],
            tensor_map_list=info['tensor_map_list'],
            full_shape_tuple=info['full_shape_tuple'],
        )
        if pending_full is None:
            pending_full = _start_full_tensor_async(
                info['ns_inputs_local'],
                info['dev_mesh'],
                info['placements'],
                rank_id,
            )
        if pending_full is None:
            info['ns_inputs_full'] = DTensor.from_local(
                info['ns_inputs_local'],
                info['dev_mesh'],
                info['placements'],
            ).full_tensor()
        else:
            info['pending_full_tensor'] = pending_full

    # Local weights, including every 3D Muon weight, do not depend on the
    # 2D allgather result.  Run them while async allgathers are in flight.
    # Same-group same-shape weights (e.g. all `mlp.experts.weight1` across
    # layers) are batched into a single bmm-NS via ``_apply_muon_ns_batched``
    # — for a 24-layer MoE this collapses ~46 sequential NS calls into ~2.
    local_groups = {}
    local_singletons = []
    for info in prepared:
        if info['needs_redist']:
            continue
        sig = info.get('group_sig')
        if sig is None:
            local_singletons.append(info)
        else:
            local_groups.setdefault(sig, []).append(info)

    for info in local_singletons:
        info['x_ret'] = _apply_muon_ns(
            info['ns_inputs_full'],
            muon_split_fn, muon_merge_fn, info['param_name'],
            eps, ns_steps, ns_coefficients,
            info['matmul_op'], info['addmm_op'],
            info['lr'], matched_adamw_rms)

    for sig, infos in local_groups.items():
        if len(infos) == 1:
            info = infos[0]
            info['x_ret'] = _apply_muon_ns(
                info['ns_inputs_full'],
                muon_split_fn, muon_merge_fn, info['param_name'],
                eps, ns_steps, ns_coefficients,
                info['matmul_op'], info['addmm_op'],
                info['lr'], matched_adamw_rms)
            continue
        full_tensors = [info['ns_inputs_full'] for info in infos]
        x_rets = _apply_muon_ns_batched(
            full_tensors,
            [info['param_name'] for info in infos],
            [info['lr'] for info in infos],
            muon_split_fn, muon_merge_fn,
            eps, ns_steps, ns_coefficients, matched_adamw_rms,
        )
        for info, x_ret in zip(infos, x_rets):
            info['x_ret'] = x_ret

    if overlap_callback is not None:
        overlap_callback()

    # ------------------------------------------------------------------
    # Phase 2/3 — Materialize gather, batched Newton-Schulz, P2P scatter
    # ------------------------------------------------------------------
    # Iterate weights sorted by (group_sig, original index).  All ranks use
    # the same sort key because group_sig is derived deterministically from
    # the model (piece shapes + rank_list), so the HCCL tag-less P2P sequence
    # stays synchronized — same property the previous param-order loop relied
    # on.  Consecutive owned weights from the same group are buffered and run
    # through a single bmm-batched ``_apply_muon_ns_batched`` instead of N
    # individual NS calls, collapsing N*15-ish kernel launches into 15-ish per
    # piece.  Non-owned weights and outside-layout weights are processed in
    # place inside the same iteration so their P2P scatter recv / placeholder
    # x_ret bookkeeping happens in the same flat order on every rank.
    flat_order_2d = sorted(
        (idx for idx, info in enumerate(prepared) if info['needs_redist']),
        key=lambda idx: (
            prepared[idx].get('group_sig') if prepared[idx].get('group_sig') is not None
            else ('__ungrouped__', idx),
            idx,
        ),
    )

    # Buffer of (info, ns_inputs_full) for the current owner-batched group.
    batch_buffer = []
    batch_group_sig = None

    def _flush_owned_batch():
        nonlocal batch_buffer, batch_group_sig
        if not batch_buffer:
            return
        # Run NS — single weight goes through the legacy path, batch>=2 uses bmm.
        full_tensors = [ns for _, ns in batch_buffer]
        infos = [info for info, _ in batch_buffer]
        x_ret_list = _apply_muon_ns_batched(
            full_tensors,
            [info['param_name'] for info in infos],
            [info['lr'] for info in infos],
            muon_split_fn, muon_merge_fn,
            eps, ns_steps, ns_coefficients, matched_adamw_rms,
        )
        # Scatter each in the buffer's iteration order (= flat-order subset on
        # this rank); the same param_index order is observed by every peer's
        # irecv path below, so per-channel FIFO is preserved.
        for info, x_ret_full in zip(infos, x_ret_list):
            if _try_start_local_shard_scatter(info, x_ret_full):
                continue
            if not info['layout_covers_world']:
                raise RuntimeError(
                    f"Muon allgather_deredundency: cannot fall back to world "
                    f"broadcast for param={info['param_name']!r} because its "
                    f"layout does not cover the full worker group."
                )
            info['x_ret'] = x_ret_full
            info['broadcast_handle'] = broadcast(
                info['x_ret'], src=info['assigned_rank'], async_op=True)
        batch_buffer = []
        batch_group_sig = None

    for prep_idx in flat_order_2d:
        info = prepared[prep_idx]
        if info['skip_redist_comm']:
            # Outside this weight's layout: nothing to gather, NS, or scatter.
            # Make sure the previous owned-group is flushed first so this
            # rank's P2P-scatter launches stay ordered relative to peers.
            _flush_owned_batch()
            info['x_ret'] = info['ns_inputs_local']
            info['x_ret_is_local'] = True
            continue

        current_sig = info.get('group_sig')
        is_owner = rank_id == info['assigned_rank']

        if is_owner:
            # Flush whenever the group_sig changes (also flushes a non-owner
            # interleave inside the same group — see below).
            if batch_group_sig is not None and current_sig != batch_group_sig:
                _flush_owned_batch()
            batch_group_sig = current_sig
            batch_buffer.append((info, _materialize_full_tensor(info)))
        else:
            # Non-owner: if we have an owned batch pending, flush it first so
            # its scatter isends are issued before this irecv is queued.
            _flush_owned_batch()
            _materialize_full_tensor(info)  # waits the gather isend handle
            if _try_start_local_shard_scatter(info, None):
                continue
            if not info['layout_covers_world']:
                raise RuntimeError(
                    f"Muon allgather_deredundency: cannot fall back to world "
                    f"broadcast for param={info['param_name']!r} because its "
                    f"layout does not cover the full worker group."
                )
            ns_inputs_full = info.get('ns_inputs_full')
            if ns_inputs_full is not None:
                info['x_ret'] = mint.empty_like(ns_inputs_full)
            else:
                raise RuntimeError(
                    f"Muon allgather_deredundency: scatter requires broadcast "
                    f"fallback for param={info['param_name']!r} but P2P gather "
                    f"already skipped the full-tensor materialization on rank "
                    f"{rank_id}."
                )
            info['broadcast_handle'] = broadcast(
                info['x_ret'], src=info['assigned_rank'], async_op=True)

    # End of iteration — flush any tail owned batch.
    _flush_owned_batch()

    # ------------------------------------------------------------------
    # Phase 4 — Apply updates
    # ------------------------------------------------------------------
    results = [None] * len(prepared)

    # Local/3D weights can be applied while 2D scatters are still in flight.
    for index, info in enumerate(prepared):
        if not info['needs_redist']:
            results[index] = _apply_prepared_update(info)

    for index, info in enumerate(prepared):
        if not info['needs_redist']:
            continue
        p2p_handles = info.pop('p2p_handles', None)
        if p2p_handles is not None:
            for handle in p2p_handles:
                handle.wait()
            # Release send-side keep-alive buffers as soon as handles complete.
            info.pop('p2p_tensors', None)
            results[index] = _apply_prepared_update(info)
        else:
            handle = info.pop('broadcast_handle', None)
            if handle is not None:
                handle.wait()
            results[index] = _apply_prepared_update(info)

    return results


class Muon(Optimizer):
    """
    Muon optimizer implementation for pynative mode.

    Args:
        params: model parameters to optimize.
        learning_rate (float): Learning rate. Default: ``2e-2``.
        weight_decay (float): Weight decay factor. Default: ``0.1``.
        matched_adamw_rms (float): RMS matching parameter for AdamW. Default: ``0.2``.
        momentum (float): Momentum factor. Default: ``0.95``.
        nesterov (bool): Whether to use Nesterov momentum. Default: ``True``.
        ns_steps (int): Number of Newton-Schulz steps. Default: ``5``.
        ns_coefficients (tuple): Newton-Schulz coefficients. Default: ``(3.4445, -4.7750, 2.0315)``.
        adamw_betas (tuple): Beta parameters for AdamW. Default: ``(0.95, 0.95)``.
        adamw_eps (float): Epsilon for AdamW. Default: ``1e-8``.
        qk_clip_enabled (bool): Whether to apply QK clip scaling. Default: ``True``.
        qk_clip_threshold (float): QK clip threshold. Default: ``100``.
        comm_strategy (str): Communication strategy for Newton-Schulz on 2D DTensors.
            - ``"allgather"`` (default): every rank all-gathers and runs NS independently
              (sequential, one weight at a time).
            - ``"allgather_deredundency"``: 2D DTensor weights all-gather on every
              rank, but NS runs only on one assigned rank per weight. 3D/local
              weights run NS independently on each rank, matching allgather mode.
        model: The model model. Default: ``None``.
    """

    def __init__(
        self,
        params,
        learning_rate=2e-2,
        weight_decay=0.1,
        matched_adamw_rms=0.2,
        momentum=0.95,
        eps=1e-7,
        nesterov=True,
        ns_steps=5,
        ns_coefficients=(3.4445, -4.7750, 2.0315),
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        qk_clip_enabled=True,
        qk_clip_threshold=100,
        model=None,
        comm_strategy="allgather",
        **kwargs,
    ):
        super().__init__(learning_rate, params, weight_decay)
        if kwargs.get('swap', False):
            raise ValueError("Muon does not support swap.")

        self._verify_model(model)

        self.beta1, self.beta2 = adamw_betas
        self.adamw_eps = adamw_eps
        self.muon_momentum = momentum
        self.eps = eps
        self.matched_adamw_rms = matched_adamw_rms
        self.use_nesterov = nesterov
        self._verify_config(ns_steps, ns_coefficients, qk_clip_enabled, qk_clip_threshold)
        self.ns_steps = ns_steps
        self.ns_coefficients = tuple(ns_coefficients)
        self.qk_clip_enabled = qk_clip_enabled
        self.param_name_tuple = tuple(p.name for p in self._parameters)

        self.muon_split_fn, self.muon_merge_fn = model.make_model_muon_fns()
        self.logit_threshold = Tensor([qk_clip_threshold], dtype=mstype.float32) if qk_clip_enabled else None

        # ---- comm_strategy ----
        if comm_strategy not in ("allgather", "allgather_deredundency"):
            raise ValueError(
                f"comm_strategy must be 'allgather' or 'allgather_deredundency', "
                f"got {comm_strategy!r}."
            )
        self.comm_strategy = comm_strategy
        if self.comm_strategy == "allgather_deredundency":
            self._rank_id = get_rank()
        else:
            self._rank_id = None  # not used in allgather mode

        self._initialize_state(model)

        self.model = model

    @staticmethod
    def _verify_model(model):
        """Verify if the model is compatible with Muon optimizer."""
        if model is None:
            raise ValueError("Model must be provided for Muon optimizer.")

        if core_context.is_legacy_model():
            raise ValueError("Muon does not support Legacy Model.")

        config = model.get_gpt_transformer_config()

        if not config.multi_latent_attention:
            raise ValueError("Current Muon implementation only supports models with Multi-Latent Attention enabled.")

    @staticmethod
    def _verify_config(ns_steps, ns_coefficients, qk_clip_enabled, qk_clip_threshold):
        """Validate Newton-Schulz and QK-clip hyperparameters."""
        if ns_steps <= 0:
            raise ValueError(f"ns_steps must be positive, got {ns_steps!r}.")
        if len(ns_coefficients) != 3:
            raise ValueError(f"ns_coefficients must have 3 elements, got {ns_coefficients!r}.")
        if qk_clip_enabled and qk_clip_threshold <= 0:
            raise ValueError(f"qk_clip_threshold must be positive, got {qk_clip_threshold!r}.")

    def _initialize_state(self, model):
        """Create Muon momentum and AdamW moment state in one pass over self._parameters."""
        muon_filter = model.get_muon_filter()

        muon_m = []
        moments1 = []
        moments2 = []
        state_indices = []
        use_muon = []

        for param in self._parameters:
            if muon_filter(param):
                state_indices.append(len(muon_m))
                muon_m.append(_create_state_parameter(param, "muon_m"))
                use_muon.append(True)
                logger.info(f"Muon apply: {type(param)=}, {param.name=}")
            else:
                state_indices.append(len(moments1))
                moments1.append(_create_state_parameter(param, "adam_m"))
                moments2.append(_create_state_parameter(param, "adam_v"))
                use_muon.append(False)
                logger.info(f"Adam apply: {type(param)=}, {param.name=}")

        self.muon_m = ParameterTuple(muon_m)
        self.moments1 = ParameterTuple(moments1)
        self.moments2 = ParameterTuple(moments2)
        self.use_muon = tuple(use_muon)
        self.state_indices = tuple(state_indices)

        # ---- Build and cache 2D weight-to-rank assignment for allgather_deredundency ----
        if self.comm_strategy == "allgather_deredundency":
            world_size = get_group_size()
            self._muon_assigned_rank = {}
            muon_work_items = []
            local_muon_count = 0
            for i, is_muon in enumerate(self.use_muon):
                if not is_muon:
                    continue
                shape = self._parameters[i].shape
                if len(shape) == 2:
                    work = _estimate_muon_work(
                        self.param_name_tuple[i], shape, self.muon_split_fn, self.ns_steps)
                    muon_work_items.append((i, work))
                else:
                    local_muon_count += 1

            rank_loads = [0] * world_size
            rank_counts = [0] * world_size
            for param_index, work in sorted(muon_work_items, key=lambda item: item[1], reverse=True):
                assigned_rank = min(range(world_size), key=lambda rank: (rank_loads[rank], rank))
                self._muon_assigned_rank[param_index] = assigned_rank
                rank_loads[assigned_rank] += work
                rank_counts[assigned_rank] += 1

            load_summary = _format_work_load_summary(rank_loads, rank_counts)
            logger.info(
                f"Muon allgather_deredundency: {len(muon_work_items)} 2D muon weights "
                f"provisionally assigned across {world_size} ranks; "
                f"{local_muon_count} 3D/local muon weights run independently on every rank "
                f"(rank_id={self._rank_id}); estimated_loads=[{load_summary}]"
            )
            # The init-time assignment ignores per-weight layout because we
            # have no access to gradient layouts here.  ``_recompute_muon_assigned_ranks``
            # refines this on the first ``construct`` call so each owner is
            # guaranteed to be inside its layout's ``rank_list`` — required by
            # the P2P gather/scatter path when layouts do not cover the full
            # worker group.  At the same time it populates ``self._muon_param_meta``
            # with frozen per-param DTensor metadata so Phase 0 / helpers never
            # have to hit ``gradient.layout`` per step (Python overhead is the
            # main contributor to the device's ``Free`` time in the profile).
            self._muon_assigned_ranks_finalized = False
            self._muon_param_meta = {}
        else:
            self._muon_assigned_rank = None
            self._muon_assigned_ranks_finalized = True
            self._muon_param_meta = {}

    def _recompute_muon_assigned_ranks(self, gradients):
        """Refine the per-weight assigned ranks once using gradient layouts.

        The init-time pass in :meth:`_initialize_state` is a best-effort
        placeholder because layouts are unknown then.  On the first call to
        :meth:`_construct_deredundency` we have real ``DTensor`` gradients and
        can pick owners that always live inside each weight's
        ``layout.rank_list``.  This is what unlocks P2P gather / scatter for
        weights whose layout does not cover the full worker group.

        Idempotent — subsequent calls are no-ops.
        """
        if getattr(self, '_muon_assigned_ranks_finalized', True):
            return
        if not self._muon_assigned_rank:
            self._muon_param_meta = {}
            self._muon_assigned_ranks_finalized = True
            return

        world_size = get_group_size()
        rank_id_int = int(self._rank_id) if self._rank_id is not None else 0
        meta = {}  # param_index -> frozen per-param metadata (see _run_muon_batched)
        items_by_group = {}  # group_sig -> list of (param_index, work, rank_list_tuple)
        weight_group_sig = {}  # param_index -> group_sig
        for i, is_muon in enumerate(self.use_muon):
            if not is_muon:
                continue
            param = self._parameters[i]
            ndim = len(param.shape)
            grad = gradients[i] if i < len(gradients) else None
            entry = {
                'ndim': ndim,
                'needs_redist': False,
                'dev_mesh': None,
                'placements': None,
                'layout': None,
                'rank_list_tuple': None,
                'mesh_shape_tuple': None,
                'tensor_map_list': None,
                'full_shape_tuple': None,
                'layout_covers_world': False,
                'skip_redist_comm': False,
                'group_sig': None,
            }
            rank_list = None
            if ndim == 2 and isinstance(grad, DTensor):
                try:
                    layout = grad.layout
                    rank_list = tuple(int(r) for r in layout.rank_list)
                    mesh_shape = tuple(int(d) for d in layout.mesh_shape)
                    tensor_map = list(layout.tensor_map)
                    local_shape = tuple(int(d) for d in grad.to_local().shape)
                    full_shape = tuple(int(d) for d in layout.get_global_shape(local_shape))
                    entry.update({
                        'needs_redist': True,
                        'dev_mesh': grad.device_mesh,
                        'placements': grad.placements,
                        'layout': layout,
                        'rank_list_tuple': rank_list,
                        'mesh_shape_tuple': mesh_shape,
                        'tensor_map_list': tensor_map,
                        'full_shape_tuple': full_shape,
                        'layout_covers_world': (len(rank_list) == world_size),
                        'skip_redist_comm': rank_id_int not in rank_list,
                    })
                except (AttributeError, TypeError, ValueError):
                    pass  # keep defaults — falls back to per-step path
            meta[i] = entry
            # Group signature for batched NS: same (piece-shape tuple, rank_list)
            # ⇒ weights can be stacked and run as a single bmm-batched NS.
            # 2D DTensors batch on their shared owner (Phase 2/3).
            # 3D / non-redist weights batch rank-locally (Phase 1.5) — they have
            # no owner, every rank just runs NS on its own piece set.
            if ndim in (2, 3):
                shape_tuple = tuple(int(d) for d in param.shape)
                try:
                    piece_shapes = _estimate_muon_piece_shapes(
                        self.param_name_tuple[i], shape_tuple, self.muon_split_fn)
                    piece_shapes_tuple = tuple(
                        tuple(int(d) for d in s) for s in piece_shapes)
                except (TypeError, ValueError, ZeroDivisionError):
                    # Force a singleton group — fall back to per-weight NS for this one.
                    piece_shapes_tuple = (("unknown", i),)
                rl_key = rank_list if rank_list is not None else tuple(range(world_size))
                sig = (piece_shapes_tuple, rl_key)
                weight_group_sig[i] = sig
                # Only 2D weights enter the owner load balancer — 3D weights run
                # locally on every rank so no assignment is needed.
                if ndim == 2:
                    work = _estimate_muon_work(
                        self.param_name_tuple[i], param.shape, self.muon_split_fn, self.ns_steps)
                    items_by_group.setdefault(sig, []).append((i, work, rl_key))

        # Distribute each group's weights as a contiguous chunk per rank inside
        # the group's rank_list.  Sorting weights by param_index inside the group
        # and then mapping ``chunk = (j * n_ranks_in_group) // n`` keeps each
        # rank's owned slice contiguous in the flat iteration order built below,
        # which is what lets Phase 2/3 buffer them into a single batched NS.
        new_assignment = {}
        rank_loads = [0] * world_size
        rank_counts = [0] * world_size
        # Process groups in deterministic order (largest total work first so
        # heaviest groups influence placement before tail groups).
        def _group_total_work(weights):
            return sum(w[1] for w in weights)
        sorted_group_items = sorted(
            items_by_group.items(),
            key=lambda kv: (-_group_total_work(kv[1]), kv[0]),
        )
        for sig, weights in sorted_group_items:
            rank_list_in_group = sig[1]
            n_ranks_in_group = max(1, len(rank_list_in_group))
            weights_sorted = sorted(weights, key=lambda w: w[0])
            n = len(weights_sorted)
            for j, (pidx, work, _) in enumerate(weights_sorted):
                chunk = (j * n_ranks_in_group) // n
                chosen = int(rank_list_in_group[chunk])
                new_assignment[pidx] = chosen
                rank_loads[chosen] += work
                rank_counts[chosen] += 1

        for pidx, sig in weight_group_sig.items():
            meta[pidx]['group_sig'] = sig

        self._muon_assigned_rank = new_assignment
        self._muon_param_meta = meta
        self._muon_assigned_ranks_finalized = True

        # One-line summary of unique 2D groups + max batch size per rank.
        # Only 2D weights get an owner assignment; 3D weights are batched
        # rank-locally (every rank runs NS on its own pieces), so they have
        # no entry in new_assignment.
        rank_group_counts = {r: {} for r in range(world_size)}
        for pidx, sig in weight_group_sig.items():
            r = new_assignment.get(pidx)
            if r is None:
                continue
            rank_group_counts[r][sig] = rank_group_counts[r].get(sig, 0) + 1
        max_batch_per_rank = [
            max(g.values()) if g else 0 for g in rank_group_counts.values()
        ]

        n_3d_groups = sum(
            1 for sig in set(weight_group_sig.values()) if sig not in items_by_group
        )
        load_summary = _format_work_load_summary(rank_loads, rank_counts)
        logger.info(
            f"Muon allgather_deredundency: refined assignment for "
            f"{sum(len(w) for w in items_by_group.values())} 2D muon weights "
            f"across {len(items_by_group)} 2D shape groups + "
            f"{n_3d_groups} 3D shape groups "
            f"(rank_id={self._rank_id}); estimated_loads=[{load_summary}]; "
            f"max_batch_per_rank={max_batch_per_rank}"
        )

    def construct(self, gradients):
        """Construct method for optimizer.

        When ``comm_strategy == "allgather"`` the original sequential loop is used.

        When ``comm_strategy == "allgather_deredundency"`` all muon weights are
        collected and processed in a single batched call. Only 2D DTensors are
        de-redundant; 3D/local weights keep the allgather-mode local update.
        """
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        step = self.global_step
        bias_correction1 = 1.0 - self.beta1 ** step
        bias_correction2 = 1.0 - self.beta2 ** step
        one_minus_beta2 = 1.0 - self.beta2

        # ------- allgather_deredundency: batched path -------
        if self.comm_strategy == "allgather_deredundency":
            return self._construct_deredundency(
                gradients, weight_decay, lr,
                bias_correction1, bias_correction2, one_minus_beta2)

        # ------- allgather: original sequential path (unchanged) -------
        optim_result = []
        for i, (param, gradient, use_muon) in enumerate(zip(self._parameters, gradients, self.use_muon)):
            param_name = self.param_name_tuple[i]

            if "max_logits_val" in param_name:
                optim_result.append(P.Cast()(gradient, F.dtype(param)))
                continue

            if not self.optim_filter[i]:
                optim_result.append(gradient)
                continue

            if self.is_group:
                param_lr = lr[i] if self.is_group_lr else lr
                param_wd = weight_decay[i]
            else:
                param_lr = lr
                param_wd = weight_decay

            if use_muon:
                muon_m = self.muon_m[self.state_indices[i]]
                result = _apply_muon_update(
                    gradient, muon_m, self.muon_momentum,
                    self.use_nesterov, param, param_lr, param_wd,
                    self.matched_adamw_rms, self.muon_split_fn, self.muon_merge_fn,
                    param_name, self.eps,
                    self.ns_steps, self.ns_coefficients)
            else:
                state_idx = self.state_indices[i]
                with SkipDTensorDispatch():
                    _run_adamw_opt(
                        self.beta1, self.beta2, self.adamw_eps,
                        param_lr, param_wd, param, gradient,
                        self.moments1[state_idx], self.moments2[state_idx],
                        True, bias_correction1, bias_correction2, one_minus_beta2)
                    result = param

            optim_result.append(result)

        if self.qk_clip_enabled and self.model.synced_max_attention_logit_fires(self.logit_threshold):
            self.model.apply_qk_clip_scaling(
                self.logit_threshold,
                self.muon_split_fn,
                self.muon_merge_fn,
            )

        return optim_result

    def _construct_deredundency(
        self, gradients, weight_decay, lr,
        bias_correction1, bias_correction2, one_minus_beta2,
    ):
        """Batched construct for ``allgather_deredundency`` strategy.

        Collects all muon weights, runs them through :func:`_run_muon_batched`,
        and interleaves adam-weight processing.
        """
        # Refine init-time owner assignment once gradients (and thus per-weight
        # layouts) are available.  After the first call this is a no-op.
        self._recompute_muon_assigned_ranks(gradients)

        # --- collect muon weight info ---
        muon_grads = []
        muon_params = []
        muon_m_ms = []
        muon_lrs = []
        muon_wds = []
        muon_names = []
        muon_assigned = []
        muon_metas = []  # frozen per-param DTensor metadata from _muon_param_meta
        # maps: param_index -> muon slot index (for result placement)
        muon_result_slot = {}    # param_index -> slot in muon lists
        muon_slot_idx = 0
        adamw_tasks = []

        optim_result = [None] * len(gradients)
        param_meta = self._muon_param_meta or {}

        for i, (param, gradient, use_muon) in enumerate(zip(self._parameters, gradients, self.use_muon)):
            param_name = self.param_name_tuple[i]

            if "max_logits_val" in param_name:
                optim_result[i] = P.Cast()(gradient, F.dtype(param))
                continue

            if not self.optim_filter[i]:
                optim_result[i] = gradient
                continue

            if self.is_group:
                param_lr = lr[i] if self.is_group_lr else lr
                param_wd = weight_decay[i]
            else:
                param_lr = lr
                param_wd = weight_decay

            if use_muon:
                muon_grads.append(gradient)
                muon_params.append(param)
                muon_m_ms.append(self.muon_m[self.state_indices[i]])
                muon_lrs.append(param_lr)
                muon_wds.append(param_wd)
                muon_names.append(param_name)
                muon_assigned.append(self._muon_assigned_rank.get(i, self._rank_id))
                muon_metas.append(param_meta.get(i))
                muon_result_slot[i] = muon_slot_idx
                muon_slot_idx += 1
            else:
                state_idx = self.state_indices[i]
                adamw_tasks.append((i, param, gradient, param_lr, param_wd, state_idx))

        adamw_done = False

        def run_adamw_tasks():
            nonlocal adamw_done
            if adamw_done:
                return
            adamw_done = True
            for task_i, task_param, task_gradient, task_lr, task_wd, task_state_idx in adamw_tasks:
                with SkipDTensorDispatch():
                    _run_adamw_opt(
                        self.beta1, self.beta2, self.adamw_eps,
                        task_lr, task_wd, task_param, task_gradient,
                        self.moments1[task_state_idx], self.moments2[task_state_idx],
                        True, bias_correction1, bias_correction2, one_minus_beta2)
                optim_result[task_i] = task_param

        # --- run batched muon update ---
        if muon_grads:
            muon_results = _run_muon_batched(
                muon_grads, muon_params, muon_m_ms,
                muon_lrs, muon_wds,
                muon_names, muon_assigned,
                self.muon_momentum, self.use_nesterov,
                self.matched_adamw_rms,
                self.muon_split_fn, self.muon_merge_fn,
                self.eps, self.ns_steps, self.ns_coefficients,
                self._rank_id,
                overlap_callback=run_adamw_tasks,
                muon_metas=muon_metas,
            )
            # place muon results back into optim_result
            for param_idx, slot in muon_result_slot.items():
                optim_result[param_idx] = muon_results[slot]
        else:
            run_adamw_tasks()

        if self.qk_clip_enabled and self.model.synced_max_attention_logit_fires(self.logit_threshold):
            self.model.apply_qk_clip_scaling(
                self.logit_threshold,
                self.muon_split_fn,
                self.muon_merge_fn,
            )

        return optim_result
