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
    return muon_merge_fn(param_name, x_list)


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
#    Phase 1 – AllGather      : gather full tensors (collective, all ranks)
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

def _prepare_muon_input(gradient, muon_m, momentum, use_nesterov):
    """Phase 0: compute the momentum-updated gradient (local, no comm).

    Returns (ns_inputs_local, next_m_fp32, needs_redist, dev_mesh, placements, layout).
    """
    op_cast = P.Cast()
    ndim = len(gradient.shape)
    if ndim not in (2, 3):
        raise ValueError(
            f"newton_schulz only supports 2D or 3D gradient, got shape={gradient.shape}")

    needs_redist = isinstance(gradient, DTensor) and ndim == 2
    dev_mesh = gradient.device_mesh if needs_redist else None
    plmts = gradient.placements if needs_redist else None
    layout = gradient.layout if needs_redist else None

    gradient_local = _to_local(gradient)
    muon_m_local = _to_local(muon_m)

    m_fp32 = op_cast(muon_m_local, mstype.float32)
    grad_fp32 = op_cast(gradient_local, mstype.float32)
    next_m = m_fp32 * momentum + grad_fp32

    if use_nesterov:
        ns_input = grad_fp32 + next_m * momentum
    else:
        ns_input = next_m

    ns_inputs_local = op_cast(ns_input, mstype.bfloat16)
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
):
    """Batched Muon update with parallel Newton-Schulz across ranks.

    Only called when ``comm_strategy == "allgather_deredundency"``.
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
        return op_cast(next_param, F.dtype(param))

    def _try_start_local_shard_scatter(info, x_ret_full):
        """Use P2P sends so each rank receives only its local shard."""
        layout = info['layout']
        rank_list = tuple(int(rank) for rank in layout.rank_list)
        assigned_rank = int(info['assigned_rank'])
        # P2P send/recv must be entered consistently. For layouts that do not
        # cover the whole current worker group, use the existing broadcast path.
        if len(rank_list) != get_group_size():
            return False
        if assigned_rank not in rank_list or int(rank_id) not in rank_list:
            return False

        if len(rank_list) == 1:
            info['x_ret'] = x_ret_full.contiguous()
            info['x_ret_is_local'] = True
            return True

        if int(rank_id) == assigned_rank:
            p2p_handles = []
            p2p_tensors = []
            local_output = None
            for dst_rank, shard in zip(rank_list, _list_full_tensor_local_shards(x_ret_full, layout)):
                shard = shard.contiguous()
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
    # Per-weight: momentum update, cast to bf16, record mesh metadata.
    prepared = []
    for i in range(n_weights):
        ns_inputs_local, next_m, needs_redist, dev_mesh, plmts, layout = _prepare_muon_input(
            muon_gradients[i], muon_m_ms[i], muon_momentum, use_nesterov)
        ndim = len(muon_gradients[i].shape)
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
            'matmul_op': matmul_op,
            'addmm_op': addmm_op,
            'param': muon_params[i],
            'muon_m': muon_m_ms[i],
            'lr': muon_lrs[i],
            'wd': muon_wds[i],
            'param_name': muon_param_names[i],
            'assigned_rank': muon_assigned_ranks[i],
        })

    # ------------------------------------------------------------------
    # Phase 1 — AllGather all inputs (collective, one weight at a time)
    # ------------------------------------------------------------------
    # Simple all_concat layouts are issued asynchronously.  Unsupported
    # layouts fall back to DTensor.full_tensor(), preserving the previous
    # behavior.
    for info in prepared:
        if info['needs_redist']:
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
        else:
            info['ns_inputs_full'] = info['ns_inputs_local']

    # Local weights, including every 3D Muon weight, do not depend on the
    # 2D allgather result.  Run them while async allgathers are in flight.
    for info in prepared:
        if not info['needs_redist']:
            info['x_ret'] = _apply_muon_ns(
                info['ns_inputs_full'],
                muon_split_fn, muon_merge_fn, info['param_name'],
                eps, ns_steps, ns_coefficients,
                info['matmul_op'], info['addmm_op'],
                info['lr'], matched_adamw_rms)

    if overlap_callback is not None:
        overlap_callback()

    # ------------------------------------------------------------------
    # Phase 2/3 — Pipeline 2D allgather wait, Newton-Schulz, and scatter
    # ------------------------------------------------------------------
    # Only 2D DTensors are de-redundant: one assigned rank computes the full
    # NS output, then scatters local shards.  Local/3D weights already have
    # x_ret from the overlap block above and never enter this communication.
    #
    # Async allgathers were all issued in Phase 1.  Do not wait for every full
    # tensor before starting NS: materialize each weight only when it reaches
    # the compute slot, then immediately launch its scatter so scatter(i) can
    # overlap with NS(i+1).  If the owner rank is not in the current DTensor
    # layout group, fall back to the previous world broadcast path.
    for info in prepared:
        if not info['needs_redist']:
            continue
        ns_inputs_full = _materialize_full_tensor(info)
        x_ret_full = None
        if rank_id == info['assigned_rank']:
            x_ret_full = _apply_muon_ns(
                ns_inputs_full,
                muon_split_fn, muon_merge_fn, info['param_name'],
                eps, ns_steps, ns_coefficients,
                info['matmul_op'], info['addmm_op'],
                info['lr'], matched_adamw_rms).contiguous()
        if _try_start_local_shard_scatter(info, x_ret_full):
            continue

        if rank_id == info['assigned_rank']:
            info['x_ret'] = x_ret_full
        else:
            info['x_ret'] = mint.empty_like(ns_inputs_full)
        info['x_ret'] = info['x_ret'].contiguous()
        info['broadcast_handle'] = broadcast(
            info['x_ret'], src=info['assigned_rank'], async_op=True)

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
            info.pop('p2p_tensors', None)
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
                f"assigned across {world_size} ranks; {local_muon_count} 3D/local muon weights "
                f"run independently on every rank "
                f"(rank_id={self._rank_id}); "
                f"estimated_loads=[{load_summary}]"
            )
        else:
            self._muon_assigned_rank = None

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

        if self.qk_clip_enabled and self.model.has_qk_clip_candidates(self.logit_threshold):
            self.model.allreduce_max_attention_logit()
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
        # --- collect muon weight info ---
        muon_grads = []
        muon_params = []
        muon_m_ms = []
        muon_lrs = []
        muon_wds = []
        muon_names = []
        muon_assigned = []
        # maps: param_index -> muon slot index (for result placement)
        muon_result_slot = {}    # param_index -> slot in muon lists
        muon_slot_idx = 0
        adamw_tasks = []

        optim_result = [None] * len(gradients)

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
            )
            # place muon results back into optim_result
            for param_idx, slot in muon_result_slot.items():
                optim_result[param_idx] = muon_results[slot]
        else:
            run_adamw_tasks()

        if self.qk_clip_enabled and self.model.has_qk_clip_candidates(self.logit_threshold):
            self.model.allreduce_max_attention_logit()
            self.model.apply_qk_clip_scaling(
                self.logit_threshold,
                self.muon_split_fn,
                self.muon_merge_fn,
            )

        return optim_result
