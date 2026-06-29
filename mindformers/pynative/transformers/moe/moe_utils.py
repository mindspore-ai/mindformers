# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
MoE Auxiliary Loss computation and gradient injection.
"""

from typing import Optional, Tuple, Union, List

from mindspore import Tensor, mint, nn, ops
from mindspore.common import dtype as mstype
from mindspore.common._grad_function import _Function
from mindspore.mint.distributed import get_world_size, all_reduce

from hyper_parallel.core.dtensor.dtensor import DTensor

from mindformers.pynative.distributed.activation_checkpoint import is_in_recompute

# MOE logging
_MOE_LAYER_WISE_LOGGING_TRACKER: dict = {}

# Global communication domain variables for MoE aux loss.
# Set via ``set_moe_aux_loss_group_info`` from the model parallelize step.
_AUX_LOSS_GROUP = None
_AUX_LOSS_GROUP_SIZE = 1


def switch_load_balancing_loss_func(
        probs: Tensor,
        tokens_per_expert: Tensor,
        total_num_tokens: int,
        topk: int,
        num_experts: int,
        moe_aux_loss_coeff: float,
        fused: bool = False,
) -> Tensor:
    """Calculate the auxiliary loss for load balancing.
    Refer to the Switch Transformer (https://arxiv.org/abs/2101.03961)
    and Global Load Balancing Loss(https://arxiv.org/abs/2501.11873) for details.

    ### Detailed explanation of the auxiliary loss #######

    The formula for the auxiliary loss is:
        loss = E * Σ_{i=1}^{E} (f_i * P_i)
    where:
        f_i = 1 / (T * topk) * Σ_{x∈B} routing_map(x, i)
             (fraction of tokens dispatched to expert i)
        P_i = 1 / T * Σ_{x∈B} probs(x, i)
             (averaged router probability allocated for expert i)
        E is the number of experts
        T is the total number of tokens in the batch B

    For distributed training with sequence or context parallelism, each rank can
    process a subset of the batch.
        loss = E * Σ_{i=1}^{E} (f_i * Σ_{j=1}^{N} P_ij)
             = E * Σ_{i=1}^{E} Σ_{j=1}^{N} (f_i * P_ij)
             = Σ_{j=1}^{N} E * (Σ_{i=1}^{E} f_i * P_ij)

    where:
        f_i = 1 / (T * topk) * Σ_{x∈B} routing_map(x, i)
             (fraction of tokens dispatched to expert i in the global batch)
        P_ij = 1 / T * Σ_{x∈B_j} probs(x, i)
              (averaged router probability allocated for expert i in local batch of the j-th rank)
        N is the number of ranks
        B_j is the batch of tokens in the j-th rank
        T is the total number of tokens in the global batch B

    Note:
    To calculate the auxiliary loss at different levels (micro-batch or global batch):
    - probs: Should always be from the local batch being processed
    - tokens_per_expert: Should represent token counts at the desired level
      (either micro-batch or global batch)
    - total_num_tokens: Should match the total token count at the same level as tokens_per_expert

    #########################################################

    Args:
        probs (Tensor): Softmax probabilities output by the router for each token.
                              Shape in [num_tokens, num_experts].
        tokens_per_expert (Tensor): Number of tokens assigned to each expert in the batch.
                                          Shape in [num_experts]
        total_num_tokens (int): Total number of tokens in the batch.
        topk (int): The number of experts selected for each token.
        num_experts (int): The number of experts.
        moe_aux_loss_coeff (float): The coefficient for the auxiliary loss.
        fused (bool): Whether to use the fused version of the auxiliary loss.

    Returns:
        Tensor: The auxiliary loss for load balancing.
    """
    if fused:
        raise ValueError("Fused version is not supported yet.")
    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = mint.sum(aggregated_probs_per_expert * tokens_per_expert) * (
            num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )
    return aux_loss


def get_tokens_per_expert_and_token_count(
        routing_map: Tensor,
        reduce_group,
        topk: int = None,
        with_padding_mask: bool = False,
) -> Tuple[Tensor, int, int]:
    """Compute per-expert global token counts and local/total token counts for MoE aux loss.

    This is the core statistics entry point for the Mixture-of-Experts load-balancing
    auxiliary loss. It performs two main tasks:

    1. Sums ``routing_map`` along the expert dimension to obtain the local per-expert
       token histogram (``local_tokens_per_expert``), then all-reduces (SUM) over
       ``reduce_group`` to aggregate the global per-expert token count
       (``global_tokens_per_expert``) across the parallel domain.
    2. Derives the local token count (``local_num_tokens``) from the row count of
       ``routing_map``, and the global token total (``total_num_tokens``) by scaling
       it with the world size of ``reduce_group``.

    ``reduce_group`` spans only the TP x CP domain: routing happens BEFORE the MoE
    dispatcher, so only TP/CP shard the router's token view, and summing over the
    group recovers the per-(batch, expert) histogram for the full batch. DP is
    intentionally excluded -- each DP rank contributes its own per-sequence aux loss,
    averaged by the caller via ``/bsz``; all-reducing over DP would double-count.

    Args:
        routing_map (Tensor): Local token-expert assignment map with shape
            ``[slen, bsz*E]`` (after the per-sequence reshape) or ``[T, E]``,
            matching the format expected by the calling aux-loss variant.
        reduce_group: Either a single process group covering the tp×cp domain,
            OR a list of 1D process groups (one per axis) covering the same
            domain. The list form is used when the mesh is 2D and only per-axis
            groups are available. ``None`` or empty means "no reduction needed;
            the local histogram is already the global view".
        topk (int): Number of experts selected per token. Required when
            ``with_padding_mask=True``; the ``seq_aux_loss`` variant passes
            ``topk * bsz`` to account for the per-batch flatten.
        with_padding_mask (bool): Whether the routing_map is padded. Currently
            unsupported in this codebase.

    Returns:
        Tuple of ``(global_tokens_per_expert, local_num_tokens, total_num_tokens)``:
        - ``global_tokens_per_expert``: per-expert token count after all-reduce over
          ``reduce_group``.
        - ``local_num_tokens``: row count of ``routing_map`` (local token count).
        - ``total_num_tokens``: ``local_num_tokens * reduce_group.size()`` (or the
          histogram-derived equivalent when padding is masked).
    """
    _ = topk
    if with_padding_mask:
        raise NotImplementedError(
            "Padding-mask aware token count is not implemented in this "
            "codebase; ``get_tokens_per_expert_and_token_count`` falls back "
            "to the row-count derivation used in the non-padded path."
        )
    local_tokens_per_expert = routing_map.sum(dim=0)
    global_tokens_per_expert = local_tokens_per_expert
    # Normalize ``reduce_group`` to a list of 1D groups so the nested-reduce
    # logic below is uniform regardless of whether the caller passed a single
    # group (legacy path) or a list of axis groups (tp×cp decomposition).
    if reduce_group is None:
        reduce_groups = []
    elif isinstance(reduce_group, (list, tuple)):
        reduce_groups = list(reduce_group)
    else:
        reduce_groups = [reduce_group]
    if reduce_groups:
        for sub_group in reduce_groups:
            all_reduce(global_tokens_per_expert, op=ops.ReduceOp.SUM, group=sub_group)
    local_num_tokens = routing_map.shape[0]
    total_num_tokens = local_num_tokens * (
        get_world_size_from_group(reduce_group)
    )
    return global_tokens_per_expert, local_num_tokens, total_num_tokens


def get_world_size_from_group(group) -> int:
    """Best-effort ``world_size`` lookup for a process group.

    Falls back to ``1`` if the size cannot be resolved (e.g. ``group`` is
    ``None`` or the build lacks a world-size API). Accepts a list/tuple of
    groups (the tp×cp decomposition produced by
    ``set_moe_aux_loss_group_info``) and returns the product of the
    per-axis sizes, i.e. ``cp_size * tp_size``.
    """
    # pylint: disable=broad-exception-caught
    if group is None:
        return 1
    if isinstance(group, (list, tuple)):
        # Early-exit on a degenerate entry to avoid inflating the product
        # with a stray 0 from a group whose size could not be resolved.
        sub_sizes = [get_world_size_from_group(g) for g in group]
        if any(s <= 0 for s in sub_sizes):
            return 1
        size = 1
        for s in sub_sizes:
            size *= s
        return size
    try:
        return group.size()
    except Exception:
        pass
    try:
        return get_world_size(group)
    except Exception:
        return 1


def set_moe_aux_loss_group_info(groups, group_size):
    """Set global communication domain variables for MoE aux loss.

    Called once during model parallelization to establish the tp×cp reduction
    group(s) used by both the router-side histogram reduction
    (``get_tokens_per_expert_and_token_count``) and the step-end logging
    aggregation (``track_moe_metrics``).

    The hyper-parallel ``DeviceMesh`` does not expose a single process group
    covering an entire multi-axis mesh, and the per-axis ``get_group()`` API
    raises when the mesh is 2D. We therefore store the list of 1D axis groups
    (e.g. ``[cp_group, tp_group]``) and reduce over them sequentially -- SUM
    is associative, so the result is identical to a single tp×cp all-reduce.

    Args:
        groups: List of 1D process groups whose Cartesian product spans the
            tp×cp domain. ``None`` or an empty list means "no reduction
            needed" (e.g. tp=cp=1). A single-element list is also accepted
            (e.g. only tp is enabled).
        group_size: World size of the full tp×cp domain
            (``cp_size * tp_size``). Used by the router to scale the aux
            loss gradient so it is independent of the mesh shape.
    """
    # pylint: disable=W0603
    global _AUX_LOSS_GROUP, _AUX_LOSS_GROUP_SIZE
    if groups is None:
        groups = []
    # Drop None entries (size-1 axes) and keep at most one entry per axis.
    _AUX_LOSS_GROUP = [g for g in groups if g is not None]
    _AUX_LOSS_GROUP_SIZE = max(group_size, 1)


def get_moe_aux_loss_group():
    """Return the list of tp×cp process groups for aux-loss histogram reduction.

    Returns a list (possibly empty) of 1D groups. Callers that perform a
    single all-reduce should use ``reduce_over_aux_loss_groups`` instead,
    which iterates the list and applies each group sequentially.
    """
    return _AUX_LOSS_GROUP


def get_moe_aux_loss_group_size():
    """Return the world size of the tp×cp aux-loss group (1 if unset)."""
    return _AUX_LOSS_GROUP_SIZE


def reduce_over_aux_loss_groups(tensor, op=ops.ReduceOp.SUM):
    """Apply SUM/other reduction sequentially over every tp×cp axis group.

    Equivalent to a single all-reduce over the full tp×cp mesh, but works
    with hyper-parallel's per-axis 1D process groups. SUM is associative so
    the order of axes does not affect the result. No-op when there are no
    enabled groups (e.g. tp=cp=1).
    """
    for group in _AUX_LOSS_GROUP:
        all_reduce(tensor, op=op, group=group)


class _MoEAuxLossAutoScaler(_Function):
    """An AutoScaler that triggers the backward pass and scales the grad for auxiliary loss."""

    main_loss_backward_scale: Optional[Tensor] = None

    @staticmethod
    def forward(ctx, output: Tensor, aux_loss: Tensor) -> Tensor:
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.

        Args:
            output (Tensor): The output tensor.
            aux_loss (Tensor): The auxiliary loss tensor.

        Returns:
            Tensor: The output tensor.
        """
        ctx.aux_loss = aux_loss
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (Tensor): The gradient of the output.

        Returns:
            Tuple[Tensor, Tensor]: The gradient of the output, scaled auxiliary loss gradient.
        """
        aux_loss = ctx.aux_loss
        if _MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            # Prefer mint operator to create Tensor
            _MoEAuxLossAutoScaler.main_loss_backward_scale = Tensor(
                1.0, dtype=aux_loss.dtype
            )

        aux_loss_backward_scale = _MoEAuxLossAutoScaler.main_loss_backward_scale
        aux_loss = aux_loss.to_local() if isinstance(aux_loss, DTensor) else aux_loss
        scaled_aux_loss_grad = mint.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: Tensor) -> None:
        """set the scale of the aux loss.

        Args:
            scale (Tensor): The scale value to set. Please ensure that the scale passed in
                            matches the scale of the main_loss.
        """
        _MoEAuxLossAutoScaler.main_loss_backward_scale = scale


class MoEAuxLossAutoScaler(nn.Cell):
    """
    Module wrapper for the custom LogSoftmax function.
    """

    @staticmethod
    def construct(output: Tensor, aux_loss: Tensor):
        """
        Forward pass for LogSoftmax.
        """
        return _MoEAuxLossAutoScaler.apply(output, aux_loss)


def compute_routing_scores_for_aux_loss(
        logits: Tensor,
        topk: int,
        score_function: str,
) -> Tuple[Tensor, Tensor]:
    """Compute routing scores based on the score function.

    Args:
        logits (Tensor): The logits tensor after gating, shape: [num_tokens, num_experts].
        topk (int): The number of top-k indices to compute.
        score_function (str): The score function to use. Can be either "softmax" or "sigmoid".
        fused (bool, optional): Whether to use the fused version. Defaults to False.
        padding_mask (Tensor, optional): Boolean mask indicating non-padding tokens.
                                         Shape in [num_tokens]. True for valid tokens,
                                         False for padding tokens. Defaults to None.

    Returns:
        Tuple[Tensor, Tensor]: The routing map and the normalized routing scores.
    """
    if score_function == "softmax":
        scores = mint.softmax(logits, dim=-1, dtype=mstype.float32)
    elif score_function == "sigmoid":
        scores = mint.sigmoid(logits)
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    _, top_indices = mint.topk(scores, k=topk, dim=1)
    routing_map = mint.zeros_like(logits).int()
    routing_map.scatter_(1, top_indices, 1).bool()
    return routing_map, scores


def get_moe_layer_wise_logging_tracker() -> dict:
    """Return the moe layer wise tracker."""
    # pylint: disable=W0602
    global _MOE_LAYER_WISE_LOGGING_TRACKER
    return _MOE_LAYER_WISE_LOGGING_TRACKER


def save_to_aux_losses_tracker(
        loss: Tensor,
        layer_number: int,
        num_layers: int,
) -> None:
    """Save the auxiliary loss for logging.
    Args:
        name (str): The name of the loss.
        loss (Tensor): The loss tensor.
        layer_number (int): Layer index of the loss.
        num_layers (int): The number of total layers.
    """
    # Skip aux loss logging if layer_number is None.
    if layer_number is None:
        return

    # Skip during activation recompute: the recomputed forward would otherwise add
    # this layer's aux loss to the tracker a second time, doubling load_balancing_loss
    # for recomputed layers.
    if is_in_recompute():
        return

    tracker = get_moe_layer_wise_logging_tracker()
    if not tracker:
        tracker["values"] = mint.zeros(num_layers)
    if isinstance(loss, DTensor):
        loss = loss.to_local()
    if hasattr(loss, "detach"):
        cur_val = tracker["values"][layer_number] + loss.detach()
        tracker["values"][layer_number] = cur_val  # Aggregate the loss for the layer.
    else:
        cur_val = tracker["values"][layer_number] + loss
        tracker["values"][layer_number] = cur_val

def clear_aux_losses_tracker() -> None:
    """Clear the auxiliary losses."""
    tracker = get_moe_layer_wise_logging_tracker()
    for value in tracker["values"]:
        value.zero_()


def track_moe_metrics(
        loss_scale: float,
        num_layers: Optional[int] = None,
        moe_layer_freq: Optional[Union[int, List[int]]] = None,
        mtp_num_layers: Optional[int] = None,
        group=None,
        group_size=None,
        pp_group=None,
        pp_group_size=None,
        has_last: bool = True,
):
    """Track and compute average MoE auxiliary loss across all MoE layers.

    Pipeline-parallel combine
    -------------------------
    The aux-loss metric is tracked per GLOBAL layer index, but each PP stage
    only fills the slots for the layers it owns (forward runs that stage's
    routers). Summing the tracker and dividing by the GLOBAL MoE-layer count
    (below) would make the last stage under-report ``load_balancing_loss`` by
    ``local_moe_layers / global_moe_layers``. So when ``pp_group_size > 1`` we
    first all-reduce the per-layer tracker over the PP group to give every
    stage the full per-layer vector.

    This all-reduce is a PP-group collective: **every** stage must reach it
    every step, so this function must be called on every stage every step
    (the ``loss_scale`` guard is config-driven and identical across ranks, so
    all stages skip or run it together — deadlock-free). Non-last stages then
    clear their tracker (they never run the reporting path below, which is what
    normally clears it) and return ``None`` to avoid double-counting on the
    next step. Only the last stage keeps the combined values to report.
    """
    if not loss_scale or loss_scale <= 0.0:
        return None

    tracker = get_moe_layer_wise_logging_tracker()

    # Step 1: CP all-reduce (SUM) — combine context-parallel shards.
    # Router inputs are replicated across TP and independent across DP;
    # only CP shards the token view, so only CP ranks (if any) are combined here.
    if _AUX_LOSS_GROUP and _AUX_LOSS_GROUP_SIZE > 1:
        if "values" not in tracker:
            num_total_layers = num_layers
            if mtp_num_layers:
                num_total_layers += mtp_num_layers
            tracker["values"] = mint.zeros(num_total_layers)
        for sub_group in _AUX_LOSS_GROUP:
            all_reduce(tracker["values"], op=ops.ReduceOp.SUM, group=sub_group)
    # Step 2: Combine the per-layer tracker across PP stages.
    if pp_group_size and pp_group_size > 1:
        if "values" not in tracker:
            num_pp_layers = num_layers
            if mtp_num_layers:
                num_pp_layers += mtp_num_layers
            tracker["values"] = mint.zeros(num_pp_layers)
        all_reduce(tracker["values"], op=ops.ReduceOp.SUM, group=pp_group)
        if not has_last:
            clear_aux_losses_tracker()
            return None

    # No MoE layer contributed aux losses.
    if "values" not in tracker:
        return None

    # Get number of MoE layers
    if moe_layer_freq is None:
        num_moe_layers = num_layers
    elif isinstance(moe_layer_freq, int):
        assert isinstance(num_layers, int)
        moe_layer_pattern = [1 if (i % moe_layer_freq == 0) else 0 for i in range(num_layers)]
        num_moe_layers = sum(moe_layer_pattern)
    elif isinstance(moe_layer_freq, list):
        num_moe_layers = sum(moe_layer_freq)
    else:
        raise ValueError(f"Invalid moe_layer_freq: {moe_layer_freq}")

    if mtp_num_layers is not None:
        num_moe_layers += mtp_num_layers

    aux_losses = tracker["values"].sum() / num_moe_layers

    if group_size is None:
        group_size = get_world_size()
    if group_size > 1:
        all_reduce(aux_losses, op=ops.ReduceOp.SUM, group=group)
        aux_losses /= group_size
    clear_aux_losses_tracker()

    return aux_losses
