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
"""Expert Parallel with comm/compute overlap — A/B/C/D sync hook protocol.

This module provides :class:`OverlapExpertParallel`, which extends the
synchronous :class:`ExpertParallel` strategy with four differentiable
synchronization hooks (A, B, C, D) that bracket the MoE dispatch and
combine all-to-all kernels.  When paired with a
:class:`~hyper_parallel.core.pipeline_parallel.comm_compute_overlap.CommComputeOverlap`
orchestrator that runs forward and backward on two threads concurrently,
the EP a2a communication on one thread overlaps with compute on the other.

Key design points
-----------------
Single HCCL stream funnel
    Every EP all-to-all on a group (counts, main token, combine — the
    routing-map a2a is gone, and the post-dispatch resort is local) routes
    through ``comm_func.all_to_all_single`` / the hyper_parallel
    platform's ``all_to_all_single`` path.  Using ``ops.AlltoAll`` /
    ``ops.AlltoAllV`` Primitives would dispatch on a *different* stream from
    ``comm_func.all_to_all_single``, so mixing them under dual-thread overlap
    lets two threads enqueue HCCL ops on two streams against the same group;
    cross-rank the ordering is non-deterministic and the next collective on
    the group deadlocks once ``MS_DEV_LAUNCH_BLOCKING`` is unset.

AsyncCollectiveTensor for main and combine a2a
    The main token dispatch a2a and the combine a2a are issued with
    ``platform.differentiable_all_to_all_single_async``, which returns an
    :class:`~hyper_parallel.platform.mindspore.platform.AsyncCollectiveTensor`
    whose ``CommHandle.wait()`` fires lazily at the first consumer op via
    ``__ms_dispatch__``.  This defers the host wait into the compute window,
    creating the actual overlap between a2a kernel and peer compute.

Local pad_size
    The vanilla :class:`ExpertParallel` stores ``self.pad_topk_indices`` on the
    instance.  Under dual-thread overlap that shared mutable state races between
    forward and backward and can trip MS PyNative's lazy shape inference.
    ``OverlapExpertParallel`` instead stores ``pad_size`` (a plain Python ``int``)
    in :attr:`ctx` alongside the other state needed by ``_token_combine``.

Recompute compatibility
    Non-reentrant recompute is prefired by the OVERLAP_B_F callback before
    entering the two-thread window. HyperParallel reentrant checkpoints replay
    lazily inside the BWD worker. Their final ``D_LAST`` replay hook is treated
    as a regular ``D`` because replay has no callback-level ``CHUNK_END`` to
    notify the last combine event.
"""

import mindspore as ms
from mindspore import mint, nn

from hyper_parallel import DeviceMesh
from hyper_parallel.platform import get_platform
from hyper_parallel.core.pipeline_parallel.hook_coordinator import HookCoordinator

from mindformers.pynative.distributed.expert_parallel import ExpertParallel
from mindformers.pynative.distributed.activation_checkpoint import is_in_recompute
from mindformers.pynative.distributed.style import register_comm_op, _call_comm_op

_platform = get_platform()


class OverlapExpertParallel(ExpertParallel):
    """Expert Parallel strategy with async A2A and A/B/C/D sync hooks.

    Extends :class:`ExpertParallel` for use with
    :class:`~hyper_parallel.core.pipeline_parallel.comm_compute_overlap.CommComputeOverlap`.
    The four hooks bracket the MoE dispatch and combine a2a phases so the
    HCCL kernels on one thread overlap with compute on the paired thread.

    Args:
        coordinator:    :class:`HookCoordinator` from the shared
                        :class:`CommComputeOverlap` orchestrator.
        is_last_layer:  When ``True``, the closing D hook is tagged
                        ``"D_LAST"`` so the rendezvous is skipped in
                        forward (no Attention follows the last MoE layer)
                        and the backward fires the out-of-band pair-8
                        rendezvous instead.  Tag the **last MoE layer in
                        each pipeline chunk** with this flag.
        moe_permute_fusion: Passed to :class:`ExpertParallel`.
        use_safe_tokens: Whether to prepend safe tokens before dispatch.

    Note:
        Pass every MoE layer's experts through this strategy and tag the
        last one per chunk with ``is_last_layer=True``.  Also add
        ``CHUNK_START`` / ``CHUNK_END`` hooks to the chunk model via
        :func:`apply_chunk_overlap_hooks`.
    """

    def __init__(
        self,
        coordinator: HookCoordinator,
        is_last_layer: bool = False,
        moe_permute_fusion: bool = False,
        async_d2h: bool = False,
        use_safe_tokens: bool = True,
    ) -> None:
        super().__init__(
            moe_permute_fusion=moe_permute_fusion,
            async_d2h=async_d2h,
            use_safe_tokens=use_safe_tokens,
        )
        self._coordinator = coordinator
        self._d_hook = "D_LAST" if is_last_layer else "D"
        # async_d2h (ParallelismConfig.expert_parallel_async_d2h) composes with the
        # dual-thread overlap: the staged dispatch keeps the A/B hooks at the a2a-segment
        # boundary (see _dispatch_a2a), so the side-stream counts D2H fits underneath.

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        module = super()._apply(module, device_mesh)
        # Replace base-class sync registrations with overlap async versions.
        module._comm_ops.clear()

        style_self = self

        def _counts_fn(output, input_tensor, group):
            result = style_self.all_to_all_single(output, input_tensor, group=group)
            return result if isinstance(result, ms.Tensor) else result[0]
        register_comm_op(module, "expert_counts.alltoallsingle", _counts_fn, "ep")

        def _async_fn(flat_input, send_splits, recv_splits, block_size):
            return style_self._async_a2a(flat_input, send_splits, recv_splits, block_size)
        register_comm_op(module, "input.alltoallsingle", _async_fn, "ep")
        register_comm_op(module, "output.alltoallsingle", _async_fn, "ep")

        return module

    # ------------------------------------------------------------------
    # Overlap seams — the only logic added on top of the synchronous base.
    # The inherited _token_dispatch / _token_combine orchestrators call the
    # a2a primitives and _dispatch_comm / _combine_comm below; here we add the
    # A/B/C/D sync hooks and swap the collectives for async / single-stream.
    # ------------------------------------------------------------------

    def _sync_hook(self, x, hook_name: str):
        """Fire a differentiable A/B/C/D sync hook on ``x`` (identity in the base)."""
        # D_LAST.forward is a pure skip for the paired main-thread forward:
        # CHUNK_END later notifies the C_last COMM event and supplies its next
        # rendezvous. A reentrant layer replay has no callback-level CHUNK_END,
        # so applying that skip there leaves the paired thread waiting forever.
        # Treat replay's D_LAST as a regular D; its notify+rendezvous closes the
        # replay-forward C event before the retained replay graph runs backward.
        if hook_name == "D_LAST" and is_in_recompute():
            hook_name = "D"
        return _platform.differentiable_sync_hook(x, hook_name, self._coordinator)

    # ---- sync a2a primitives: plain comm.all_to_all_single (self.all_to_all_single),
    # the same collective the base uses, kept on the EP group stream so it funnels
    # with the async token/combine a2a. Overridden vs the base only to use the
    # _single_ variant (the base counts path uses the list-based comm.all_to_all,
    # which would dispatch on a different stream). ----

    def _counts_a2a(self, num_tokens_per_expert, ep_degree, cell=None):  # pylint: disable=unused-argument
        if (cell is not None and hasattr(cell, '_comm_ops')
                and "expert_counts.alltoallsingle" in cell._comm_ops):
            counts_size = int(num_tokens_per_expert.shape[0])
            output = mint.empty((counts_size,), dtype=num_tokens_per_expert.dtype)
            result = _call_comm_op(cell, "expert_counts.alltoallsingle",
                                    output, num_tokens_per_expert, group=self.ep_group)
            return result if isinstance(result, ms.Tensor) else result[0]
        counts_size = int(num_tokens_per_expert.shape[0])
        output = mint.empty((counts_size,), dtype=num_tokens_per_expert.dtype)
        result = self.all_to_all_single(output, num_tokens_per_expert, group=self.ep_group)
        return result if isinstance(result, ms.Tensor) else result[0]

    def _main_a2a(self, flat_in, input_splits, output_splits, block_size, cell=None):
        if (cell is not None and hasattr(cell, '_comm_ops')
                and "input.alltoallsingle" in cell._comm_ops):
            result = _call_comm_op(
                cell,
                "input.alltoallsingle",
                flat_in,
                input_splits,
                output_splits,
                block_size,
            )
            return result[0] if isinstance(result, tuple) else result
        return self._async_a2a(flat_in, input_splits, output_splits, block_size)

    def _combine_a2a(self, flat_in, send_splits, recv_splits, block_size, cell=None):
        if (cell is not None and hasattr(cell, '_comm_ops')
                and "output.alltoallsingle" in cell._comm_ops):
            result = _call_comm_op(
                cell,
                "output.alltoallsingle",
                flat_in,
                send_splits,
                recv_splits,
                block_size,
            )
            return result[0] if isinstance(result, tuple) else result
        return self._async_a2a(flat_in, send_splits, recv_splits, block_size)

    # ---- communication segments: wrap the a2a with the A/B/C/D sync hooks ----

    def _dispatch_comm(
            self,
            flat_in,
            num_tokens_per_expert,
            ep_degree,
            block_size,
            cell=None,
            shared_expert_ctx=None,
    ):
        """Overlap dispatch comm: A hook -> counts -> splits -> chunk counts
        (small host list) -> async main a2a -> B hook.

        Reading the chunk counts consumes only the already-complete counts a2a, so
        it does not force the async main a2a's lazy wait. The local fused permute
        (or split/cat fallback) after B consumes the async output, preserving the
        overlap window without a routing-map collective.
        """
        flat_in = self._sync_hook(flat_in, "A")
        num_tokens_per_expert_group = self._counts_a2a(
            num_tokens_per_expert, ep_degree, cell=cell)
        # Single batched D2H (consumes only the already-complete counts a2a, not the
        # async main a2a below — the overlap window is preserved); group_list on device.
        input_splits, output_splits, group_counts = self._host_token_splits(
            num_tokens_per_expert, num_tokens_per_expert_group, ep_degree)
        num_tokens_per_expert = self._compute_group_list(num_tokens_per_expert_group, ep_degree)
        flat_out = self._dispatch_a2a_with_shared_expert(
            flat_in,
            input_splits,
            output_splits,
            block_size,
            cell=cell,
            shared_expert_ctx=shared_expert_ctx,
        )
        # The routing map depends only on the completed counts a2a. Build it
        # after launching the async token a2a so repeat_interleave overlaps HCCL.
        resort_routing_map = self._build_resort_routing_map(
            num_tokens_per_expert_group, group_counts, ep_degree)
        flat_out = self._sync_hook(flat_out, "B")
        return (
            flat_out, group_counts, input_splits, output_splits,
            num_tokens_per_expert, resort_routing_map
        )

    def _dispatch_a2a(self, flat_in, host_buf, event, num_tokens_per_expert_group,
                      num_experts, ep_degree, block_size, cell=None, shared_expert_ctx=None):
        """Async-D2H stage 2 with A/B hooks (used when expert_parallel_async_d2h is on).

        The counts a2a + the async counts D2H were already issued in
        :meth:`ExpertParallel._dispatch_preprocess` (before the permute); here we
        only bracket the *main token a2a* with the A/B rendezvous. The deferred
        ``event.synchronize()`` waits solely on this rank's side-stream counts copy
        (a tiny, by-now-complete transfer) — it is NOT a cross-thread rendezvous, so
        it does not couple with the coordinator's A/B/C/D barriers. During
        recompute, the forward's cached host splits bypass both the D2H and wait.
        """
        flat_in = self._sync_hook(flat_in, "A")
        input_splits, output_splits, group_counts = self._finish_async_d2h(
            host_buf, event, num_experts, ep_degree)
        num_tokens_per_expert = self._compute_group_list(num_tokens_per_expert_group, ep_degree)
        flat_out = self._dispatch_a2a_with_shared_expert(
            flat_in,
            input_splits,
            output_splits,
            block_size,
            cell=cell,
            shared_expert_ctx=shared_expert_ctx,
        )
        # Keep the same overlap window as the synchronous-D2H dispatch path.
        resort_routing_map = self._build_resort_routing_map(
            num_tokens_per_expert_group, group_counts, ep_degree)
        flat_out = self._sync_hook(flat_out, "B")
        return (
            flat_out, group_counts, input_splits, output_splits,
            num_tokens_per_expert, resort_routing_map
        )

    def _combine_comm(
            self,
            flat_in,
            input_splits,
            output_splits,
            block_size,
            cell=None,
            shared_expert_ctx=None,
    ):
        """Overlap combine comm: C hook -> async combine a2a -> D / D_LAST hook."""
        flat_in = self._sync_hook(flat_in, "C")
        flat_out = self._combine_a2a_with_shared_expert(
            flat_in,
            output_splits,
            input_splits,
            block_size,
            cell=cell,
            shared_expert_ctx=shared_expert_ctx,
        )
        return self._sync_hook(flat_out, self._d_hook)


# ---------------------------------------------------------------------------
# Chunk-level CHUNK_START / CHUNK_END hooks
# ---------------------------------------------------------------------------

def apply_chunk_overlap_hooks(model_part, coordinator: HookCoordinator):
    """Register CHUNK_START and CHUNK_END differentiable sync hooks on a pipeline stage model.

    The hooks fire on the primary ``hidden_states`` tensor (first positional
    argument on entry, first element of the output tuple on exit) so that:

    * ``CHUNK_START`` on entry pairs with ``D_LAST.bwd`` on the backward
      thread, ensuring combine.bwd of the last MoE layer in the chunk runs
      inside a barrier-synchronised window.
    * ``CHUNK_END`` on exit pairs with the explicit
      ``coordinator.rendezvous(HookRole.COMPUTE)`` called by the
      ``OVERLAP_B_F`` callback after ``backward_one_chunk``, ensuring neither
      thread exits the chunk before the other finishes its tail-end local work.

    These hooks are differentiable: ``_MSSyncHookFunction.apply`` records
    itself in the PyNative autograd graph and fires its backward (with the
    reversed hook roles) during ``backward_one_chunk``.

    Args:
        model_part: The pipeline stage model (``nn.Cell``).
        coordinator: The :class:`HookCoordinator` driving the rendezvous.

    Returns:
        ``model_part`` with hooks registered (mutated in place).
    """

    def _chunk_start_pre_hook(cell, args):  # pylint: disable=W0613
        if not args:
            # hidden_states arrived via kwargs (unexpected for the decoder, whose
            # first positional arg is decoder_input); nothing to bracket.
            return args
        x = args[0]
        x = _platform.differentiable_sync_hook(x, "CHUNK_START", coordinator)
        return (x,) + args[1:]

    def _chunk_end_hook(cell, args, output):  # pylint: disable=W0613
        if isinstance(output, tuple):
            x = output[0]
            x = _platform.differentiable_sync_hook(x, "CHUNK_END", coordinator)
            return (x,) + output[1:]
        return _platform.differentiable_sync_hook(output, "CHUNK_END", coordinator)

    model_part.register_forward_pre_hook(_chunk_start_pre_hook)
    model_part.register_forward_hook(_chunk_end_hook)
    return model_part
