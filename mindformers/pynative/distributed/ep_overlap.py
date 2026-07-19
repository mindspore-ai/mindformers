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
    routing-map a2a is gone, the post-dispatch resort is a fixed chunk
    permutation applied on device) routes through ``comm_func.all_to_all_single`` / the hyper_parallel
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
    Activation recompute composes with overlap through the OVERLAP_B_F
    callback (:func:`_make_overlap_b_f_callback`), which calls
    ``bwd_stage.recompute_one_chunk(bwd_mi)`` on the main thread BEFORE
    ``overlap.run`` enables the coordinator and spawns the BWD daemon.  The
    re-run's A/B/C/D hooks are then no-ops (the coordinator is still
    disabled, so every ``_MSSyncHookFunction.apply`` passes through via the
    ``is_enabled()`` gate), and its activations are cached, so the
    dual-thread ``backward_one_chunk`` reuses them instead of re-running the
    forward on the daemon thread (which would be concurrent FWD-record +
    BWD-replay, re-fire the hooks, and deadlock the coordinator).
"""

import mindspore as ms
from mindspore import mint

from hyper_parallel.platform import get_platform
from hyper_parallel.core.pipeline_parallel.hook_coordinator import HookCoordinator

from mindformers.pynative.distributed.expert_parallel import ExpertParallel

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

    # ------------------------------------------------------------------
    # Overlap seams — the only logic added on top of the synchronous base.
    # The inherited _token_dispatch / _token_combine orchestrators call the
    # a2a primitives and _dispatch_comm / _combine_comm below; here we add the
    # A/B/C/D sync hooks and swap the collectives for async / single-stream.
    # ------------------------------------------------------------------

    def _sync_hook(self, x, hook_name: str):
        """Fire a differentiable A/B/C/D sync hook on ``x`` (identity in the base)."""
        return _platform.differentiable_sync_hook(x, hook_name, self._coordinator)

    @staticmethod
    def _elem_splits(splits, block_size: int):
        """Scale a host split ``int`` list to element counts (pure host, no D2H)."""
        return [s * block_size for s in splits]

    def _async_a2a(self, flat_input, send_splits, recv_splits, block_size: int):
        """Async all-to-all-v returning an AsyncCollectiveTensor (lazy wait)."""
        return _platform.differentiable_all_to_all_single_async(
            flat_input,
            self._elem_splits(send_splits, block_size),
            self._elem_splits(recv_splits, block_size),
            self.ep_group,
        )

    # ---- sync a2a primitives: plain comm.all_to_all_single (self.all_to_all_single),
    # the same collective the base uses, kept on the EP group stream so it funnels
    # with the async token/combine a2a. Overridden vs the base only to use the
    # _single_ variant (the base counts path uses the list-based comm.all_to_all,
    # which would dispatch on a different stream). ----

    def _counts_a2a(self, num_tokens_per_expert, ep_degree):  # pylint: disable=unused-argument
        counts_size = int(num_tokens_per_expert.shape[0])
        output = mint.empty((counts_size,), dtype=num_tokens_per_expert.dtype)
        result = self.all_to_all_single(output, num_tokens_per_expert, group=self.ep_group)
        return result if isinstance(result, ms.Tensor) else result[0]

    def _main_a2a(self, flat_in, input_splits, output_splits, block_size):
        return self._async_a2a(flat_in, input_splits, output_splits, block_size)

    def _combine_a2a(self, flat_in, send_splits, recv_splits, block_size):
        return self._async_a2a(flat_in, send_splits, recv_splits, block_size)

    # ---- communication segments: wrap the a2a with the A/B/C/D sync hooks ----

    def _dispatch_comm(self, flat_in, num_tokens_per_expert, ep_degree, block_size):
        """Overlap dispatch comm: A hook -> counts -> splits -> chunk counts
        (small host list) -> async main a2a -> B hook.

        Reading the chunk counts consumes only the already-complete counts a2a, so
        it does not force the async main a2a's lazy wait — the device-side
        ``split``/``cat`` resort (:meth:`_resort_after_dispatch`) after B is what
        consumes the async output, preserving the overlap window. Replaces the
        routing-map a2a (a third collective) + device sort with the fixed chunk
        permutation of :meth:`_chunk_perm`.
        """
        flat_in = self._sync_hook(flat_in, "A")
        num_tokens_per_expert_group = self._counts_a2a(num_tokens_per_expert, ep_degree)
        # Single batched D2H (consumes only the already-complete counts a2a, not the
        # async main a2a below — the overlap window is preserved); group_list on device.
        input_splits, output_splits, group_counts = self._host_token_splits(
            num_tokens_per_expert, num_tokens_per_expert_group, ep_degree)
        num_tokens_per_expert = self._compute_group_list(num_tokens_per_expert_group, ep_degree)
        flat_out = self._main_a2a(flat_in, input_splits, output_splits, block_size)
        flat_out = self._sync_hook(flat_out, "B")
        return flat_out, group_counts, input_splits, output_splits, num_tokens_per_expert

    def _dispatch_a2a(self, flat_in, host_buf, event, num_tokens_per_expert_group,
                      num_experts, ep_degree, block_size):
        """Async-D2H stage 2 with A/B hooks (used when expert_parallel_async_d2h is on).

        The counts a2a + the async counts D2H were already issued in
        :meth:`ExpertParallel._dispatch_preprocess` (before the permute); here we
        only bracket the *main token a2a* with the A/B rendezvous. The deferred
        ``event.synchronize()`` waits solely on this rank's side-stream counts copy
        (a tiny, by-now-complete transfer) — it is NOT a cross-thread rendezvous, so
        it does not couple with the coordinator's A/B/C/D barriers.
        """
        flat_in = self._sync_hook(flat_in, "A")
        event.synchronize()
        input_splits, output_splits, group_counts = self._derive_splits(
            host_buf.tolist(), num_experts, ep_degree)
        num_tokens_per_expert = self._compute_group_list(num_tokens_per_expert_group, ep_degree)
        flat_out = self._main_a2a(flat_in, input_splits, output_splits, block_size)
        flat_out = self._sync_hook(flat_out, "B")
        return flat_out, group_counts, input_splits, output_splits, num_tokens_per_expert

    def _combine_comm(self, flat_in, input_splits, output_splits, block_size):
        """Overlap combine comm: C hook -> async combine a2a -> D / D_LAST hook."""
        flat_in = self._sync_hook(flat_in, "C")
        flat_out = self._combine_a2a(flat_in, output_splits, input_splits, block_size)
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
