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
"""Transformer SharedExpertMLP"""
__all__ = [
    "SharedExpertMLP",
    "SharedExpertOverlapContext",
]

from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from typing import Optional

import mindspore as ms
from mindspore import ops, Tensor, mint
from mindspore.common._grad_function import _Function
from mindspore.nn.layer import Dense
from mindformers.pynative.transformers.mlp import MLP, MLPSubmodules
from mindformers.parallel_core.transformer_config import TransformerConfig


class _BackwardUseSharedStream(_Function):
    """Switch subsequent shared-expert backward kernels to the side stream."""

    @staticmethod
    def forward(ctx, tensor, overlap_ctx, wait_for_output_grad):  # pylint: disable=arguments-differ
        ctx.overlap_ctx = overlap_ctx
        ctx.wait_for_output_grad = wait_for_output_grad
        return tensor

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        overlap_ctx = ctx.overlap_ctx
        if ctx.wait_for_output_grad and getattr(overlap_ctx, "use_side_stream", True):
            if overlap_ctx.backward_grad_ready_event is None:
                raise RuntimeError("Shared-expert backward started before its output gradient was recorded.")
            overlap_ctx.shared_stream.wait_event(overlap_ctx.backward_grad_ready_event)
        ms.runtime.set_cur_stream(overlap_ctx.shared_stream)
        return grad_output, None, None


class _BackwardUseMainStream(_Function):
    """Restore the main stream after a shared-expert backward segment."""

    @staticmethod
    def forward(ctx, tensor, overlap_ctx, wait_for_shared, *anchors):  # pylint: disable=arguments-differ
        ctx.overlap_ctx = overlap_ctx
        ctx.wait_for_shared = wait_for_shared
        ctx.anchor_count = len(anchors)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        overlap_ctx = ctx.overlap_ctx
        ms.runtime.set_cur_stream(overlap_ctx.main_stream)
        if ctx.wait_for_shared and getattr(overlap_ctx, "use_side_stream", True):
            # The returned gradient is produced on the shared stream.  Insert a
            # stream wait before the autograd engine accumulates it on main.
            overlap_ctx.main_stream.wait_stream(overlap_ctx.shared_stream)
        return (grad_output, None, None) + (None,) * ctx.anchor_count


class _SharedExpertMerge(_Function):
    """Add routed/shared outputs and publish their common backward gradient."""

    @staticmethod
    def forward(ctx, routed_output, shared_output, overlap_ctx):  # pylint: disable=arguments-differ
        ctx.overlap_ctx = overlap_ctx
        return routed_output + shared_output

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        overlap_ctx = ctx.overlap_ctx
        if overlap_ctx.use_side_stream:
            overlap_ctx.backward_grad_ready_event = ms.runtime.Event()
            overlap_ctx.backward_grad_ready_event.record(overlap_ctx.main_stream)
        return grad_output, grad_output, None


@dataclass
class SharedExpertOverlapContext:
    """Invocation-local intermediates for an overlapped shared-expert forward.

    Megatron caches these values on ``SharedExpertMLP`` while its dispatcher
    steps through FC1 and FC2.  Keeping them in a per-invocation object avoids
    cross-talk when a checkpoint replay and the next pipeline microbatch use
    the same layer concurrently.
    """

    hidden_states: Optional[Tensor]
    gate_score: Optional[Tensor] = None
    intermediate: Optional[Tensor] = None
    output: Optional[Tensor] = None
    state: str = "prepared"
    main_stream: object = None
    shared_stream: object = None
    backward_grad_ready_event: object = None
    use_side_stream: bool = True


class SharedExpertMLP(MLP):
    """
    Implementation of a shared expert feedforward block that inherits from MLP.

    This module extends the standard MLP to support shared expert logic, typically used in MoE settings.

    Args:
        config (TransformerConfig): Configuration for the transformer model.
        submodules (MLPSubmodules): The submodules used to construct the MLP, such as activation and linear layers.

    Inputs:
        - **hidden_states** (Tensor) - Input tensor of shape :math:`(S, B, H)`, where
          :math:`S` is sequence length, :math:`B` is batch size, and :math:`H` is hidden size.

    Outputs:
        - **output** (Tensor) - Output tensor of shape :math:`(S, B, H)`.
        - **output_bias** (Tensor) - Bias tensor of shape :math:`(S, B, H)` (if applicable).

    Supported Platforms:
        ``Ascend``
    """

    # Match Megatron's one shared side stream per process.  It is constructed
    # lazily on the active device, never while model parameters live on meta.
    _overlap_stream = None
    _overlap_stream_lock = Lock()

    def __init__(self, config: TransformerConfig, submodules: MLPSubmodules):
        config = deepcopy(config)
        config.ffn_hidden_size = config.moe_shared_expert_intermediate_size
        super().__init__(config, submodules)
        self.cast = ops.cast
        self.router_dense_type = config.moe_router_dtype
        self.use_shared_expert_gate = config.use_shared_expert_gating
        self.compute_dtype = config.compute_dtype
        if self.use_shared_expert_gate:
            self.shared_experts_gate = Dense(in_channels=config.hidden_size,
                                             out_channels=1,
                                             has_bias=False,
                                             dtype=self.router_dense_type)
            self.sigmoid = mint.nn.functional.sigmoid
            self.mul_shared_gate = mint.mul

    @classmethod
    def _get_overlap_stream(cls):
        """Return the lazily-created stream used by all shared experts."""
        if cls._overlap_stream is None:
            with cls._overlap_stream_lock:
                if cls._overlap_stream is None:
                    cls._overlap_stream = ms.runtime.Stream()
        return cls._overlap_stream

    @staticmethod
    def _check_overlap_state(ctx: SharedExpertOverlapContext, expected: str, method: str):
        """Reject a dispatcher calling the split forward out of order."""
        if not isinstance(ctx, SharedExpertOverlapContext):
            raise TypeError(
                f"{method} expects SharedExpertOverlapContext, but got {type(ctx).__name__}."
            )
        if ctx.state != expected:
            raise RuntimeError(
                f"{method} expects shared-expert overlap state '{expected}', "
                f"but got '{ctx.state}'."
            )

    @staticmethod
    def wait_current_stream(ctx: Optional[SharedExpertOverlapContext] = None):
        """Order the selected shared compute stream after the current stream."""
        current_stream = ms.runtime.current_stream()
        shared_stream = (
            ctx.shared_stream if ctx is not None else SharedExpertMLP._get_overlap_stream()
        )
        if ctx is None or ctx.use_side_stream:
            shared_stream.wait_stream(current_stream)

    def _backward_stream_anchors(self):
        """Keep stream-restore nodes live when only parameter grads are requested."""
        anchors = [self.linear_fc1.weight, self.linear_fc2.weight]
        if self.use_shared_expert_gate:
            anchors.append(self.shared_experts_gate.weight)
        return tuple(anchor for anchor in anchors if anchor.requires_grad)

    def overlap_pre_forward(
            self, hidden_states: Tensor, use_side_stream: bool = True
    ) -> SharedExpertOverlapContext:
        """Prepare one shared-expert invocation on the selected compute stream.

        The optional gate is deliberately launched before token permutation,
        mirroring Megatron's ``pre_forward_comm`` placement.  FC1 is deferred
        until immediately after dispatch all-to-all is issued.  EP-only overlap
        may reuse the current stream because HCCL already runs independently;
        callers that need compute/compute concurrency may request the side stream.
        """
        main_stream = ms.runtime.current_stream()
        shared_stream = self._get_overlap_stream() if use_side_stream else main_stream
        if use_side_stream:
            shared_stream.wait_stream(main_stream)
        overlap_ctx = SharedExpertOverlapContext(
            hidden_states=hidden_states,
            main_stream=main_stream,
            shared_stream=shared_stream,
            use_side_stream=use_side_stream,
        )
        with ms.runtime.StreamCtx(shared_stream):
            if self.use_shared_expert_gate:
                gate_input = hidden_states
                if use_side_stream:
                    gate_input = _BackwardUseMainStream.apply(
                        hidden_states, overlap_ctx, True, *self._backward_stream_anchors())
                gate_input = self.cast(gate_input, self.router_dense_type)
                gate_score = self.sigmoid(self.shared_experts_gate(gate_input))
                if use_side_stream:
                    gate_score = _BackwardUseSharedStream.apply(
                        gate_score, overlap_ctx, False)
                overlap_ctx.gate_score = gate_score
        return overlap_ctx

    def overlap_fc1(self, ctx: SharedExpertOverlapContext):
        """Queue shared-expert FC1 and activation after dispatch A2A launch."""
        self._check_overlap_state(ctx, "prepared", "overlap_fc1")
        with ms.runtime.StreamCtx(ctx.shared_stream):
            fc1_input = ctx.hidden_states
            if ctx.use_side_stream:
                fc1_input = _BackwardUseMainStream.apply(
                    fc1_input, ctx, True, *self._backward_stream_anchors())
            intermediate = self._forward_fc1_and_act(fc1_input)
            if ctx.use_side_stream:
                intermediate = _BackwardUseSharedStream.apply(intermediate, ctx, False)
            ctx.intermediate = intermediate
        ctx.hidden_states = None
        ctx.state = "fc1_done"

    def overlap_fc2(self, ctx: SharedExpertOverlapContext):
        """Queue shared-expert FC2 after combine A2A launch."""
        self._check_overlap_state(ctx, "fc1_done", "overlap_fc2")
        with ms.runtime.StreamCtx(ctx.shared_stream):
            fc2_input = ctx.intermediate
            if ctx.use_side_stream:
                fc2_input = _BackwardUseMainStream.apply(
                    fc2_input, ctx, False, *self._backward_stream_anchors())
            output = self._forward_fc2(fc2_input)
            if self.use_shared_expert_gate:
                output = self.mul_shared_gate(output, self.cast(ctx.gate_score, self.compute_dtype))
            if ctx.use_side_stream:
                output = _BackwardUseSharedStream.apply(output, ctx, True)
            ctx.output = output
        ctx.intermediate = None
        ctx.gate_score = None
        ctx.state = "fc2_done"

    def overlap_finish(self, ctx: SharedExpertOverlapContext) -> Tensor:
        """Join the selected shared stream and return its completed output."""
        self._check_overlap_state(ctx, "fc2_done", "overlap_finish")
        current_stream = ms.runtime.current_stream()
        if ctx.use_side_stream:
            current_stream.wait_stream(ctx.shared_stream)
        output = ctx.output
        ctx.output = None
        ctx.state = "finished"
        return output

    @staticmethod
    def merge_with_routed_output(
            routed_output: Tensor,
            shared_output: Tensor,
            ctx: SharedExpertOverlapContext,
    ) -> Tensor:
        """Merge both branches while recording the gradient-ready point for backward."""
        if not ctx.use_side_stream:
            return routed_output + shared_output
        return _SharedExpertMerge.apply(routed_output, shared_output, ctx)

    def construct(self, hidden_states: Tensor, input_ids: Tensor = None) -> Tensor:
        """Construct the shared-expert MLP block."""
        shared_experts_output = super().construct(hidden_states, input_ids)
        if self.use_shared_expert_gate:
            gate_input = self.cast(hidden_states, self.router_dense_type)
            gate = self.sigmoid(self.shared_experts_gate(gate_input))
            shared_experts_output = self.mul_shared_gate(
                shared_experts_output, self.cast(gate, self.compute_dtype))
        return shared_experts_output
