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
"""MC2 (matmul + communication) fused linear primitives for tensor parallelism.

MindSpore exposes the fused kernels ``ops.all_gather_matmul`` and
``ops.matmul_reduce_scatter`` for forward computation only -- neither primitive
ships a registered ``bprop``. This module wraps each fused forward with a custom
backward so the kernels can be used inside a trainable tensor-parallel model.

The two kernels are mathematical duals, which is what makes the backward closed
form (and itself an MC2 kernel):

* ``all_gather_matmul``     forward : ``Y = AllGather_m(X) @ W^T``
                            backward: ``dX = matmul_reduce_scatter(dY, W)``
                                      ``dW = dY^T @ AllGather_m(X)``

* ``matmul_reduce_scatter`` forward : ``Y = ReduceScatter_m(X @ W^T)``
                            backward: ``dX, dY_full = all_gather_matmul(dY, W)``
                                      ``dW = dY_full^T @ X``

All functions operate on **local** (non-DTensor) 2-D tensors; the communication
group and world size are supplied explicitly. ``W`` is always passed with its
native ``(out, in)`` layout and ``trans_x2=True`` so no host-side transpose is
needed (the CANN kernel transposes internally).
"""
from mindspore import Tensor, mint, ops

from hyper_parallel import DTensor
from mindspore.common._grad_function import _Function

from mindformers.pynative.layers.linear import Linear

__all__ = [
    "AllGatherMatmulFunction",
    "MatmulReduceScatterFunction",
    "MC2Linear",
]


def _transpose_2d(tensor):
    """Transpose a 2-D tensor (m, n) -> (n, m)."""
    return mint.transpose(tensor, 0, 1)


class AllGatherMatmulFunction(_Function):
    """Column-parallel fused all-gather + matmul with custom backward.

    Forward (local view): ``out = AllGather_m(x) @ w^T``.

    Args (forward):
        x (Tensor): Local input shard, shape ``(m_local, k)`` (m sharded on the
            communication group / sequence dim).
        w (Tensor): Local weight shard, shape ``(n_local, k)`` (``(out, in)``
            layout, output dim sharded).
        group (str): Communication group name.
        world_size (int): Number of ranks in ``group``.
        bias (Tensor): Optional local bias shard ``(n_local,)`` or ``None``.

    Returns:
        Tensor: ``(m_full, n_local)`` output (m gathered across the group).
    """

    @staticmethod
    def forward(ctx, x, w, group, world_size, bias):  # pylint: disable=arguments-differ
        """Run the fused all-gather + matmul and stash tensors for backward."""
        out, gathered = ops.all_gather_matmul(
            x, w, group, world_size, gather_output=True, trans_x2=True
        )
        if bias is not None:
            out = mint.add(out, bias)
        ctx.save_for_backward(gathered, w)
        ctx.group = group
        ctx.world_size = world_size
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):  # pylint: disable=arguments-differ
        """Gradient: dx via matmul_reduce_scatter, dw via gathered-input matmul."""
        gathered, w = ctx.saved_tensors
        # dx = ReduceScatter_m(grad_out @ w) -> (m_local, k)
        grad_x = ops.matmul_reduce_scatter(
            grad_out, w, ctx.group, ctx.world_size, trans_x2=False
        )
        # dw = grad_out^T @ AllGather_m(x) -> (n_local, k)
        grad_w = mint.matmul(_transpose_2d(grad_out), gathered)
        grad_bias = grad_out.sum(0) if ctx.has_bias else None
        return grad_x, grad_w, None, None, grad_bias


class MatmulReduceScatterFunction(_Function):
    """Row-parallel fused matmul + reduce-scatter with custom backward.

    Forward (local view): ``out = ReduceScatter_m(x @ w^T)``.

    Args (forward):
        x (Tensor): Local input shard, shape ``(m_full, k_local)`` (k/contraction
            sharded on the communication group, m replicated).
        w (Tensor): Local weight shard, shape ``(n, k_local)`` (``(out, in)``
            layout, input dim sharded).
        group (str): Communication group name.
        world_size (int): Number of ranks in ``group``.
        bias (Tensor): Optional replicated bias ``(n,)`` or ``None``.

    Returns:
        Tensor: ``(m_local, n)`` output (m reduce-scattered across the group).
    """

    @staticmethod
    def forward(ctx, x, w, group, world_size, bias):  # pylint: disable=arguments-differ
        """Run the fused matmul + reduce-scatter and stash tensors for backward."""
        out = ops.matmul_reduce_scatter(
            x, w, group, world_size, trans_x2=True
        )
        if bias is not None:
            out = mint.add(out, bias)
        ctx.save_for_backward(x, w)
        ctx.group = group
        ctx.world_size = world_size
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):  # pylint: disable=arguments-differ
        """Gradient: dx via all_gather_matmul, dw via gathered-grad matmul."""
        x, w = ctx.saved_tensors
        # dx = AllGather_m(grad_out) @ w -> (m_full, k_local); also expose grad_out_full
        grad_x, grad_out_full = ops.all_gather_matmul(
            grad_out, w, ctx.group, ctx.world_size, gather_output=True, trans_x2=False
        )
        # dw = AllGather_m(grad_out)^T @ x -> (n, k_local)
        grad_w = mint.matmul(_transpose_2d(grad_out_full), x)
        grad_bias = grad_out_full.sum(0) if ctx.has_bias else None
        return grad_x, grad_w, None, None, grad_bias


class MC2Linear(Linear):
    """Linear using fused matmul and tensor-parallel communication kernels."""

    def configure_mc2(self, mode: str, group: str, world_size: int) -> None:
        """Configure the fused collective used by this layer."""
        if mode not in ("all_gather", "reduce_scatter"):
            raise ValueError(
                "For MC2Linear.configure_mc2, mode should be 'all_gather' or "
                f"'reduce_scatter', but got {mode}."
            )
        self.mc2_mode = mode
        self.mc2_group = group
        self.mc2_world_size = world_size

    @classmethod
    def from_linear(cls, linear: Linear) -> "MC2Linear":
        """Convert a Linear in place while preserving parameters and Cell state."""
        if not isinstance(linear, Linear):
            raise TypeError(f"MC2Linear can only replace Linear, but got {type(linear).__name__}.")
        linear.__class__ = cls
        return linear

    def _mc2_forward(self, input_: Tensor, weight: Tensor) -> Tensor:
        """Run the configured fused kernel on local tensors."""
        seq, batch = int(input_.shape[0]), int(input_.shape[1])
        weight_local = weight.to_local() if isinstance(weight, DTensor) else weight
        input_2d = input_.reshape(-1, input_.shape[-1])

        bias_local = None
        if self.has_bias:
            bias = self.bias
            if bias.dtype != self.compute_dtype:
                bias = self.cast(bias, self.compute_dtype)
            bias_local = bias.to_local() if isinstance(bias, DTensor) else bias

        if self.mc2_mode == "all_gather":
            output_2d = AllGatherMatmulFunction.apply(
                input_2d, weight_local, self.mc2_group, self.mc2_world_size, bias_local
            )
            output_seq = seq * self.mc2_world_size
            output = output_2d.reshape(output_seq, batch, output_2d.shape[-1])
            return output

        output_2d = MatmulReduceScatterFunction.apply(
            input_2d, weight_local, self.mc2_group, self.mc2_world_size, bias_local
        )
        output = output_2d.reshape(seq // self.mc2_world_size, batch, output_2d.shape[-1])
        return output

    def construct(self, input_: Tensor, weight: Tensor = None) -> Tensor:
        """Forward using MC2 with local activations."""
        if isinstance(input_, DTensor):
            raise TypeError("MC2Linear expects a local Tensor activation, but got DTensor.")
        if not hasattr(self, "mc2_mode"):
            raise RuntimeError("MC2Linear must be configured by an MC2 parallel style before use.")

        if weight is None:
            if self.skip_weight_param_allocation:
                raise ValueError(
                    "For MC2Linear, when `skip_weight_param_allocation` is enabled, "
                    "`weight` is required, but got None"
                )
            weight = self.weight

        ori_dtype = input_.dtype
        if input_.dtype != self.compute_dtype:
            input_ = self.cast(input_, self.compute_dtype)
        if weight.dtype != self.compute_dtype:
            weight = self.cast(weight, self.compute_dtype)
        output = self._mc2_forward(input_, weight)
        return self.cast(output, ori_dtype) if output.dtype != ori_dtype else output
