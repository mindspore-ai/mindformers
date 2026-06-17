# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Parallel style for distributed tensor parallel.

Modifications from the original PyTorch implementation:
1. Framework migration: PyTorch (torch/nn.Module) replaced with MindSpore
   (mindspore/nn.Cell). DTensor/DeviceMesh/Placement now imported from
   hyper_parallel.core instead of distributed.tensor.
2. Type and API adaptation: nn.Module -> nn.Cell, Tensor -> ms.Tensor
   throughout. DTensor.from_local() and redistribute() calls adjusted to
   hyper_parallel API.
"""

from functools import partial

from mindspore import nn
from hyper_parallel import DTensor, DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Placement, Shard, Replicate

from mindformers.pynative.distributed.utils import distribute_module
from mindformers.pynative.distributed.style import ParallelStyle


__all__ = [
    "NoParallel",
]


class NoParallel(ParallelStyle):
    """
    No parallel style for module.

    This style applies no parallelism to the module, keeping all parameters replicated
    and maintaining local tensor outputs by default.

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Cell, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be replicated.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Cell, this is used to ensure the output of the nn.Cell
            with the user desired layout. If not specified, the output tensor is replicated.
        use_local_output (bool, optional):
            Whether to use local :class:`Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents no parallelism for the nn.Cell.
    """
    def __init__(
            self,
            *,
            input_layouts: Placement | None = None,
            output_layouts: Placement | None = None,
            use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = input_layouts or Replicate()
        self.output_layouts = output_layouts or Replicate()
        self.desired_input_layouts = Replicate()
        self.use_local_output = use_local_output

    @staticmethod
    # pylint: disable=W0613
    def _prepare_input_fn(
            input_layouts, desired_input_layouts, device_mesh, mod, args,
    ):
        """Prepare input tensor for the module."""
        input_tensor = args[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, (input_layouts,)
            )
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=(desired_input_layouts,), device_mesh=device_mesh
            )
        return (input_tensor, *args[1:])

    @staticmethod
    # pylint: disable=W0613
    def _prepare_output_fn(output_layouts, use_local_output, device_mesh, mod, args, outputs):
        """Prepare output tensor for the module."""
        if outputs.placements != (output_layouts,):
            outputs = outputs.redistribute(
                placements=(output_layouts,),
                device_mesh=device_mesh,
            )
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        input_fn = partial(
            self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
        )
        output_fn = partial(
            self._prepare_output_fn, self.output_layouts, self.use_local_output
        )
        # Caller-supplied per-parameter layout overrides, applied only to params the module
        # actually has (e.g. a frozen weight that is otherwise absent from any plan and would
        # stay a plain Tensor); this style stays agnostic of what those extra params are.
        parameter_shard_plan = {
            name: layout for name, layout in getattr(self, "extra_param_layouts", {}).items()
            if getattr(module, name, None) is not None
        }
        return distribute_module(
            module=module,
            device_mesh=device_mesh,
            parameter_shard_plan=parameter_shard_plan,
            input_fn=input_fn,
            output_fn=output_fn
        )

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"input_layouts={self.input_layouts}, "
        tmpstr += f"output_layouts={self.output_layouts}, "
        tmpstr += f"use_local_output={self.use_local_output}"
        tmpstr += ")"
        return tmpstr
