# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Parallel style for distributed tensor parallel.

Modifications from the original PyTorch implementation:
1. Framework migration: PyTorch (torch/nn.Module) replaced with MindSpore
   (mindspore/nn.Cell). DTensor/DeviceMesh/Placement now imported from
   hyper_parallel.core instead of torch.distributed.tensor.
2. Type and API adaptation: nn.Module -> nn.Cell, torch.Tensor -> ms.Tensor
   throughout. DTensor.from_local() and redistribute() calls adjusted to
   hyper_parallel API.
"""
from typing import Any, Optional, Tuple, Dict, Union
from abc import ABC, abstractmethod

from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.core.device_mesh import DeviceMesh
from hyper_parallel.core.placement_types import Placement

import mindspore as ms
from mindspore import nn


__all__ = [
    "ParallelStyle",
    "PrepareModuleInput",
    "PrepareModuleOutput",
    "PrepareModuleInputOutput",
]


class ParallelStyle(ABC):
    """
    The parallel style contract defines how the module or submodule should be parallelized.

    It only defines the ``_apply`` method for parallelization to use, this allows maximum
    flexibility for different kind of style implementations.
    """

    src_data_rank: Optional[int] = 0

    @abstractmethod
    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        """Apply parallel style to the module."""
        raise NotImplementedError("Subclasses must implement the _apply method.")


class PrepareModuleInput(ParallelStyle):
    """
    Configure the nn.Cell's inputs to convert the input tensors of the nn.Cell to DTensors at runtime according to
    ``input_layouts``, and perform layout redistribution according to the ``desired_input_layouts``.

    Keyword Args:
        input_layouts (Union[Placement, Tuple[Optional[Placement], ...]], optional):
            The DTensor layouts of input tensors for the nn.Cell, this is used to convert the input tensors to
            DTensors. If some inputs are not Tensor or no need to convert to DTensors, ``None`` need to be specified
            as a placeholder. default: None.
        desired_input_layouts (Union[Placement, Tuple[Optional[Placement], ...]], optional):
            The desired DTensor layout of input tensors for the nn.Cell, this is used to
            ensure the inputs of the nn.Cell have the desired DTensor layouts. This argument
            needs to have the same length with ``input_layouts``. default: ``None``.
        input_kwarg_layouts (Dict[str, Placement], optional):
            The DTensor layouts of input kwargs for the nn.Cell, this is used to convert
            the input kwarg tensors to DTensors. default: None.
        desired_input_kwarg_layouts (Dict[str, Placement], optional):
            The desired DTensor layout of input kwargs for the nn.Cell, this is used to
            ensure the inputs of the nn.Cell have the desired DTensor layouts. default: None.
        use_local_output (bool, optional):
            Whether to use local :class:`Tensor` instead of :class:`DTensor` for the module inputs, default: False.
    Returns:
        A :class:`ParallelStyle` object that prepares the sharding layouts of the nn.Cell's inputs.

    Example::
        >>> from mindformers.pynative.distributed import PrepareModuleInput
        >>> from hyper_parallel.core.placement_types import Shard, Replicate
        >>> from hyper_parallel.core.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Cell that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh(mesh_shape=(8,), alias_name=("tp",))
        >>>
        >>> # According to the style specified below, the first input of attn will be annotated to Sharded DTensor
        >>> # and then redistributed to Replicated DTensor.
        >>> prepare_input = PrepareModuleInput(
        >>>     input_layouts=(Shard(0), None, None),
        >>>     desired_input_layouts=(Replicate(), None, None)
        >>> )
        >>> prepare_input._apply(block.attn, tp_mesh)
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Union[Placement, Tuple[Optional[Placement], ...]]] = None,
        desired_input_layouts: Optional[Union[Placement, Tuple[Optional[Placement], ...]]] = None,
        input_kwarg_layouts: Optional[Dict[str, Placement]] = None,
        desired_input_kwarg_layouts: Optional[Dict[str, Placement]] = None,
        use_local_output: bool = False,
    ):
        self.input_layouts = (
            (input_layouts,) if isinstance(input_layouts, Placement) else input_layouts
        )
        self.desired_input_layouts = (
            (desired_input_layouts,)
            if isinstance(desired_input_layouts, Placement)
            else desired_input_layouts
        )
        self.use_local_output = use_local_output
        if self.input_layouts is not None:
            if self.desired_input_layouts is None:
                raise ValueError("desired module inputs should not be None!")
            if len(self.input_layouts) != len(self.desired_input_layouts):
                raise ValueError(
                    "input_layouts and desired_input_layouts should have same length!"
                )
        self.with_kwargs = input_kwarg_layouts is not None
        self.input_kwarg_layouts = input_kwarg_layouts or {}
        self.desired_input_kwarg_layouts = desired_input_kwarg_layouts or {}
        if self.with_kwargs:
            if len(self.input_kwarg_layouts) != len(self.desired_input_kwarg_layouts):
                raise ValueError(
                    "input_kwarg_layouts and desired_input_kwarg_layouts should have same length!"
                )

    def _prepare_input_arg(
        self,
        inp: Any,
        mesh: DeviceMesh,
        input_layout: Optional[Placement],
        desired_layout: Optional[Placement],
    ):
        """Prepare a single input argument."""
        if input_layout is not None:
            if isinstance(inp, DTensor):
                dt_inp = inp
            else:
                if not isinstance(inp, ms.Tensor):
                    raise ValueError(
                        f"expecting input to be a Tensor, but got {type(inp)}"
                    )
                dt_inp = DTensor.from_local(
                    inp, mesh, (input_layout,)
                )

            if desired_layout is not None and input_layout != desired_layout:
                dt_inp = dt_inp.redistribute(mesh, (desired_layout,))

            return dt_inp.to_local() if self.use_local_output else dt_inp
        return inp

    def _prepare_input_fn(self, inputs, device_mesh):
        """Prepare input arguments."""
        if self.input_layouts is None:
            return inputs
        prepared_inputs = []
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if len(inputs) != len(self.input_layouts):
            raise ValueError("module inputs and input_layouts should have same length!")

        if self.desired_input_layouts is None:
            raise ValueError("desired module inputs should not be None!")

        for inp, input_layout, desired_layout in zip(
            inputs, self.input_layouts, self.desired_input_layouts
        ):
            prepared_inputs.append(
                self._prepare_input_arg(inp, device_mesh, input_layout, desired_layout)
            )
        return tuple(prepared_inputs)

    def _prepare_input_kwarg_fn(self, inputs, kwarg_inputs, device_mesh):
        """Prepare input arguments and keyword arguments."""
        prepared_arg_inputs = self._prepare_input_fn(inputs, device_mesh)
        prepared_kwarg_inputs = {}
        for kwarg_key, kwarg_val in kwarg_inputs.items():
            input_layout = self.input_kwarg_layouts.get(kwarg_key)
            desired_input_layout = self.desired_input_kwarg_layouts.get(kwarg_key)

            prepared_kwarg_inputs[kwarg_key] = self._prepare_input_arg(
                kwarg_val, device_mesh, input_layout, desired_input_layout
            )

        return (prepared_arg_inputs, prepared_kwarg_inputs)

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        """
        Apply PrepareModuleInput style to the module.

        This method registers a forward pre-hook on the module to prepare inputs
        according to the specified layouts before the module's construct method is called.
        """
        if self.with_kwargs:
            module.register_forward_pre_hook(
                lambda _, inputs, kwargs: self._prepare_input_kwarg_fn(
                    inputs, kwargs, device_mesh
                ),
                with_kwargs=True,
            )
        else:
            module.register_forward_pre_hook(
                lambda _, inputs: self._prepare_input_fn(inputs, device_mesh)
            )
        return module

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"input_layouts={self.input_layouts}, "
        tmpstr += f"desired_input_layouts={self.desired_input_layouts}, "
        tmpstr += f"input_kwarg_layouts={self.input_kwarg_layouts}, "
        tmpstr += f"desired_input_kwarg_layouts={self.desired_input_kwarg_layouts}, "
        tmpstr += f"use_local_output={self.use_local_output}"
        tmpstr += ")"
        return tmpstr


class PrepareModuleOutput(ParallelStyle):
    """
    Configure the nn.Cell's outputs to convert the output tensors of the nn.Cell to DTensors at runtime according to
    ``output_layouts``, and perform layout redistribution according to the ``desired_output_layouts``.

    Keyword Args:
        output_layouts (Union[Placement, Tuple[Optional[Placement], ...]]):
            The DTensor layouts of output tensors for the nn.Cell, this is used to convert the output tensors to
            DTensors if they are :class:`Tensor`. If some outputs are not Tensor or no need to convert to DTensors,
            ``None`` need to be specified as a placeholder.
        desired_output_layouts (Union[Placement, Tuple[Placement, ...]]):
            The desired DTensor layouts of output tensors for the nn.Cell, this is used
            to ensure the outputs of the nn.Cell have the desired DTensor layouts.
        use_local_output (bool, optional):
            Whether to use local :class:`Tensor` instead of :class:`DTensor` for the module outputs, default: True.
    Returns:
        A :class:`ParallelStyle` object that prepares the sharding layouts of the nn.Cell's outputs.

    Example::
        >>> from mindformers.pynative.distributed import PrepareModuleOutput
        >>> from hyper_parallel.core.placement_types import Shard, Replicate
        >>> from hyper_parallel.core.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Cell that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh(mesh_shape=(8,), alias_name=("tp",))
        >>> 
        >>> # According to the style specified below, the output of the TransformerBlock will be converted to Replicated DTensor
        >>> # and then redistributed to Sharded DTensor.
        >>> prepare_output = PrepareModuleOutput(
        >>>     output_layouts=Replicate(),
        >>>     desired_output_layouts=Shard(0)
        >>> )
        >>> prepare_output._apply(block, tp_mesh)
    """

    def __init__(
        self,
        *,
        output_layouts: Union[Placement, Tuple[Optional[Placement], ...]],
        desired_output_layouts: Union[Placement, Tuple[Placement, ...]],
        use_local_output: bool = True,
    ):
        self.output_layouts = (
            (output_layouts,) if isinstance(output_layouts, Placement) else output_layouts
        )
        self.desired_output_layouts = (
            (desired_output_layouts,)
            if isinstance(desired_output_layouts, Placement)
            else desired_output_layouts
        )
        self.use_local_output = use_local_output
        if len(self.output_layouts) != len(self.desired_output_layouts):
            raise ValueError(
                "output_layouts and desired_output_layouts should have same length!"
            )

    def _prepare_out_fn(self, outputs, device_mesh):
        """Prepare output arguments."""
        prepared_outputs = []
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        if len(outputs) != len(self.output_layouts):
            raise ValueError(
                "module outputs and output_layouts should have same length!"
            )

        for out, out_layout, desired_out_layout in zip(
            outputs, self.output_layouts, self.desired_output_layouts
        ):
            if out_layout is not None:
                if isinstance(out, DTensor):
                    dt_out = out
                else:
                    if not isinstance(out, ms.Tensor):
                        raise ValueError(f"expecting output to be a Tensor, but got {type(out)}")
                    dt_out = DTensor.from_local(
                        out, device_mesh, (out_layout,)
                    )

                if out_layout != desired_out_layout:
                    dt_out = dt_out.redistribute(device_mesh, (desired_out_layout,))
                prepared_outputs.append(
                    dt_out.to_local() if self.use_local_output else dt_out
                )
            else:
                prepared_outputs.append(out)

        if len(prepared_outputs) == 1:
            return prepared_outputs[0]
        return tuple(prepared_outputs)

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        """
        Apply PrepareModuleOutput style to the module.

        This method registers a forward hook on the module to prepare outputs
        according to the specified layouts after the module's construct method is called.
        """
        module.register_forward_hook(
            lambda _, inputs, outputs: self._prepare_out_fn(outputs, device_mesh)
        )
        return module

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"output_layouts={self.output_layouts}, "
        tmpstr += f"desired_output_layouts={self.desired_output_layouts}, "
        tmpstr += f"use_local_output={self.use_local_output}"
        tmpstr += ")"
        return tmpstr


class PrepareModuleInputOutput(ParallelStyle):
    """
    Configure the nn.Cell's inputs (and outputs) to convert the input tensors (and output
    tensors, respectively) of the nn.Cell to DTensors at runtime according to
    ``input_layouts`` (and output_layouts, respectively), and perform layout redistribution
    according to the ``desired_input_layouts`` (and ``desired_output_layouts``,
    respectively). This is a combination of :class:`PrepareModuleInput` and
    :class:`PrepareModuleOutput`.

    Keyword Args:
        input_layouts (Union[Placement, Tuple[Optional[Placement], ...]], optional):
            The DTensor layouts of input tensors for the nn.Cell, this is used to convert
            the input tensors to DTensors. If some inputs are not Tensor or no need to
            convert to DTensors, ``None`` need to be specified as a placeholder.
            default: ``None``.
        desired_input_layouts (Union[Placement, Tuple[Optional[Placement], ...]], optional):
            The desired DTensor layout of input tensors for the nn.Cell, this is used to
            ensure the inputs of the nn.Cell have the desired DTensor layouts. This argument
            needs to have the same length with ``input_layouts``. default: None.
        input_kwarg_layouts (Dict[str, Placement], optional):
            The DTensor layouts of input kwargs for the nn.Cell, this is used to convert
            the input kwarg tensors to DTensors. default: None
        desired_input_kwarg_layouts (Dict[str, Placement], optional):
            The desired DTensor layout of input kwargs for the nn.Cell, this is used to ensure the inputs of the nn.Cell
            have the desired DTensor layouts. default: None.
        use_local_input (bool, optional):
            Whether to use local :class:`Tensor` instead of :class:`DTensor` for the module inputs, default: False.
        output_layouts (Union[Placement, Tuple[Optional[Placement], ...]]):
            The DTensor layouts of output tensors for the nn.Cell, this is used to convert
            the output tensors to DTensors if they are :class:`Tensor`. If some outputs are
            not Tensor or no need to convert to DTensors, ``None`` need to be specified as
            a placeholder.
        desired_output_layouts (Union[Placement, Tuple[Placement, ...]]):
            The desired DTensor layouts of output tensors for the nn.Cell,
            this is used to ensure the outputs of the nn.Cell
            have the desired DTensor layouts.
        use_local_output (bool, optional):
            Whether to use local :class:`Tensor` instead of :class:`DTensor` for the module outputs, default: True.
    Returns:
        A :class:`ParallelStyle` object that prepares the sharding layouts of the nn.Cell's inputs and outputs.

    Example::
        >>> from mindformers.pynative.distributed import PrepareModuleInputOutput
        >>> from hyper_parallel.core.placement_types import Shard, Replicate
        >>> from hyper_parallel.core.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Cell that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh(mesh_shape=(8,), alias_name=("tp",))
        >>>
        >>> # According to the style specified below, the first input of attn will be annotated as Sharded DTensor
        >>> # and then redistributed to Replicated DTensor, and the output of the TransformerBlock will be annotated
        >>> # as Replicated DTensor and then redistributed to Sharded DTensor.
        >>> prepare_input_output = PrepareModuleInputOutput(
        >>>     input_layouts=(Shard(0), None, None, ...),
        >>>     desired_input_layouts=(Replicate(), None, None, ...),
        >>>     output_layouts=Replicate(),
        >>>     desired_output_layouts=Shard(0),
        >>> )
        >>> prepare_input_output._apply(block.attn, tp_mesh)
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Union[Placement, Tuple[Optional[Placement], ...]]] = None,
        desired_input_layouts: Optional[Union[Placement, Tuple[Optional[Placement], ...]]] = None,
        input_kwarg_layouts: Optional[Dict[str, Placement]] = None,
        desired_input_kwarg_layouts: Optional[Dict[str, Placement]] = None,
        use_local_input: bool = False,
        output_layouts: Union[Placement, Tuple[Optional[Placement], ...]],
        desired_output_layouts: Union[Placement, Tuple[Placement, ...]],
        use_local_output: bool = True,
    ):
        self.prepare_module_input = PrepareModuleInput(
            input_layouts=input_layouts,
            desired_input_layouts=desired_input_layouts,
            input_kwarg_layouts=input_kwarg_layouts,
            desired_input_kwarg_layouts=desired_input_kwarg_layouts,
            use_local_output=use_local_input,
        )
        self.prepare_module_output = PrepareModuleOutput(
            output_layouts=output_layouts,
            desired_output_layouts=desired_output_layouts,
            use_local_output=use_local_output,
        )

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        """
        Apply PrepareModuleInputOutput style to the module.

        This method applies both PrepareModuleInput and PrepareModuleOutput to the module.
        """
        # Accessing protected _apply is intended here for composing styles.
        # pylint: disable=protected-access
        self.prepare_module_input._apply(module, device_mesh)
        self.prepare_module_output._apply(module, device_mesh)
        return module

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"input_layouts={self.prepare_module_input.input_layouts}, "
        tmpstr += f"desired_input_layouts={self.prepare_module_input.desired_input_layouts}, "
        tmpstr += f"input_kwarg_layouts={self.prepare_module_input.input_kwarg_layouts}, "
        tmpstr += f"desired_input_kwarg_layouts={self.prepare_module_input.desired_input_kwarg_layouts}, "
        tmpstr += f"use_local_input={self.prepare_module_input.use_local_output}, "
        tmpstr += f"output_layouts={self.prepare_module_output.output_layouts}, "
        tmpstr += f"desired_output_layouts={self.prepare_module_output.desired_output_layouts}, "
        tmpstr += f"use_local_output={self.prepare_module_output.use_local_output}"
        tmpstr += ")"
        return tmpstr
