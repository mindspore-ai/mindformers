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
from typing import Any, Optional, Tuple, Dict, Union
from functools import partial
from abc import ABC, abstractmethod

import mindspore as ms
from mindspore import nn, Tensor

from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.core.device_mesh import DeviceMesh
from hyper_parallel.core.placement_types import Placement, Shard, Replicate

from mindformers.pynative.layers.linear import Linear
from mindformers.pynative.base_models.common.embeddings.vocab_embedding import VocabEmbedding
from mindformers.pynative.distributed.utils import distribute_module


__all__ = [
    "ParallelStyle",
    "ColwiseParallel",
    "RowwiseParallel",
    "SequenceParallel",
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


class ColwiseParallel(ParallelStyle):
    """
    Partition a compatible nn.Cell in a column-wise fashion. Currently supports nn.Linear and nn.Embedding.
    Users can compose it together with RowwiseParallel to achieve the sharding of more complicated modules.
    (i.e. MLP, Attention)

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Cell, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be replicated.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Cell, this is used to ensure the output of the nn.Cell
            with the user desired layout. If not specified, the output tensor is sharded on the last dimension.
        use_local_output (bool, optional):
            Whether to use local :class:`Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents Colwise sharding of the nn.Cell.
    """

    def __init__(
        self,
        *,
        input_layouts: Placement | None = None,
        output_layouts: Placement | None = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Replicate(),)
        self.output_layouts = (output_layouts or Shard(-1),)
        # colwise linear runtime sharding (desired sharding):
        # 1. requires replicate input
        # 2. shard output on last dim
        self.desired_input_layouts = (Replicate(),)
        self.use_local_output = use_local_output

    @staticmethod
    # pylint: disable=W0613
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, device_mesh, mod, args
    ):
        """"prepare input."""
        input_tensor = args[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts
            )
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, device_mesh=device_mesh
            )
        return input_tensor

    @staticmethod
    # pylint: disable=W0613
    def _prepare_output_fn(output_layouts, use_local_output, device_mesh, mod, args, outputs):
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, device_mesh=device_mesh)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _create_weight_sharding_plan(self, module: nn.Cell) -> Dict:
        sharding_plan = {}
        if isinstance(module, Linear):
            if getattr(module, "weight", None) is not None:
                sharding_plan.update({"weight": (Shard(0),)})
            if getattr(module, "bias", None) is not None:
                sharding_plan.update({"bias": (Shard(0),)})
        elif isinstance(module, VocabEmbedding):
            sharding_plan = {"weight": (Shard(1),)}
        else:
            raise NotImplementedError(
                "ColwiseParallel currently only support nn.Linear and nn.Embedding!"
            )
        return sharding_plan

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        sharding_plan = self._create_weight_sharding_plan(module=module)

        input_fn = partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            )
        
        output_fn = partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            )
        
        return distribute_module(
            module=module,
            device_mesh=device_mesh,
            parameter_shard_plan=sharding_plan,
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


class RowwiseParallel(ParallelStyle):
    """
    Partition a compatible nn.Cell in a row-wise fashion. Currently supports nn.Linear and nn.Embedding.
    Users can compose it with ColwiseParallel to achieve the sharding of more complicated modules.
    (i.e. MLP, Attention)

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Cell, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be sharded on the last dimension.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Cell, this is used to ensure the output of the nn.Cell
            with the user desired layout. If not specified, the output tensor is replicated.
        use_local_output (bool, optional):
            Whether to use local :class:`Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents Rowwise sharding of the nn.Cell.
    """

    def __init__(
        self,
        *,
        input_layouts: Placement | None = None,
        output_layouts: Placement | None = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Shard(-1),)
        self.output_layouts = (output_layouts or Replicate(),)
        self.use_local_output = use_local_output

    @staticmethod
    # pylint: disable=W0613
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, device_mesh, mod, args,
    ):
        """"prepare input."""
        input_tensor = args[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts
            )
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, device_mesh=device_mesh
            )
        return input_tensor

    @staticmethod
    # pylint: disable=W0613
    def _prepare_output_fn(output_layouts, use_local_output, device_mesh, mod, args, outputs):
        # Rowwise sharding produces partial output, depending on output layouts:
        # 1. to replicate -> allreduce
        # 2. to shard -> reduce_scatter
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, device_mesh=device_mesh)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _create_weight_sharding_plan(self, module: nn.Cell) -> Dict:
        sharding_plan = {}
        if isinstance(module, Linear):
            if getattr(module, "weight", None) is not None:
                sharding_plan.update({"weight": (Shard(1),)})
            if getattr(module, "bias", None) is not None:
                sharding_plan.update({"bias": (Replicate(),)})
            # rowwise linear runtime sharding requires input tensor shard on last dim
            self.desired_input_layouts: tuple[Placement, ...] = (Shard(-1),)
        elif isinstance(module, VocabEmbedding):
            sharding_plan = {"weight": (Shard(0),)}
            # rowwise embedding runtime sharding requires input tensor replicated
            self.desired_input_layouts = (Replicate(),)
        else:
            raise NotImplementedError(
                "RowwiseParallel currently only support nn.Linear and nn.Embedding!"
            )
        return sharding_plan

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        sharding_plan = self._create_weight_sharding_plan(module=module)

        input_fn = partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            )

        output_fn = partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            )

        return distribute_module(
            module=module,
            device_mesh=device_mesh,
            parameter_shard_plan=sharding_plan,
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


class SequenceParallel(ParallelStyle):
    """
    SequenceParallel replicates a compatible ``nn.Cell`` parameters and runs the sharded computation with
    input sharded on the sequence dimension. This currently supports ``nn.LayerNorm``, ``nn.Dropout``, and the
    `RMSNorm python implementation <https://github.com/facebookresearch/llama/blob/main/llama/model.py#L34>`__

    This style implements the operation that is described in the paper
    `Reducing Activation Recomputation in Large Transformer Models <https://arxiv.org/abs/2205.05198>`__

    The output of the ``nn.Cell`` will be sharded on the sequence dimension.

    Keyword Args:
        sequence_dim (int, optional):
            The sequence dimension of the input tensor for the ``nn.Cell``,
            this is used to annotate the input tensor to become a DTensor
            that is sharded on the sequence dimension, default: 1.
        use_local_output (bool, optional):
            Whether to use local :class:`Tensor` instead of :class:`DTensor` for the module output, default: False.
    Returns:
        A :class:`ParallelStyle` object that represents Sequence Parallel of the ``nn.Cell``.

    .. note:: SequenceParallel style assumes ones initialization if there are weights in the nn.Cell (i.e.
        ``nn.LayerNorm`` or ``RMSNorm``, and they by default have ones initialization). If you have custom
        inits for the weights on those modules, you need to broadcast the weights before/after parallelizing
        to ensure that they are replicated.
    """

    def __init__(self, *, sequence_dim: int = 1, use_local_output: bool = False):
        super().__init__()
        self.sequence_sharding = (Shard(sequence_dim),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(sequence_sharding, device_mesh, mod, args):
        """"prepare input."""
        input_tensor = args[0]
        if isinstance(input_tensor, DTensor):
            # if the passed in input DTensor is not sharded on the sequence dim, we need to redistribute it
            if input_tensor.placements != sequence_sharding:
                input_tensor = input_tensor.redistribute(
                    placements=sequence_sharding, device_mesh=device_mesh
                )
            return input_tensor
        if isinstance(input_tensor, Tensor):
            # assume the input passed in already sharded on the sequence dim and create the DTensor
            return DTensor.from_local(
                input_tensor, device_mesh, sequence_sharding
            )
        raise ValueError(
            f"expecting input of {mod} to be a Tensor or DTensor, but got {input_tensor}"
            )

    @staticmethod
    # pylint: disable=W0613
    def _prepare_output_fn(use_local_output, device_mesh, mod, args, outputs):
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        sharding_plan = {}
        for p_name, _ in module.parameters_and_names():
            sharding_plan.update({p_name: (Replicate(),)})

        input_fn = partial(self._prepare_input_fn, self.sequence_sharding)

        output_fn = partial(self._prepare_output_fn, self.use_local_output)

        return distribute_module(
            module=module,
            device_mesh=device_mesh,
            parameter_shard_plan=sharding_plan,
            input_fn=input_fn,
            output_fn=output_fn
        )

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        if len(self.sequence_sharding) == 1:
            tmpstr += f"sequence_dim={self.sequence_sharding[0].dim}, "
        tmpstr += f"use_local_output={self.use_local_output}"
        tmpstr += ")"
        return tmpstr


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
