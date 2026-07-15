# Copyright (c) Meta Platforms, Inc. and affiliates
"""Parallel styles for local-tensor distributed execution."""
from typing import Any, Optional, Tuple, Dict, Union, Literal
from abc import ABC, abstractmethod

import mindspore as ms
from mindspore import mint, nn
from mindspore.common._grad_function import _Function
from mindspore.ops.function import comm_func

from hyper_parallel import DTensor, DeviceMesh
from hyper_parallel.core.context_parallel.context_parallel import ContextParallel as HPContextParallel
from hyper_parallel.core.context_parallel.async_context_parallel import AsyncContextParallel as HPAsyncContextParallel
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate

from mindformers.pynative.layers.linear import Linear
from mindformers.pynative.base_models.common.embeddings.vocab_embedding import VocabEmbedding
from mindformers.pynative.distributed.utils import distribute_module


__all__ = [
    "ParallelStyle",
    "HPContextParallelAdapter",
    "HPAsyncContextParallelAdapter",
    "ColwiseParallel",
    "RowwiseParallel",
    "NoParallel",
    "AllGather",
    "ShardTensor",
    "SequenceParallel",
    "PrepareModuleInput",
    "PrepareModuleOutput",
    "PrepareModuleInputOutput",
    "build_hp_cp_style",
    "build_hp_async_cp_style",
]


def _contiguous(tensor):
    return tensor.contiguous() if hasattr(tensor, "contiguous") else tensor


def _normalize_dim(dim: int, ndim: int) -> int:
    """Normalize and validate a possibly-negative tensor dimension."""
    normalized = dim if dim >= 0 else ndim + dim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"Dimension {dim} is out of range for a {ndim}-D tensor.")
    return normalized


def _all_gather_dim(tensor, dim, group):
    """All-gather a local tensor shard along an arbitrary dimension."""
    value = tensor.movedim(dim, 0).contiguous()
    value, _ = comm_func.all_gather_into_tensor(None, value, group=group)
    return value.movedim(0, dim).contiguous()


def _reduce_scatter_dim(tensor, dim, group):
    """Reduce-scatter a tensor along an arbitrary dimension."""
    value = tensor.movedim(dim, 0).contiguous()
    value, _ = comm_func.reduce_scatter_tensor(None, value, group=group)
    return value.movedim(0, dim).contiguous()


def _slice_dim(tensor, dim, world, rank):
    """Select one equal-sized rank shard along ``dim``."""
    if tensor.shape[dim] % world != 0:
        raise ValueError(
            f"Dimension {dim} with size {tensor.shape[dim]} must be divisible by "
            f"mesh size {world}."
        )
    shard_size = tensor.shape[dim] // world
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(rank * shard_size, (rank + 1) * shard_size)
    return tensor[tuple(slices)]


class _AllReduceFunction(_Function):
    """AllReduce partial values in forward and preserve replicated gradients."""

    @staticmethod
    def forward(ctx, tensor, group):  # pylint: disable=arguments-differ
        del ctx
        output, _ = comm_func.all_reduce(tensor, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        del ctx
        return grad_output, None


class _AllGatherFunction(_Function):
    """AllGather with either ReduceScatter or local-slice backward."""

    @staticmethod
    def forward(ctx, tensor, dim, group, world, rank, reduce_grad):  # pylint: disable=arguments-differ
        ctx.dim = dim
        ctx.group = group
        ctx.world = world
        ctx.rank = rank
        ctx.reduce_grad = reduce_grad
        return _all_gather_dim(tensor, dim, group)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        if ctx.reduce_grad:
            grad_input = _reduce_scatter_dim(grad_output, ctx.dim, ctx.group)
        else:
            grad_input = _slice_dim(grad_output, ctx.dim, ctx.world, ctx.rank)
        return grad_input, None, None, None, None, None


class _ShardFunction(_Function):
    """Select a local shard in forward and gather shard gradients in backward."""

    @staticmethod
    def forward(ctx, tensor, dim, group, world, rank):  # pylint: disable=arguments-differ
        ctx.dim = dim
        ctx.group = group
        return _slice_dim(tensor, dim, world, rank)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        return _all_gather_dim(grad_output, ctx.dim, ctx.group), None, None, None, None


class AllGather:
    """Gather a sharded local tensor along ``dim``."""

    def __init__(self, dim: int, *, reduce_grad: bool = True):
        self.dim = dim
        self.reduce_grad = reduce_grad

    def __repr__(self):
        if not self.reduce_grad:
            return f"AllGather(dim={self.dim}, reduce_grad=False)"
        return f"AllGather(dim={self.dim})"


class ShardTensor:
    """Select this rank's local shard along ``dim``."""

    def __init__(self, dim: int):
        self.dim = dim

    def __repr__(self):
        return f"ShardTensor(dim={self.dim})"


TensorTransform = Union[AllGather, ShardTensor]


def _apply_local_transform(tensor, transform, device_mesh):
    """Apply an explicit local tensor transform on a one-dimensional mesh."""
    if transform is None:
        return tensor
    if isinstance(tensor, DTensor):
        raise TypeError("Local parallel styles expect plain Tensor activations, but got DTensor.")

    world = device_mesh.size()
    group = device_mesh.get_group()
    rank = device_mesh.get_local_rank()
    if isinstance(transform, AllGather):
        dim = _normalize_dim(transform.dim, tensor.ndim)
        return _AllGatherFunction.apply(
            tensor, dim, group, world, rank, transform.reduce_grad
        )

    if isinstance(transform, ShardTensor):
        dim = _normalize_dim(transform.dim, tensor.ndim)
        return _ShardFunction.apply(tensor, dim, group, world, rank)

    raise TypeError(f"Unsupported local tensor transform: {transform!r}.")


def _layout_transform(current_layout, desired_layout):
    """Map a simple DTensor layout transition to a local tensor transform."""
    if current_layout is None or desired_layout is None or current_layout == desired_layout:
        return None
    if isinstance(current_layout, Shard) and isinstance(desired_layout, Replicate):
        return AllGather(current_layout.dim)
    if isinstance(current_layout, Replicate) and isinstance(desired_layout, Shard):
        return ShardTensor(desired_layout.dim)
    raise ValueError(
        f"Unsupported local layout transition: {current_layout!r} -> {desired_layout!r}."
    )


def _layout_transforms(current_layouts, desired_layouts):
    """Map matching layout tuples to local tensor transforms."""
    if current_layouts is None or desired_layouts is None:
        return None
    current_items = current_layouts if isinstance(current_layouts, tuple) else (current_layouts,)
    desired_items = desired_layouts if isinstance(desired_layouts, tuple) else (desired_layouts,)
    if len(current_items) != len(desired_items):
        raise ValueError("current_layouts and desired_layouts should have same length!")
    return tuple(
        _layout_transform(current_layout, desired_layout)
        for current_layout, desired_layout in zip(current_items, desired_items)
    )


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


class HPContextParallelAdapter(ParallelStyle):
    """
    Adapter that bridges Hyper-Parallel CP styles into MindFormers' style API.

    The CP runtime logic remains in Hyper-Parallel. This wrapper only lets
    ``parallelize_module`` treat it like a local ``ParallelStyle``.
    """

    def __init__(self, hp_style: HPContextParallel):
        super().__init__()
        self.hp_style = hp_style

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        return self.hp_style.apply(module, device_mesh)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hp_style={self.hp_style})"


class HPAsyncContextParallelAdapter(ParallelStyle):
    """
    Adapter for Hyper-Parallel async CP styles.

    ``parallelize_module`` can still apply the style synchronously when no
    handoff modules are provided, while ``apply_with_qkv`` is used by the MLA
    async path to pass the final Q/K/V handoff cells explicitly.
    """

    def __init__(self, hp_style: HPAsyncContextParallel):
        super().__init__()
        self.hp_style = hp_style

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        return self.hp_style.apply(module, device_mesh)

    def apply_with_qkv(
            self,
            module: nn.Cell,
            device_mesh: DeviceMesh,
            q_proj: nn.Cell,
            k_proj: nn.Cell,
            v_proj: nn.Cell,
    ) -> nn.Cell:
        """Apply async CP with explicit final Q/K/V handoff modules."""
        return self.hp_style.apply(
            module,
            device_mesh,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
        )

    def apply_to_attention(self, attention_module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        """Apply async CP to an attention module that exposes handoff boundaries."""
        required_cells = ("core_attention", "q_handoff", "k_handoff", "v_handoff")
        missing = [name for name in required_cells if not hasattr(attention_module, name)]
        if missing:
            raise ValueError(f"Async context parallel requires attention cells {missing}.")
        config = getattr(attention_module, "config", None)
        if config is not None:
            if not getattr(config, "use_flash_attention", False):
                raise NotImplementedError("Async context parallel requires flash attention.")
            if getattr(config, "use_ring_attention", False):
                raise NotImplementedError("Async context parallel does not support ring attention.")
            input_layout = getattr(config, "input_layout", None)
            use_eod_mask = getattr(config, "use_eod_attn_mask_compression", False)
            if input_layout == "TND" and not use_eod_mask:
                raise NotImplementedError("Async TND context parallel requires compressed EOD masks.")
            if input_layout != "TND" and use_eod_mask:
                raise NotImplementedError(
                    "Async context parallel with compressed EOD masks requires input_layout='TND'."
                )
            if input_layout not in ("BNSD", "TND"):
                raise NotImplementedError(
                    f"Async context parallel expects input_layout='BNSD' or 'TND', got {input_layout!r}."
                )
        return self.apply_with_qkv(
            module=attention_module.core_attention,
            device_mesh=device_mesh,
            q_proj=attention_module.q_handoff,
            k_proj=attention_module.k_handoff,
            v_proj=attention_module.v_handoff,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hp_style={self.hp_style})"


def build_hp_cp_style(
        method: str,
        cp_size: int,
        ulysses_degree_in_cp: Optional[int] = None,
        input_layout: Optional[str] = None,
) -> HPContextParallelAdapter:
    """
    Build a thin MindFormers wrapper for Hyper-Parallel CP styles.

    Args:
        method: Context parallel method from trainer config.
        cp_size: Context parallel degree.

    Returns:
        HPContextParallelAdapter instance ready for ``parallelize_module``.
    """
    method = method.lower()
    input_layout = input_layout.upper() if isinstance(input_layout, str) else input_layout
    if input_layout == "TND":
        if method == "colossal":
            ulysses_degree = 1
        elif method == "ulysses":
            ulysses_degree = cp_size if ulysses_degree_in_cp is None else ulysses_degree_in_cp
            if ulysses_degree != cp_size:
                raise ValueError("TND Ulysses CP requires ulysses_degree_in_cp == context_parallel.")
        elif method == "hybrid":
            if ulysses_degree_in_cp is None:
                raise ValueError("TND Hybrid CP requires ulysses_degree_in_cp to be set.")
            ulysses_degree = ulysses_degree_in_cp
        else:
            raise NotImplementedError(f"unsupported context parallel method: {method}")
        hp_style = HPContextParallel(
            seq_dim=0,
            head_dim=1,
            ulysses_degree=ulysses_degree,
            qkv_indices=(0, 1, 2),
        )
        return HPContextParallelAdapter(hp_style)

    if method == "colossal":
        hp_style = HPContextParallel(
            # Colossal CP feeds FlashAttention with BSH QKV, so sequence is dim 1.
            seq_dim=1,
            head_dim=2,
            ulysses_degree=1,
            qkv_indices=(0, 1, 2),
        )
        return HPContextParallelAdapter(hp_style)

    if method == "ulysses":
        ulysses_degree = cp_size if ulysses_degree_in_cp is None else ulysses_degree_in_cp
        hp_style = HPContextParallel(
            # Synchronous Ulysses/Hybrid feeds FlashAttention with FA-native
            # BNSD tensors, so sequence/head are dims 2/1 respectively.
            seq_dim=2,
            head_dim=1,
            ulysses_degree=ulysses_degree,
            qkv_indices=(0, 1, 2),
        )
        return HPContextParallelAdapter(hp_style)

    if method == "hybrid":
        hp_style = HPContextParallel(
            # Synchronous Ulysses/Hybrid feeds FlashAttention with FA-native
            # BNSD tensors, so sequence/head are dims 2/1 respectively.
            seq_dim=2,
            head_dim=1,
            ulysses_degree=ulysses_degree_in_cp,
            qkv_indices=(0, 1, 2),
        )
        return HPContextParallelAdapter(hp_style)

    raise NotImplementedError(f"unsupported context parallel method: {method}")


def build_hp_async_cp_style(
        method: str,
        cp_size: int,
        ulysses_degree_in_cp: Optional[int] = None,
        input_layout: Optional[str] = None,
) -> HPAsyncContextParallelAdapter:
    """Build a MindFormers wrapper for Hyper-Parallel async CP styles."""
    method = method.lower()
    input_layout = input_layout.upper() if isinstance(input_layout, str) else input_layout
    if method == "colossal":
        raise NotImplementedError(
            "Async context parallel currently supports 'ulysses' and 'hybrid' only."
        )

    if method == "ulysses":
        ulysses_degree = cp_size if ulysses_degree_in_cp is None else ulysses_degree_in_cp
    elif method == "hybrid":
        if ulysses_degree_in_cp is None:
            raise ValueError("Hybrid async CP requires ulysses_degree_in_cp to be set.")
        ulysses_degree = ulysses_degree_in_cp
    else:
        raise NotImplementedError(f"unsupported context parallel method: {method}")

    hp_style = HPAsyncContextParallel(
        seq_dim=0,
        head_dim=1 if input_layout == "TND" else 2,
        ulysses_degree=ulysses_degree,
        qkv_indices=(0, 1, 2),
    )
    return HPAsyncContextParallelAdapter(hp_style)


class ColwiseParallel(ParallelStyle):
    """Shard linear output features and compute with local tensors.

    ``gather_input`` gathers a sequence-sharded input before the local matmul.
    """

    def __init__(
        self,
        *,
        gather_input: bool = True,
        gather_output: bool = False,
        output_layouts=None,
        use_local_output=None,
    ):
        super().__init__()
        del output_layouts, use_local_output
        # Local flow: all-gather the sequence-sharded input before the local matmul
        # (SP). ``gather_input=False`` when the input is already full sequence.
        self.gather_input = gather_input
        self.gather_output = gather_output

    def _create_weight_sharding_plan(self, module: nn.Cell) -> Dict:
        """Create weight sharding plan for column-wise parallelism."""
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
        # Caller-supplied per-parameter layout overrides, applied only to params the module
        # actually has, so this style stays agnostic of what those extra params are.
        for name, layout in getattr(self, "extra_param_layouts", {}).items():
            if getattr(module, name, None) is not None:
                sharding_plan[name] = layout
        return sharding_plan

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        # Shard the weight (Shard(0)); no DTensor redistribute hooks -- the module
        # computes locally on its weight shard. The tensor-parallel collective is a
        # hand-written all-gather of the sequence-sharded input in a pre-hook.
        sharding_plan = self._create_weight_sharding_plan(module=module)
        distribute_module(
            module=module,
            device_mesh=device_mesh,
            parameter_shard_plan=sharding_plan,
            input_fn=None,
            output_fn=None,
        )
        if self.gather_input:
            input_transform = AllGather(0)

            def _pre_hook(unused_cell, args):
                del unused_cell
                gathered = _apply_local_transform(args[0], input_transform, device_mesh)
                return (gathered,) + tuple(args[1:])

            module.register_forward_pre_hook(_pre_hook)
        if self.gather_output:
            # The gathered output is replicated, so its backward is a local slice,
            # matching Megatron's gather_from_tensor_model_parallel_region.
            output_transform = AllGather(-1, reduce_grad=False)

            def _post_hook(unused_cell, unused_args, output):
                del unused_cell, unused_args
                return _apply_local_transform(output, output_transform, device_mesh)

            module.register_forward_hook(_post_hook)
        return module

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"gather_input={self.gather_input}, gather_output={self.gather_output}"
        tmpstr += ")"
        return tmpstr


class RowwiseParallel(ParallelStyle):
    """Shard linear input features and reduce partial outputs over TP."""

    def __init__(
        self,
        *,
        reduce_mode: Literal["all_reduce", "reduce_scatter"] = "reduce_scatter",
        input_is_parallel: bool = True,
    ):
        super().__init__()
        if reduce_mode not in ("all_reduce", "reduce_scatter"):
            raise ValueError(
                "reduce_mode must be 'all_reduce' or 'reduce_scatter', "
                f"but got {reduce_mode!r}."
            )
        self.reduce_mode = reduce_mode
        if not isinstance(input_is_parallel, bool):
            raise TypeError(
                f"input_is_parallel must be a bool, but got {type(input_is_parallel).__name__}."
            )
        self.input_is_parallel = input_is_parallel

    def _create_weight_sharding_plan(self, module: nn.Cell) -> Dict:
        """Create weight sharding plan for row-wise parallelism."""
        sharding_plan = {}
        if isinstance(module, Linear):
            if getattr(module, "weight", None) is not None:
                sharding_plan.update({"weight": (Shard(1),)})
            if getattr(module, "bias", None) is not None:
                sharding_plan.update({"bias": (Replicate(),)})
        elif isinstance(module, VocabEmbedding):
            sharding_plan = {"weight": (Shard(0),)}
        else:
            raise NotImplementedError(
                "RowwiseParallel currently only support nn.Linear and nn.Embedding!"
            )
        # Caller-supplied per-parameter layout overrides, applied only to params the module
        # actually has, so this style stays agnostic of what those extra params are.
        for name, layout in getattr(self, "extra_param_layouts", {}).items():
            if getattr(module, name, None) is not None:
                sharding_plan[name] = layout
        return sharding_plan

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        if isinstance(module, VocabEmbedding):
            return self._apply_vocab_embedding(module, device_mesh)

        # Shard the weight (Shard(1)); no DTensor redistribute hooks. The partial
        # output is reduced by a hand-written post-hook; a Replicate bias is added
        # there, after the reduce (defer it out of the per-rank matmul).
        sharding_plan = self._create_weight_sharding_plan(module=module)
        if getattr(module, "bias", None) is not None:
            module.skip_add_bias = True
        distribute_module(
            module=module,
            device_mesh=device_mesh,
            parameter_shard_plan=sharding_plan,
            input_fn=None,
            output_fn=None,
        )
        group = device_mesh.get_group()
        if not self.input_is_parallel:
            transform = ShardTensor(-1)

            def _pre_hook(unused_cell, args):
                del unused_cell
                sharded = _apply_local_transform(args[0], transform, device_mesh)
                return (sharded,) + tuple(args[1:])

            module.register_forward_pre_hook(_pre_hook)

        def _post_hook(cell, unused_args, output):
            del unused_args
            out = _contiguous(output)
            if self.reduce_mode == "reduce_scatter":
                out, _ = comm_func.reduce_scatter_tensor(None, out, group=group)
            else:
                out = _AllReduceFunction.apply(out, group)
            if getattr(cell, "has_bias", False) and getattr(cell, "skip_add_bias", False):
                bias = cell.bias.to_local() if isinstance(cell.bias, DTensor) else cell.bias
                out = out + cell.cast(bias, out.dtype)
            return out

        module.register_forward_hook(_post_hook)
        return module

    def _apply_vocab_embedding(self, module: VocabEmbedding, device_mesh: DeviceMesh) -> nn.Cell:
        """Apply vocab-row sharding, masking, and output reduction to an embedding."""
        distribute_module(
            module=module,
            device_mesh=device_mesh,
            parameter_shard_plan={"weight": (Shard(0),)},
            input_fn=None,
            output_fn=None,
        )
        world = device_mesh.size()
        rank = device_mesh.get_local_rank()
        group = device_mesh.get_group()
        vocab_per_rank = module.num_embeddings // world
        vocab_start = rank * vocab_per_rank
        vocab_end = vocab_start + vocab_per_rank
        masks = []

        def _pre_hook(unused_cell, args):
            del unused_cell
            ids = args[0]
            in_range = mint.logical_and(ids >= vocab_start, ids < vocab_end)
            masks.append(in_range)
            local_ids = (ids - vocab_start) * in_range.to(ids.dtype)
            return (local_ids,) + tuple(args[1:])

        def _post_hook(unused_cell, unused_args, output):
            del unused_cell, unused_args
            in_range = masks.pop()
            bsz, seq = in_range.shape
            output = output * in_range.reshape(bsz, seq, 1).to(output.dtype)
            if self.reduce_mode == "all_reduce":
                return _AllReduceFunction.apply(output.contiguous(), group)
            output = output.transpose(0, 1).contiguous()
            output, _ = comm_func.reduce_scatter_tensor(None, output, group=group)
            return output.transpose(0, 1).contiguous()

        module.register_forward_pre_hook(_pre_hook)
        module.register_forward_hook(_post_hook)
        return module

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(reduce_mode={self.reduce_mode!r}, "
            f"input_is_parallel={self.input_is_parallel})"
        )


class NoParallel(ParallelStyle):
    """Replicate module parameters and leave local activations unchanged."""

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        sharding_plan = {name: (Replicate(),) for name, _ in module.parameters_and_names()}
        for name, layout in getattr(self, "extra_param_layouts", {}).items():
            if getattr(module, name, None) is not None:
                sharding_plan[name] = layout
        return distribute_module(
            module=module,
            device_mesh=device_mesh,
            parameter_shard_plan=sharding_plan,
            input_fn=None,
            output_fn=None,
        )


class SequenceParallel(NoParallel):
    """Replicate module parameters and compute on a local sequence shard."""

    def __init__(self, *, sequence_dim: int = 1, input_is_parallel: bool = True):
        super().__init__()
        if not isinstance(sequence_dim, int):
            raise TypeError(f"sequence_dim must be an int, but got {type(sequence_dim).__name__}.")
        self.sequence_dim = sequence_dim
        if not isinstance(input_is_parallel, bool):
            raise TypeError(
                f"input_is_parallel must be a bool, but got {type(input_is_parallel).__name__}."
            )
        self.input_is_parallel = input_is_parallel

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        module = super()._apply(module, device_mesh)
        if not self.input_is_parallel:
            transform = ShardTensor(self.sequence_dim)

            def _pre_hook(unused_cell, args):
                del unused_cell
                sharded = _apply_local_transform(args[0], transform, device_mesh)
                return (sharded,) + tuple(args[1:])

            module.register_forward_pre_hook(_pre_hook)
        return module

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sequence_dim={self.sequence_dim}, "
            f"input_is_parallel={self.input_is_parallel})"
        )


class PrepareModuleInput(ParallelStyle):
    """Apply explicit local transforms to positional or keyword inputs."""

    def __init__(
        self,
        *,
        input_transforms: Optional[Union[TensorTransform, Tuple[Optional[TensorTransform], ...]]] = None,
        input_kwarg_transforms: Optional[Dict[str, TensorTransform]] = None,
        input_layouts=None,
        desired_input_layouts=None,
        use_local_input=None,
    ):
        del use_local_input
        if input_transforms is None and input_layouts is not None:
            input_transforms = _layout_transforms(input_layouts, desired_input_layouts)
        self.input_transforms = (
            (input_transforms,) if isinstance(input_transforms, (AllGather, ShardTensor))
            else input_transforms
        )
        self.with_kwargs = input_kwarg_transforms is not None
        self.input_kwarg_transforms = input_kwarg_transforms or {}

    def _prepare_input_arg(
        self,
        inp: Any,
        mesh: DeviceMesh,
        transform: Optional[TensorTransform],
    ):
        """Prepare a single plain-tensor input argument."""
        if transform is None:
            return inp
        if inp is None:
            # Optional tensor kwargs (for example ``input_ids`` on non-hash
            # MTP MoE layers) remain absent under a parallel input transform.
            return None
        if not isinstance(inp, ms.Tensor):
            raise ValueError(f"expecting input to be a Tensor, but got {type(inp)}")
        return _apply_local_transform(inp, transform, mesh)

    def _prepare_input_fn(self, inputs, device_mesh):
        """Prepare input arguments."""
        if self.input_transforms is None:
            return inputs
        prepared_inputs = []
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if len(inputs) < len(self.input_transforms):
            missing_transforms = self.input_transforms[len(inputs):]
            if any(transform is not None for transform in missing_transforms):
                raise ValueError(
                    "module inputs cannot omit arguments with a non-empty input transform!"
                )

        for inp, transform in zip(inputs, self.input_transforms):
            prepared_inputs.append(
                self._prepare_input_arg(inp, device_mesh, transform)
            )
        prepared_inputs.extend(inputs[len(self.input_transforms):])
        return tuple(prepared_inputs)

    def _prepare_input_kwarg_fn(self, inputs, kwarg_inputs, device_mesh):
        """Prepare input arguments and keyword arguments."""
        prepared_arg_inputs = self._prepare_input_fn(inputs, device_mesh)
        prepared_kwarg_inputs = {}
        for kwarg_key, kwarg_val in kwarg_inputs.items():
            prepared_kwarg_inputs[kwarg_key] = self._prepare_input_arg(
                kwarg_val, device_mesh, self.input_kwarg_transforms.get(kwarg_key)
            )

        return (prepared_arg_inputs, prepared_kwarg_inputs)

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        """
        Apply PrepareModuleInput style to the module.

        This method registers a forward pre-hook on the module to prepare inputs
        according to the specified transforms before the module's construct method is called.
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
        tmpstr += f"input_transforms={self.input_transforms}, "
        tmpstr += f"input_kwarg_transforms={self.input_kwarg_transforms}"
        tmpstr += ")"
        return tmpstr


class PrepareModuleOutput(ParallelStyle):
    """Apply explicit local transforms to module outputs."""

    def __init__(
        self,
        *,
        output_transforms: Optional[Union[TensorTransform, Tuple[Optional[TensorTransform], ...]]] = None,
        output_layouts=None,
        desired_output_layouts=None,
        use_local_output=None,
    ):
        del use_local_output
        if output_transforms is None and output_layouts is not None:
            output_transforms = _layout_transforms(output_layouts, desired_output_layouts)
        self.output_transforms = (
            (output_transforms,) if isinstance(output_transforms, (AllGather, ShardTensor))
            else output_transforms
        )

    def _prepare_out_fn(self, outputs, device_mesh):
        """Prepare output arguments."""
        prepared_outputs = []
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        if len(outputs) != len(self.output_transforms):
            raise ValueError(
                "module outputs and output_transforms should have same length!"
            )

        for out, transform in zip(outputs, self.output_transforms):
            if transform is None:
                prepared_outputs.append(out)
                continue
            if not isinstance(out, ms.Tensor):
                raise ValueError(f"expecting output to be a Tensor, but got {type(out)}")
            prepared_outputs.append(
                _apply_local_transform(out, transform, device_mesh)
            )

        if len(prepared_outputs) == 1:
            return prepared_outputs[0]
        return tuple(prepared_outputs)

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        """
        Apply PrepareModuleOutput style to the module.

        This method registers a forward hook on the module to prepare outputs
        according to the specified transforms after the module's construct method is called.
        """
        module.register_forward_hook(
            lambda _, inputs, outputs: self._prepare_out_fn(outputs, device_mesh)
        )
        return module

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"output_transforms={self.output_transforms}"
        tmpstr += ")"
        return tmpstr


class PrepareModuleInputOutput(ParallelStyle):
    """Compose explicit local input and output transforms around a module."""

    def __init__(
        self,
        *,
        input_transforms: Optional[Union[TensorTransform, Tuple[Optional[TensorTransform], ...]]] = None,
        input_kwarg_transforms: Optional[Dict[str, TensorTransform]] = None,
        output_transforms: Optional[Union[TensorTransform, Tuple[Optional[TensorTransform], ...]]] = None,
        input_layouts=None,
        desired_input_layouts=None,
        output_layouts=None,
        desired_output_layouts=None,
        use_local_input=None,
        use_local_output=None,
    ):
        self.prepare_module_input = PrepareModuleInput(
            input_transforms=input_transforms,
            input_kwarg_transforms=input_kwarg_transforms,
            input_layouts=input_layouts,
            desired_input_layouts=desired_input_layouts,
            use_local_input=use_local_input,
        )
        self.prepare_module_output = PrepareModuleOutput(
            output_transforms=output_transforms,
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
        tmpstr += f"input_transforms={self.prepare_module_input.input_transforms}, "
        tmpstr += f"input_kwarg_transforms={self.prepare_module_input.input_kwarg_transforms}, "
        tmpstr += f"output_transforms={self.prepare_module_output.output_transforms}"
        tmpstr += ")"
        return tmpstr
