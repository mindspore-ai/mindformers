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
"""Context parallel utilities and wrappers for pynative modules."""

from mindspore import mint, nn, ops
from mindspore.common import dtype as mstype

from hyper_parallel import DeviceMesh

from mindformers.pynative.distributed.style import (
    ParallelStyle,
    build_hp_async_cp_style,
    build_hp_cp_style,
)


def _prepare_attention_qkv(method, args, input_layout=None):
    """Convert model-native SBND QKV to the layout expected by CP FlashAttention."""
    if (input_layout or "").upper() == "TND":
        return tuple(args)
    new_args = list(args)
    query, key, value = new_args[:3]
    seq_len, batch_size = query.shape[:2]
    if method == "colossal":
        query = mint.reshape(query, (seq_len, batch_size, -1))
        key = mint.reshape(key, (key.shape[0], key.shape[1], -1))
        value = mint.reshape(value, (value.shape[0], value.shape[1], -1))
        query = mint.transpose(query, 0, 1)
        key = mint.transpose(key, 0, 1)
        value = mint.transpose(value, 0, 1)
    else:
        query = mint.permute(query, (1, 2, 0, 3))
        key = mint.permute(key, (1, 2, 0, 3))
        value = mint.permute(value, (1, 2, 0, 3))
    new_args[:3] = [query, key, value]
    return tuple(new_args)


def _restore_attention_output(method, output, input_layout=None):
    """Convert CP FlashAttention output back to model-native SBH."""
    if (input_layout or "").upper() == "TND":
        return output
    if method == "colossal":
        return mint.transpose(output, 0, 1)

    output = mint.permute(output, (2, 0, 1, 3))
    seq_len, batch_size = output.shape[:2]
    return mint.reshape(output, (seq_len, batch_size, -1))


class ContextParallelAttentionStyle(ParallelStyle):
    """Attention-side CP: QKV layout hooks + Hyper-Parallel CP on ``core_attention``.

    This matches the intended split with :class:`ContextParallelModelIOStyle`
    (root-model I/O and loss).
    """

    def __init__(self, method: str, cp_size: int, ulysses_degree_in_cp=None, input_layout=None):
        super().__init__()
        self.method = method.lower()
        self.cp_size = cp_size
        self.ulysses_degree_in_cp = ulysses_degree_in_cp
        self.input_layout = input_layout

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        if hasattr(module, "config"):
            setattr(module.config, "_mf_runtime_cp_layout_in_parallel_style", True)
        setattr(module, "_mf_runtime_input_already_in_fa_layout", True)

        def input_fn(hook_module, args, kwargs):
            del hook_module
            return _prepare_attention_qkv(self.method, args, self.input_layout), kwargs

        def output_fn(hook_module, args, kwargs, outputs):
            del hook_module, args, kwargs
            return _restore_attention_output(self.method, outputs, self.input_layout)

        module.register_forward_pre_hook(input_fn, with_kwargs=True)
        hp_style = build_hp_cp_style(
            method=self.method,
            cp_size=self.cp_size,
            ulysses_degree_in_cp=self.ulysses_degree_in_cp,
            input_layout=self.input_layout,
        )
        hp_style._apply(module, device_mesh)
        module.register_forward_hook(output_fn, with_kwargs=True)
        return module


def build_context_parallel_attention_style(method: str,
                                           cp_size: int,
                                           ulysses_degree_in_cp=None,
                                           input_layout=None,
                                           async_enabled: bool = False) -> ParallelStyle:
    """Build the MindFormers attention-side CP style."""
    if async_enabled:
        return build_hp_async_cp_style(
            method=method,
            cp_size=cp_size,
            ulysses_degree_in_cp=ulysses_degree_in_cp,
            input_layout=input_layout,
        )
    return ContextParallelAttentionStyle(
        method=method,
        cp_size=cp_size,
        ulysses_degree_in_cp=ulysses_degree_in_cp,
        input_layout=input_layout,
    )


class ContextParallelModelIOStyle(ParallelStyle):
    """Apply CP input slicing to the root model."""

    _INPUT_NAMES = (
        "input_ids",
        "position_ids",
        "attention_mask",
        "decoder_input",
        "labels",
        "loss_mask",
        "actual_seq_len",
    )

    def __init__(self, cp_mesh, cp_method: str = "colossal",
                 ulysses_degree_in_cp: int = None, mask_type: str = "causal"):
        super().__init__()
        self.cp_mesh = cp_mesh
        self.cp_method = cp_method
        self.ulysses_degree_in_cp = ulysses_degree_in_cp
        self.mask_type = mask_type

    def _prepare_inputs(self, module, args, kwargs):
        """Prepare root-model inputs for CP."""
        del module
        inputs = {}
        for index, name in enumerate(self._INPUT_NAMES):
            if index < len(args):
                inputs[name] = args[index]
            else:
                inputs[name] = kwargs.get(name)

        sharded_inputs = prepare_context_parallel_input(
            inputs,
            self.cp_mesh,
            cp_method=self.cp_method,
            ulysses_degree_in_cp=self.ulysses_degree_in_cp,
            mask_type=self.mask_type,
        )

        new_args = list(args)
        new_kwargs = dict(kwargs)
        for index, name in enumerate(self._INPUT_NAMES):
            if name not in sharded_inputs:
                continue
            if index < len(new_args):
                new_args[index] = sharded_inputs[name]
            else:
                new_kwargs[name] = sharded_inputs[name]
        return tuple(new_args), new_kwargs

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh = None) -> nn.Cell:
        del device_mesh

        def input_fn(mod, args, kwargs):
            return self._prepare_inputs(mod, args, kwargs)

        module.register_forward_pre_hook(input_fn, with_kwargs=True)
        return module


def apply_context_parallel_model_io(model: nn.Cell, parallel_dims, parallelism) -> nn.Cell:
    """Apply root-model CP input slicing hooks."""
    if not getattr(parallel_dims, "cp_enabled", False):
        return model
    return ContextParallelModelIOStyle(
        cp_mesh=parallel_dims.world_mesh["cp"],
        cp_method=getattr(parallelism, "context_parallel_method", "colossal"),
        ulysses_degree_in_cp=getattr(parallelism, "ulysses_degree_in_cp", None),
        mask_type=getattr(parallelism, "context_parallel_mask_type", "causal"),
    )._apply(model)


def attach_context_parallel_runtime_hints(model_config, parallelism_config):
    """Attach context-parallel runtime hints to model config."""
    if parallelism_config is None:
        return model_config

    context_parallel = getattr(parallelism_config, "context_parallel", 1)
    model_config._mf_runtime_context_parallel = context_parallel

    context_parallel_method = getattr(parallelism_config, "context_parallel_method", "colossal").lower()
    model_config._mf_runtime_context_parallel_method = context_parallel_method
    async_enabled = bool(getattr(parallelism_config, "context_parallel_async", False))
    model_config._mf_runtime_context_parallel_async = async_enabled
    if context_parallel > 1 and context_parallel_method == "colossal":
        model_config.context_parallel = context_parallel
        model_config.context_parallel_size = context_parallel
        parallel_config = getattr(model_config, "parallel_config", None)
        if isinstance(parallel_config, dict):
            parallel_config["context_parallel"] = context_parallel
        elif parallel_config is not None:
            setattr(parallel_config, "context_parallel", context_parallel)
        model_config.input_layout = "BSH"

    requested_ulysses_degree = getattr(parallelism_config, "ulysses_degree_in_cp", None)
    if requested_ulysses_degree is not None:
        model_config._mf_runtime_ulysses_degree_in_cp = requested_ulysses_degree
    elif context_parallel_method == "colossal":
        model_config._mf_runtime_ulysses_degree_in_cp = 1
    if async_enabled and context_parallel_method == "ulysses":
        ulysses_degree = getattr(model_config, "_mf_runtime_ulysses_degree_in_cp", None) or context_parallel
        if model_config.num_attention_heads % ulysses_degree != 0:
            raise ValueError(
                f"num_attention_heads ({model_config.num_attention_heads}) must be divisible by "
                f"ulysses_degree ({ulysses_degree})."
            )
    return model_config


def _validate_cp_method(cp_method: str, cp_size: int, ulysses_degree_in_cp: int = None):
    """Validate CP method settings and return the effective Ulysses degree."""
    cp_method = (cp_method or "colossal").lower()
    if cp_method not in {"colossal", "ulysses", "hybrid"}:
        raise NotImplementedError(f"Unsupported context_parallel_method: {cp_method!r}.")
    if cp_method == "colossal":
        return 1
    if cp_method == "ulysses":
        ulysses_degree = cp_size if ulysses_degree_in_cp is None else ulysses_degree_in_cp
        if ulysses_degree != cp_size:
            raise ValueError("Ulysses CP requires ulysses_degree_in_cp == context_parallel.")
        return ulysses_degree

    if ulysses_degree_in_cp is None:
        raise ValueError("Hybrid CP requires ulysses_degree_in_cp to be set.")
    ulysses_degree = ulysses_degree_in_cp
    if ulysses_degree <= 1 or ulysses_degree >= cp_size or cp_size % ulysses_degree != 0:
        raise ValueError("Hybrid CP requires 1 < ulysses_degree_in_cp < context_parallel and divisibility.")
    return ulysses_degree


def prepare_context_parallel_input(inputs: dict,
                                   cp_mesh=None,
                                   cp_method: str = "colossal",
                                   ulysses_degree_in_cp: int = None,
                                   mask_type: str = "causal") -> dict:
    """Slice sequence inputs for context parallel training and build the CP causal mask."""
    if cp_mesh is None or not inputs:
        return inputs

    cp_size = cp_mesh.size()
    if cp_size <= 1:
        return inputs

    cp_rank = cp_mesh.get_local_rank()
    cp_method = (cp_method or "colossal").lower()
    mask_type = (mask_type or "causal").lower()
    use_tnd_actual_seq = inputs.get("actual_seq_len") is not None

    if mask_type != "causal":
        raise NotImplementedError(
            f"Context parallel currently only supports pure causal mask, got {mask_type!r}."
        )
    ulysses_degree = _validate_cp_method(cp_method, cp_size, ulysses_degree_in_cp)

    def _get_chunk_bounds(seq_len):
        if seq_len % cp_size != 0:
            raise ValueError(
                f"Sequence length {seq_len} is not divisible by context_parallel {cp_size}."
            )
        chunk = seq_len // cp_size
        start = cp_rank * chunk
        return start, start + chunk

    def _slice_seq_dim1(tensor):
        if tensor is None or not hasattr(tensor, "shape") or len(tensor.shape) < 2:
            return tensor
        start, end = _get_chunk_bounds(tensor.shape[1])
        return tensor[:, start:end]

    def _slice_position_ids(tensor):
        if tensor is None or not hasattr(tensor, "shape"):
            return tensor
        if len(tensor.shape) == 1:
            start, end = _get_chunk_bounds(tensor.shape[0])
            return tensor[start:end]
        return _slice_seq_dim1(tensor)

    def _find_seq_tensor():
        for key in ("input_ids", "labels", "loss_mask", "position_ids"):
            tensor = inputs.get(key)
            if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) >= 2:
                return tensor
        return None

    def _build_cp_causal_mask(batch_size, q_seq_len, kv_seq_len, q_offset):
        query_positions = mint.arange(q_offset, q_offset + q_seq_len, dtype=mstype.int32)
        key_positions = mint.arange(0, kv_seq_len, dtype=mstype.int32)
        query_positions = mint.reshape(query_positions, (q_seq_len, 1))
        key_positions = mint.reshape(key_positions, (1, kv_seq_len))
        mask = ops.cast(key_positions > query_positions, mstype.uint8)
        mask = mint.reshape(mask, (1, 1, q_seq_len, kv_seq_len))
        return ops.broadcast_to(mask, (batch_size, 1, q_seq_len, kv_seq_len))

    seq_tensor = _find_seq_tensor()
    if seq_tensor is None:
        raise ValueError("Context parallel input preparation requires a sequence tensor.")

    batch_size = seq_tensor.shape[0]
    global_seq_len = seq_tensor.shape[1]
    global_tnd_len = batch_size * global_seq_len if use_tnd_actual_seq else global_seq_len
    global_start, global_end = _get_chunk_bounds(global_tnd_len)
    local_seq_len = global_end - global_start

    def _slice_tnd_flattened(tensor):
        if tensor is None or not hasattr(tensor, "shape") or len(tensor.shape) < 2:
            return tensor
        batch, seq_len = tensor.shape[:2]
        total_len = batch * seq_len
        flat_shape = (1, total_len) + tuple(tensor.shape[2:])
        flat = mint.reshape(tensor, flat_shape)
        start, end = _get_chunk_bounds(total_len)
        return flat[:, start:end]

    sharded = {}
    for key, value in inputs.items():
        if key in {"input_ids", "labels", "loss_mask"}:
            sharded[key] = _slice_tnd_flattened(value) if use_tnd_actual_seq else _slice_seq_dim1(value)
        elif key == "position_ids":
            sharded[key] = _slice_tnd_flattened(value) if use_tnd_actual_seq else _slice_position_ids(value)
        elif key == "attention_mask":
            if value is not None:
                raise NotImplementedError(
                    "Context parallel currently only supports pure causal mask; "
                    "custom attention_mask is not supported."
                )
        elif key == "actual_seq_len" and value is not None:
            sharded[key] = value
        else:
            sharded[key] = value

    if sharded.get("position_ids", None) is None:
        raise ValueError("Context parallel requires position_ids from dataset inputs.")

    if use_tnd_actual_seq:
        return sharded

    if cp_method == "colossal":
        q_seq_len = local_seq_len
        q_offset = cp_rank * local_seq_len
    elif cp_method == "ulysses":
        q_seq_len = global_seq_len
        q_offset = 0
    else:
        co_rank = cp_rank // ulysses_degree
        q_seq_len = local_seq_len * ulysses_degree
        q_offset = co_rank * q_seq_len

    sharded["attention_mask"] = _build_cp_causal_mask(
        batch_size=batch_size,
        q_seq_len=q_seq_len,
        kv_seq_len=global_seq_len,
        q_offset=q_offset,
    )
    return sharded
