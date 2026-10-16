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
"""Common utilities for PyNative parameter-efficient tuning."""

import hashlib

import numpy as np
import mindspore as ms

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import _infer_slice_area_by_rank

from mindformers.pynative.dtensor_compat import inplace_copy
from mindformers.tools.logger import logger


def is_lora_a_param_name(name: str) -> bool:
    """Return whether a parameter name identifies a LoRA A tensor."""
    return name == "lora_a" or name.endswith((".lora_a", "_lora_a"))


def is_lora_param_name(name: str) -> bool:
    """Return whether a parameter name follows a supported PyNative LoRA convention."""
    return (
        is_lora_a_param_name(name)
        or name == "lora_b"
        or name.endswith((".lora_b", "_lora_b"))
    )


def init_lora_parameters(lora_a_params, lora_b_params, lora_a_std, lora_b_std):
    """Apply the shared LoRA initialization policy to one or more A/B parameters."""
    for param in lora_a_params:
        param.normal_(mean=0.0, std=lora_a_std)
    for param in lora_b_params:
        if lora_b_std > 0.0:
            param.normal_(mean=0.0, std=lora_b_std)
        else:
            param.zero_()


def reinitialize_lora_parameters(models, seed, lora_a_std, lora_b_std):
    """Initialize LoRA tensors deterministically by parameter name and global shape.

    Pipeline parallelism rebuilds only the layers owned by each stage, while FSDP
    builds every layer on every rank. Consuming one process-global RNG stream during
    delayed initialization therefore assigns different random values to the same LoRA
    parameter under different parallel topologies. A name-derived seed removes that
    construction-order dependency. Explicit layout slicing then maps the same global
    random tensor onto the parameter's local shard, independent of its placement.

    Args:
        models: One model or the model parts local to this rank.
        seed: Training seed used as the root of the per-parameter seeds.
        lora_a_std: Standard deviation for LoRA A initialization.
        lora_b_std: Standard deviation for LoRA B initialization; zero means exact zeros.

    Returns:
        Number of distinct LoRA parameters initialized on this rank.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]

    parameters = {}
    for model in models:
        for _, param in model.parameters_and_names():
            if is_lora_param_name(param.name):
                parameters.setdefault(id(param), param)

    seed = int(seed)
    for param in parameters.values():
        std = lora_a_std if is_lora_a_param_name(param.name) else lora_b_std
        if std <= 0.0:
            local_shape = param.local_shape if isinstance(param, DTensor) else param.shape
            local_value = ms.Tensor(np.zeros(tuple(local_shape), dtype=np.float32), dtype=param.dtype)
            inplace_copy(param, local_value)
            continue

        digest = hashlib.sha256(f"{seed}:{param.name}".encode("utf-8")).digest()
        param_seed = int.from_bytes(digest[:4], byteorder="little")
        global_shape = tuple(param.shape)
        rng = np.random.default_rng(param_seed)
        global_value = rng.normal(0.0, std, size=global_shape).astype(np.float32)

        if isinstance(param, DTensor):
            layout = param.layout
            inner_rank = layout.rank_list.index(layout.mesh.rank)
            area = _infer_slice_area_by_rank(
                layout.mesh_shape, layout.tensor_map, inner_rank, global_shape
            )
            global_value = global_value[tuple(slice(begin, end) for begin, end in area)]
        local_value = ms.Tensor(global_value, dtype=param.dtype)
        inplace_copy(param, local_value)

    logger.info("Deterministically initialized %d LoRA parameter(s) by name.", len(parameters))
    return len(parameters)
