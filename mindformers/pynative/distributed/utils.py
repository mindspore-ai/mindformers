# Copyright (c) Meta Platforms, Inc. and affiliates
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

"""Utilities for distributing modules in pynative mode."""

import inspect
from functools import partial
from typing import Any, Callable, Dict, Optional

from mindspore import nn, Tensor
from mindspore.common import dtype as mstype
from mindspore.mint.distributed import get_world_size

from hyper_parallel import DeviceMesh, shard_module
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan

from mindformers.tools import logger


def vocab_parallel_shard_dim(logits):
    """If ``logits`` is a DTensor sharded on its last (vocab) dimension, return the
    index of the mesh dimension carrying that shard; otherwise return ``None``.

    The output_layer is registered as ``ColwiseParallel`` with
    ``output_layouts=Shard(-1)`` when ``enable_loss_parallel`` is set, so the logits
    reaching the loss are sharded on the vocab dimension across the TP mesh. Every
    other mesh dimension stays ``Replicate``. Detecting the shard at runtime keeps the
    loss free of extra plumbing and falls back automatically when logits are replicated.
    """
    if not isinstance(logits, DTensor):
        return None
    last_dim = len(logits.shape) - 1
    for mesh_dim, placement in enumerate(logits.placements):
        if placement.is_shard() and placement.dim in (last_dim, -1):
            return mesh_dim
    return None


def distribute_module(
    module: nn.Cell,
    device_mesh: DeviceMesh,
    parameter_shard_plan: Optional[Dict[str, Any]] = None,
    input_fn: Optional[Callable[..., None]] = None,
    output_fn: Optional[Callable[..., None]] = None,
) -> nn.Cell:
    """
    Distribute a module to the device mesh.
    Args:
        module (nn.Cell): The module to be distributed.
        device_mesh (DeviceMesh): The device mesh to distribute the module to.
        parameter_shard_plan (Dict[str, Any]): The shard plan for the parameters.
        input_fn (Callable[[DeviceMesh, nn.Cell, Any], None]): The input function.
        output_fn (Callable[[DeviceMesh, nn.Cell, Any, Any], None]): The output function.
    Returns:
        nn.Cell: The distributed module.
    """
    already_distributed = getattr(module, "_distribute_module_applied", False)
    if already_distributed:
        raise RuntimeError(
            "distribute_module should only be called once on a module, "
            "but it has already been called on this module!"
        )

    parameter_shard_plan = parameter_shard_plan or {}
    for name, _ in module.parameters_and_names():
        if name in parameter_shard_plan:
            continue
        logger.info(f"Add replicate plan for {name} because it is not in parameter_shard_plan")
        parameter_shard_plan[name] = (Replicate(), )
    sharding_plan = ShardingPlan(plan=parameter_shard_plan)
    shard_module(module, device_mesh, sharding_plan)

    if input_fn is not None:
        num_args = len(inspect.signature(input_fn).parameters)
        if num_args == 3:
            module.register_forward_pre_hook(partial(input_fn, device_mesh))
        elif num_args == 4:
            module.register_forward_pre_hook(
                partial(input_fn, device_mesh),
                with_kwargs=True,
            )
        else:
            raise ValueError(
                "input_fn should take in 3 arguments "
                "(device_mesh, module, args) or 4 arguments "
                "(device_mesh, module, args, kwargs), but got "
                f"{num_args} arguments!"
            )

    if output_fn is not None:
        num_args = len(inspect.signature(output_fn).parameters)
        if num_args == 4:
            module.register_forward_hook(partial(output_fn, device_mesh))
        elif num_args == 5:
            module.register_forward_hook(
                partial(output_fn, device_mesh),
                with_kwargs=True,
            )
        else:
            raise ValueError(
                "output_fn should take in 4 arguments "
                "(device_mesh, module, args, output) or 5 arguments "
                "(device_mesh, module, args, kwargs, output), but got "
                f"{num_args} arguments!"
            )

    setattr(module, "_distribute_module_applied", True)
    return module


def get_loss_sense(
    parallelism,
    enable_parallel: bool = False,
    gradient_accumulation_steps: int = 1,
    apply_gradient_accumulation: bool = False,
) -> Tensor:
    """Calculate the backward sense for the main or auxiliary loss.

    Args:
        parallelism: Parallelism configuration containing ``pipeline_parallel``.
        enable_parallel: Whether distributed training is enabled.
        gradient_accumulation_steps: Number of micro-batches accumulated per optimizer step.
        apply_gradient_accumulation: Whether to include gradient accumulation in the sense.

    Returns:
        A float32 single-element tensor containing the loss sense.
    """
    if gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps must be greater than or equal to 1, "
            f"but got {gradient_accumulation_steps}."
        )

    sense = 1.0
    if enable_parallel:
        num_devices = get_world_size()
        pipeline_parallel = int(parallelism.pipeline_parallel)
        sense /= num_devices // pipeline_parallel

    if apply_gradient_accumulation:
        sense /= gradient_accumulation_steps

    logger.debug(
        "Got loss sense(%s), enable_parallel=%s, gradient_accumulation_steps=%s, "
        "apply_gradient_accumulation=%s",
        sense,
        enable_parallel,
        gradient_accumulation_steps,
        apply_gradient_accumulation,
    )
    return Tensor([sense], dtype=mstype.float32)
