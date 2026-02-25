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

from hyper_parallel import DeviceMesh, shard_module
from hyper_parallel.core.placement_types import Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan
from mindspore import nn
from mindformers.tools import logger


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
    for name, param in module.parameters_and_names():
        if name in parameter_shard_plan or param.requires_grad is False:
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
