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

"""Parallelize a module in pynative mode."""

from typing import Optional, Union
import warnings
from mindspore import nn
from hyper_parallel import DeviceMesh

from mindformers.pynative.distributed.style import ParallelStyle


def parallelize_module(
    module: nn.Cell,
    device_mesh: Optional[DeviceMesh] = None,
    parallelize_plan: Optional[Union[ParallelStyle, dict[str, ParallelStyle]]] = None,
) -> nn.Cell:
    """
    Parallelize a module in pynative mode.

    Args:
        module (:class:`nn.Cell`):
            The module to be parallelized.
        device_mesh (:class:`DeviceMesh`):
            The device mesh to parallelize the module on.
        parallelize_plan (:class:`ParallelStyle` or dict[str, ParallelStyle]):
            The parallelization plan to apply to the module.

    Returns:
        :class:`nn.Cell`: The parallelized module.
    """
    if parallelize_plan is None:
        warnings.warn(
            "No parallelize_plan is provided and auto-parallel is not supported "
            "at the moment, so this parallelize_module call will do nothing."
        )
        return module

    if isinstance(parallelize_plan, ParallelStyle):
        # pylint: disable=W0212
        return parallelize_plan._apply(module, device_mesh)
    if isinstance(parallelize_plan, dict):
        module_map = dict(module.cells_and_names())
        for module_path, parallelize_style in parallelize_plan.items():
            if not module_path:
                raise ValueError(
                    "Expect module path to be non-empty, but got empty string!"
                )
            if module_path in module_map:
                parallelize_module(
                    module_map[module_path],
                    device_mesh,
                    parallelize_style
                )
            else:
                raise ValueError(
                    f"Module path {module_path} not found in the module!"
                )
        return module
    raise TypeError(
        "Expect Union[ParallelStyle, Dict[str, ParallelStyle]] for"
        f" parallelize_plan, {type(parallelize_plan)} found!"
    )
