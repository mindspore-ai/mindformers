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

"""Parallelize a module in pynative mode.

This module provides:
- parallelize_module: Low-level TP parallelization for individual modules.
- parallelize_model: High-level entry point that routes to model-specific
  parallelization functions via a registration dict.
"""

from typing import Any, Callable, Dict, Optional, Type, Union
import warnings

from mindspore import nn
from hyper_parallel import DeviceMesh

from mindformers.pynative.distributed.style import ParallelStyle
from mindformers.tools.logger import logger

__all__ = [
    "parallelize_module",
    "parallelize_model",
    "register_parallelize_fn",
    "register_parallelize",
]


def parallelize_module(
    module: nn.Cell,
    device_mesh: Optional[DeviceMesh] = None,
    parallelize_plan: Optional[Union[ParallelStyle, dict[str, ParallelStyle]]] = None,
) -> nn.Cell:
    """
    Parallelize a module in pynative mode (for Tensor Parallelism).

    Args:
        module: The module to be parallelized.
        device_mesh: The device mesh to parallelize the module on.
        parallelize_plan: The parallelization plan to apply to the module.

    Returns:
        The parallelized module.
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
        module_map = dict(module.name_cells().items())
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


_PARALLELIZE_FN: Dict[Type, Callable] = {}


def register_parallelize_fn(model_cls: Type, fn: Callable) -> None:
    """Register a model-specific parallelization function.

    Each model class registers its own parallelize function, which will be
    called by parallelize_model() when that model type is encountered.

    Args:
        model_cls: The model class (e.g., GPTModel).
        fn: The parallelization function. Must accept:
            (model, parallel_dims, training_config, parallelism_config) -> nn.Cell

    Example:
        >>> from mindformers.pynative.distributed.parallelize import register_parallelize_fn
        >>> register_parallelize_fn(GPTModel, parallelize_gptmodel)
    """
    if model_cls in _PARALLELIZE_FN:
        logger.warning(
            "Overwriting parallelize function for %s", model_cls.__name__
        )
    _PARALLELIZE_FN[model_cls] = fn
    logger.debug("Registered parallelize function for %s", model_cls.__name__)


def register_parallelize(fn: Callable) -> Callable:
    """Class decorator that binds a model class to its parallelization function.

    Args:
        fn: Parallelization function with signature:
            (model, parallel_dims, training_config, parallelism_config) -> nn.Cell

    Returns:
        The decorated class.
    """
    def decorator(cls: Type) -> Type:
        register_parallelize_fn(cls, fn)
        return cls
    return decorator


def parallelize_model(
    model: nn.Cell,
    parallel_dims: Any,
    training_config: Any,
    parallelism_config: Any,
) -> nn.Cell:
    """Route to the correct model-specific parallelization function.

    Args:
        model: The model to parallelize.

    Returns:
        The parallelized model.

    Raises:
        ValueError: If no parallelize function is registered for this model type.
    """
    model_cls = type(model)
    if model_cls in _PARALLELIZE_FN:
        logger.info("Parallelizing %s", model_cls.__name__)
        return _PARALLELIZE_FN[model_cls](
            model, parallel_dims, training_config, parallelism_config
        )
    registered = [c.__name__ for c in _PARALLELIZE_FN]
    raise ValueError(
        f"No parallelize function registered for {model_cls.__name__}. "
        f"Registered model types: {registered}. "
        f"Use register_parallelize() to register one."
    )
