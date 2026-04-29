# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
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
# This file is derived from TorchTitan and adapted for MindSpore.
# Modifications:
#     - Adapted to MindSpore framework: replaced nn.Module with nn.Cell,
#       model.modules() with model.name_cells().items().
#     - Replaced torch FSDP APIs (FSDPModule, set_gradient_divide_factor)
#       with hyper_parallel HSDPModule APIs (set_reduce_op_type).
#     - Replaced Python 3.10 match/case with if/elif/else for broader
#       Python version compatibility.
# ============================================================================
"""FSDP/HSDP shared utilities for MindFormers PyNative."""

from mindspore import nn

from hyper_parallel.core.fully_shard.api import HSDPModule

__all__ = [
    "get_fsdp_reshard_after_forward_policy",
    "disable_fsdp_gradient_division",
]


def get_fsdp_reshard_after_forward_policy(
    reshard_after_forward_policy: str,
    pp_enabled: bool,
) -> bool:
    """Resolve fsdp_reshard_after_forward policy string to a boolean.

    Args:
        reshard_after_forward_policy: One of "always", "never", or "default".
        pp_enabled: Whether pipeline parallelism is enabled.

    Returns:
        Boolean indicating whether to reshard after forward.

    Raises:
        ValueError: If reshard_after_forward_policy is not one of the valid options.

    Example:
        >>> get_fsdp_reshard_after_forward_policy("default", pp_enabled=False)
        True
        >>> get_fsdp_reshard_after_forward_policy("default", pp_enabled=True)
        False
    """
    if reshard_after_forward_policy == "always":
        return True
    if reshard_after_forward_policy == "never":
        return False
    if reshard_after_forward_policy == "default":
        # For PP, default to no reshard-after-forward to avoid repeated all-gathers per microbatch.
        return not pp_enabled
    raise ValueError(
        f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}. "
        "Valid options are: 'always', 'never', 'default'."
    )


def disable_fsdp_gradient_division(model: nn.Cell) -> None:
    """Set reduce op to "sum" for all HSDP modules, disabling automatic gradient division.

    Call this after fully_shard wrapping is complete if gradient scaling is handled externally.
    """
    for _, submodule in model.cells_and_names():
        if isinstance(submodule, HSDPModule):
            submodule.set_reduce_op_type("sum")
