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
"""Parallel styles for MC2 fused linear layers."""

from hyper_parallel import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from mindspore import nn

from mindformers.pynative.distributed.style import ColwiseParallel, RowwiseParallel
from mindformers.pynative.layers.linear import Linear
from mindformers.pynative.layers.mc2 import MC2Linear
from mindformers.pynative.distributed.utils import distribute_module

__all__ = ["MC2ColwiseParallel", "MC2RowwiseParallel"]


def _replace_with_mc2_linear(module: nn.Cell, mode: str, device_mesh: DeviceMesh) -> MC2Linear:
    if not isinstance(module, Linear):
        raise NotImplementedError(
            f"MC2 parallel style only supports Linear modules, but got {type(module).__name__}."
        )
    module = MC2Linear.from_linear(module)
    module.configure_mc2(mode, device_mesh.get_group(), device_mesh.size())
    return module


class MC2ColwiseParallel(ColwiseParallel):
    """Column parallelism using fused all-gather and matmul."""

    def __init__(self, *, gather_input: bool = True, gather_output: bool = False):
        super().__init__(gather_input=False, gather_output=gather_output)
        if not gather_input:
            raise ValueError("MC2ColwiseParallel requires a sequence-sharded input.")

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh):
        module = _replace_with_mc2_linear(module, "all_gather", device_mesh)
        return super()._apply(module, device_mesh)


class MC2RowwiseParallel(RowwiseParallel):
    """Row parallelism using fused matmul and reduce-scatter."""

    def __init__(self):
        super().__init__(reduce_mode="reduce_scatter", input_is_parallel=True)

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh):
        module = _replace_with_mc2_linear(module, "reduce_scatter", device_mesh)
        sharding_plan = {"weight": (Shard(1),)}
        if getattr(module, "bias", None) is not None:
            sharding_plan["bias"] = (Replicate(),)
        distribute_module(
            module=module,
            device_mesh=device_mesh,
            parameter_shard_plan=sharding_plan,
            input_fn=None,
            output_fn=None,
        )
        return module
