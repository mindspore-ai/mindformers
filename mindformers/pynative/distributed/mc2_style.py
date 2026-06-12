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
from hyper_parallel.core.dtensor.placement_types import Placement, Shard
from mindspore import nn

from mindformers.pynative.distributed.style import ColwiseParallel, RowwiseParallel
from mindformers.pynative.layers.linear import Linear
from mindformers.pynative.layers.mc2 import MC2Linear

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

    def __init__(
        self,
        *,
        input_layouts: Placement | None = None,
        output_layouts: Placement | None = None,
        use_local_output: bool = True,
    ):
        super().__init__(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output,
        )
        if not isinstance(self.input_layouts[0], Shard):
            raise ValueError("MC2ColwiseParallel requires a sharded input layout.")
        self.desired_input_layouts = self.input_layouts

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh):
        module = _replace_with_mc2_linear(module, "all_gather", device_mesh)
        return super()._apply(module, device_mesh)


class MC2RowwiseParallel(RowwiseParallel):
    """Row parallelism using fused matmul and reduce-scatter."""

    def __init__(
        self,
        *,
        input_layouts: Placement | None = None,
        output_layouts: Placement | None = None,
        use_local_output: bool = True,
    ):
        super().__init__(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output,
        )
        if not isinstance(self.output_layouts[0], Shard):
            raise ValueError("MC2RowwiseParallel requires a sharded output layout.")

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh):
        module = _replace_with_mc2_linear(module, "reduce_scatter", device_mesh)
        return super()._apply(module, device_mesh)
