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
"""Tests for MC2 parallel styles."""

import pytest

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard

from mindformers.pynative.distributed.mc2_style import MC2ColwiseParallel, MC2RowwiseParallel
from mindformers.pynative.distributed.style import ColwiseParallel, RowwiseParallel
from mindformers.pynative.layers.linear import Linear
from mindformers.pynative.layers.mc2 import MC2Linear


class _FakeMesh:
    @staticmethod
    def get_group():
        return "tp-group"

    @staticmethod
    def size():
        return 4


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize(
    "style,base_style,mode",
    [
        (MC2ColwiseParallel(input_layouts=Shard(0)), ColwiseParallel, "all_gather"),
        (MC2RowwiseParallel(output_layouts=Shard(0)), RowwiseParallel, "reduce_scatter"),
    ],
)
def test_mc2_style_replaces_linear_in_place(monkeypatch, style, base_style, mode):
    """MC2 styles preserve parameters while replacing the Linear implementation."""
    monkeypatch.setattr(base_style, "_apply", lambda self, module, device_mesh: module)
    linear = Linear(4, 8, "float32", "float32", bias=False)
    weight = linear.weight

    result = style._apply(linear, _FakeMesh())

    assert result is linear
    assert isinstance(result, MC2Linear)
    assert result.weight is weight
    assert result.parameters_dict()["weight"] is weight
    assert result.mc2_mode == mode
    assert result.mc2_group == "tp-group"
    assert result.mc2_world_size == 4


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_mc2_style_requires_sequence_sharding():
    """MC2 styles reject layouts that cannot fuse sequence communication."""
    with pytest.raises(ValueError, match="sharded input"):
        MC2ColwiseParallel(input_layouts=Replicate())
    with pytest.raises(ValueError, match="sharded output"):
        MC2RowwiseParallel(output_layouts=Replicate())
