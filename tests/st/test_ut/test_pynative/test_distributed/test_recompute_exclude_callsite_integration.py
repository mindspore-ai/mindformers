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
"""Integration tests for communication callsites used by ``exclude_op``."""
import unittest

import pytest
import mindspore as ms
from mindspore import nn, Tensor
from hyper_parallel.platform.mindspore.activation_checkpoint import (
    CheckpointExcludeWrapper,
)

from mindformers.pynative.config.config import RecomputeCommConfig, RecomputeConfig
from mindformers.pynative.distributed.activation_checkpoint import apply_recompute
from mindformers.pynative.distributed.ep_overlap import OverlapExpertParallel
from mindformers.pynative.distributed.style import (
    AllGather,
    PrepareModuleInput,
    PrepareModuleOutput,
)


class _Echo(nn.Cell):
    def construct(self, *values):
        return values


class _CheckpointModel(nn.Cell):
    def __init__(self, layer):
        super().__init__()
        self.layers = nn.CellList([layer])
        self.layer_start = 0
        self.layer_end = 0

    def construct(self, value):
        return self.layers[0](value).sum()


def _recording_transform(trace, label):
    def transform(value):
        trace.append(label)
        return value
    return transform


def _make_input_style_and_bound(trace):
    style = PrepareModuleInput(input_transforms=(AllGather(0), AllGather(0)))
    bound = (_recording_transform(trace, "input-0"),
             _recording_transform(trace, "input-1"))
    style._bind_input_transforms = lambda _m: (bound, {})
    return style, bound


def _make_output_style_and_bound(trace):
    style = PrepareModuleOutput(output_transforms=(AllGather(0), AllGather(0)))
    bound = (_recording_transform(trace, "output-0"),
             _recording_transform(trace, "output-1"))
    style._bind_output_transforms = lambda _m: bound
    return style, bound


def _make_region_layer(is_input):
    """Build a layer whose region is applied with two bound allgathers."""
    if is_input:
        class Region(nn.Cell):
            def construct(self, left, right):
                return left * right
        class Layer(nn.Cell):
            def __init__(self):
                super().__init__()
                self.region = Region()
            def construct(self, value):
                return self.region(value, value + 0.25)
    else:
        class Region(nn.Cell):
            def construct(self, value):
                return value + 0.25, value * 1.25
        class Layer(nn.Cell):
            def __init__(self):
                super().__init__()
                self.region = Region()
            def construct(self, value):
                left, right = self.region(value)
                return left * right
    return Layer()


# ---- TP local-transform AllGather slots ----------------------------------

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestLocalTransformCommCallsites(unittest.TestCase):
    """Positional allgather slots registered by input/output transforms."""

    def setUp(self):
        self.value = Tensor([1.0], ms.float32)

    def test_multiple_allgathers_keep_positional_identity(self):
        """Each bound allgather keeps its own positional slot and label."""
        for maker, labels, is_input in (
                (_make_input_style_and_bound, ["input-0", "input-1"], True),
                (_make_output_style_and_bound, ["output-0", "output-1"], False)):
            trace = []
            style, bound = maker(trace)
            cell = style._apply(_Echo(), object())
            if is_input:
                style._prepare_bound_inputs((self.value, self.value), bound, cell=cell)
            else:
                style._prepare_bound_outputs(
                    (self.value, self.value), bound, cell=cell,
                    raw_transforms=style.output_transforms)
            self.assertEqual(trace, labels)

    def test_exclude_config_selects_one_allgather_slot(self):
        """An exact positional slot pattern excludes only the matching allgather."""
        for maker, direction, labels in (
                (_make_input_style_and_bound, "input", ["input-0", "input-1"]),
                (_make_output_style_and_bound, "output", ["output-0", "output-1"])):
            trace = []
            layer = _make_region_layer(direction == "input")
            style, _ = maker(trace)
            style._apply(layer.region, object())
            model = _CheckpointModel(layer)
            rc = RecomputeConfig(
                mode="full", full_recompute_layer=["0"],
                exclude_op={rf"region\.{direction}\.0\.allgather": ["0"]})
            apply_recompute(model, rc, RecomputeCommConfig(enable=False))
            slots = model.layers[0].region._comm_ops
            self.assertIsInstance(slots[f"{direction}.0.allgather"]["fn"], CheckpointExcludeWrapper)
            self.assertNotIsInstance(slots[f"{direction}.1.allgather"]["fn"], CheckpointExcludeWrapper)
            value = Tensor([0.25, 0.5, 0.75], ms.float32)
            ms.value_and_grad(model, grad_position=0)(value)
            self.assertEqual(trace.count(labels[0]), 1)
            self.assertEqual(trace.count(labels[1]), 2)


# ---- EP overlap AllToAll slots -------------------------------------------

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestOverlapExpertParallelCommCallsites(unittest.TestCase):
    """Overlap EP alltoall callsites route through the comm-op registry."""

    @staticmethod
    def _cell_with_registry(trace):
        """Build a cell with input/output alltoallsingle registry entries."""
        class Cell:
            pass
        cell = Cell()
        cell._comm_ops = {
            "input.alltoallsingle": {
                "fn": lambda *a, **kw: trace.append("input.registry") or "input.registry.result",
                "domain": "ep"},
            "output.alltoallsingle": {
                "fn": lambda *a, **kw: trace.append("output.registry") or "output.registry.result",
                "domain": "ep"},
        }
        return cell

    def test_overlap_a2a_uses_registry(self):
        for method, label in ((OverlapExpertParallel._main_a2a, "input.registry"),
                                (OverlapExpertParallel._combine_a2a, "output.registry")):
            trace = []
            strategy = OverlapExpertParallel(coordinator=object())
            strategy._async_a2a = lambda *_a, _t=trace: _t.append("raw") or "raw.result"
            cell = self._cell_with_registry(trace)
            result = method(strategy, "input", [1], [1], block_size=1, cell=cell)
            self.assertEqual(result, f"{label}.result")
            self.assertIn(label, trace)


if __name__ == "__main__":
    unittest.main()
