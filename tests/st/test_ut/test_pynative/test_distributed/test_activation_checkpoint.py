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
"""Test recompute functional scenarios."""
import pytest
from mindspore import nn
from hyper_parallel.platform.mindspore.activation_checkpoint import CheckpointWrapper, SwapWrapper

from mindformers.pynative.config.config import (
    RecomputeCommConfig,
    RecomputeConfig,
    SwapConfig,
)
from mindformers.pynative.distributed.activation_checkpoint import apply_recompute, apply_swap


class MockAttention(nn.Cell):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Cell()
        self.proj = nn.Cell()

    def construct(self, x):
        return x


class MockMLP(nn.Cell):
    def __init__(self):
        super().__init__()
        self.gate = nn.Cell()
        self.proj = nn.Cell()

    def construct(self, x):
        return x


class MockTransformerLayer(nn.Cell):
    def __init__(self):
        super().__init__()
        self.attention = MockAttention()
        self.mlp = MockMLP()

    def construct(self, x):
        return x


class MockModel(nn.Cell):
    def __init__(self, num_layers=2):
        super().__init__()
        self.layers = nn.CellList([MockTransformerLayer() for _ in range(num_layers)])
        self.config = type("Config", (), {"num_layers": num_layers})()

    def construct(self, x):
        return x


def _make_recompute_config(mode="None", full_recompute_layer=None,
                 select_module=None, comm_enable=False, comm_select_module=None):
    """Build recompute and recompute_comm configs."""
    rc = RecomputeConfig(mode=mode, full_recompute_layer=full_recompute_layer,
                         select_module=select_module)
    rc_comm = RecomputeCommConfig(enable=comm_enable, select_module=comm_select_module)
    return rc, rc_comm


# ======================== Recompute ========================


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestFullRecompute:
    """Test full recompute: entire layers are wrapped with CheckpointWrapper."""

    def test_full_recompute_single_layer(self):
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(mode="full", full_recompute_layer=["0"])
        apply_recompute(model, rc, rc_comm, num_layers=2)
        assert isinstance(model.layers[0], CheckpointWrapper)
        assert not isinstance(model.layers[1], CheckpointWrapper)

    def test_full_recompute_all_layers(self):
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(mode="full", full_recompute_layer=["0-1"])
        apply_recompute(model, rc, rc_comm, num_layers=2)
        assert isinstance(model.layers[0], CheckpointWrapper)
        assert isinstance(model.layers[1], CheckpointWrapper)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestSelectRecompute:
    """Test select recompute: specific modules within layers are wrapped."""

    def test_select_by_wildcard(self):
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(mode="select", select_module={".*\\.proj": ["0"]})
        apply_recompute(model, rc, rc_comm, num_layers=2)
        assert isinstance(model.layers[0].attention.proj, CheckpointWrapper)
        assert isinstance(model.layers[0].mlp.proj, CheckpointWrapper)

    def test_select_parent_covers_children(self):
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(mode="select", select_module={"attention": ["0"], "attention.qkv": ["0"]})
        apply_recompute(model, rc, rc_comm, num_layers=2)
        assert isinstance(model.layers[0].attention, CheckpointWrapper)
        assert not isinstance(model.layers[0].attention.qkv, CheckpointWrapper)

    def test_select_multiple_modules(self):
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(mode="select", select_module={"attention": ["0"], "mlp": ["1"]})
        apply_recompute(model, rc, rc_comm, num_layers=2)
        assert isinstance(model.layers[0].attention, CheckpointWrapper)
        assert isinstance(model.layers[1].mlp, CheckpointWrapper)

    # ======================== Swap ========================


def _make_swap_config(enable=True, default_prefetch=1, layer_swap=None, op_swap=None):
    """Build a SwapConfig."""
    return SwapConfig(enable=enable, default_prefetch=default_prefetch,
                      layer_swap=layer_swap, op_swap=op_swap)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestLayerSwap:
    """Test layer swap: entire layers are wrapped with swap_wrapper."""

    def test_layer_swap_single_layer(self):
        model = MockModel(num_layers=3)
        sc = _make_swap_config(layer_swap=[{"layers": ["0"]}])
        apply_swap(model, sc, num_layers=3)
        assert isinstance(model.layers[0], SwapWrapper)
        assert not isinstance(model.layers[1], SwapWrapper)
        assert not isinstance(model.layers[2], SwapWrapper)

    def test_layer_swap_range(self):
        model = MockModel(num_layers=3)
        sc = _make_swap_config(layer_swap=[{"layers": ["0-1"]}])
        apply_swap(model, sc, num_layers=3)
        assert isinstance(model.layers[0], SwapWrapper)
        assert isinstance(model.layers[1], SwapWrapper)
        assert not isinstance(model.layers[2], SwapWrapper)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestOpSwap:
    """Test op swap: specific modules within layers are wrapped."""

    def test_op_swap_by_name(self):
        model = MockModel(num_layers=3)
        sc = _make_swap_config(op_swap=[{"op_name": "attention", "layers": ["0"]}])
        apply_swap(model, sc, num_layers=3)
        assert isinstance(model.layers[0].attention, SwapWrapper)
        assert not isinstance(model.layers[0].mlp, SwapWrapper)

    def test_op_swap_multiple_modules(self):
        model = MockModel(num_layers=3)
        sc = _make_swap_config(op_swap=[
            {"op_name": "attention", "layers": ["0"]},
            {"op_name": "mlp", "layers": ["1"]},
        ])
        apply_swap(model, sc, num_layers=3)
        assert isinstance(model.layers[0].attention, SwapWrapper)
        assert isinstance(model.layers[1].mlp, SwapWrapper)
