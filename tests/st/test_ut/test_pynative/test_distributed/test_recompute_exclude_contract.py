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
"""Contract tests for recompute exclude_op configuration."""
from collections import defaultdict
import importlib
import unittest

import numpy as np
import pytest
import mindspore as ms
from mindspore import mint, nn, ops, Tensor
from hyper_parallel.platform.mindspore.activation_checkpoint import (
    CheckpointExcludeWrapper,
)

import mindformers.pynative.distributed.activation_checkpoint as ac_mod
from mindformers.pynative.config.config import RecomputeCommConfig, RecomputeConfig
from mindformers.pynative.distributed.activation_checkpoint import apply_recompute
from mindformers.pynative.distributed.style import _call_comm_op


_pynative_executor = importlib.import_module("mindspore.graph.api")._pynative_executor


class _CountedCell(nn.Cell):
    def __init__(self, calls, key, scale=1.0):
        super().__init__()
        self.calls = calls
        self.key = key
        self.scale = scale

    def construct(self, value):
        self.calls[self.key] += 1
        return value * self.scale


class _CountedCallable:
    def __init__(self, calls, key, scale):
        self.calls = calls
        self.key = key
        self.scale = scale

    def __call__(self, value):
        self.calls[self.key] += 1
        return value * self.scale


class _ContractRegion(nn.Cell):
    """Region exposing every supported exclude target kind, with comm-op slots."""

    def __init__(self, calls, layer_id):
        super().__init__()
        self.calls = calls
        self.layer_id = layer_id
        self.cell_target = _CountedCell(calls, (layer_id, "cell"), scale=1.1)

        def _replay(value):
            calls[(layer_id, "replay_function")] += 1
            return value * value
        self.replay_function = _replay

        def _fn_target(value):
            calls[(layer_id, "function")] += 1
            return value * 1.02
        self.function_target = _fn_target
        self.operator_target = mint.add
        self.primitive_target = ops.Mul()
        self.callable_target = _CountedCallable(calls, (layer_id, "callable"), scale=1.03)

        def _ag(v):
            calls[(layer_id, "allgather")] += 1
            return v * 1.01

        def _rs(v):
            calls[(layer_id, "reducescatter")] += 1
            return v * 0.99

        self._comm_ops = {"input.allgather": {"fn": _ag, "domain": "tp"},
                          "output.reducescatter": {"fn": _rs, "domain": "tp"}}
        self.region_tail = _CountedCell(calls, (layer_id, "region_tail"), scale=1.05)

    def construct(self, value):
        """Run every target kind in order, starting and ending with comm ops."""
        self.calls[(self.layer_id, "region")] += 1
        value = _call_comm_op(self, "input.allgather", value)
        value = _call_comm_op(self, "output.reducescatter", value)
        value = self.cell_target(value)
        value = self.replay_function(value)
        value = self.function_target(value)
        value = self.operator_target(value, 0.125)
        value = self.primitive_target(value, 1.01)
        value = self.callable_target(value)
        return self.region_tail(value)


class _ContractLayer(nn.Cell):
    def __init__(self, calls, layer_id):
        super().__init__()
        self.region = _ContractRegion(calls, layer_id)
        self.layer_tail = _CountedCell(calls, (layer_id, "layer_tail"), scale=0.9)

    def construct(self, value):
        return self.layer_tail(self.region(value))


class _ContractModel(nn.Cell):
    """Stack of contract layers sharing one call counter."""

    def __init__(self, num_layers):
        super().__init__()
        self.calls = defaultdict(int)
        self.layers = nn.CellList(
            [_ContractLayer(self.calls, i) for i in range(num_layers)])
        self.layer_start = 0
        self.layer_end = num_layers - 1
        self.config = type("Config", (), {"num_layers": num_layers})()

    def construct(self, value):
        for layer in self.layers:
            value = layer(value)
        return value.sum()


def _three_way_config(full_layers, select_layers, exclude_layers):
    """Build the production-style full + select + exclude configuration."""
    return (
        RecomputeConfig(
            mode="select",
            full_recompute_layer=full_layers,
            select_module={"region": select_layers},
            exclude_op={
                ".*cell_target": exclude_layers,
                ".*function_target": exclude_layers,
                ".*operator_target": exclude_layers,
                ".*primitive_target": exclude_layers,
                ".*callable_target": exclude_layers,
                ".*allgather": exclude_layers,
                ".*reducescatter": exclude_layers,
            },
        ),
        RecomputeCommConfig(enable=False),
    )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestRecomputeExcludeContract(unittest.TestCase):
    """Contract: whitelist discovery, replay counts and gradient parity for exclude_op."""

    @classmethod
    def setUpClass(cls):
        ms.set_context(mode=ms.PYNATIVE_MODE)

    def setUp(self):
        ac_mod._config_list = {}
        _pynative_executor.clear_res()

    def tearDown(self):
        ac_mod._config_list = {}
        _pynative_executor.clear_res()

    def _assert_gradients_match(self, model, reference):
        value = Tensor(np.linspace(-0.8, 0.7, 8, dtype=np.float32))
        a_loss, a_grad = ms.value_and_grad(model, grad_position=0)(value)
        e_loss, e_grad = ms.value_and_grad(reference, grad_position=0)(value)
        np.testing.assert_allclose(a_loss.asnumpy(), e_loss.asnumpy(), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(a_grad.asnumpy(), e_grad.asnumpy(), atol=1e-6, rtol=1e-6)

    def test_every_supported_target_kind_is_discovered_and_wrapped(self):
        """Whitelist discovery covers every target kind; comm slots get wrapped."""
        model = _ContractModel(num_layers=1)
        whitelist = ac_mod._get_modules_and_ops_list(model)[0]
        for key in ("region.cell_target", "region.function_target", "region.operator_target",
                     "region.primitive_target", "region.callable_target",
                     "region.input.allgather", "region.output.reducescatter"):
            self.assertIn(key, whitelist)
        rc, rc_comm = _three_way_config(["0"], ["0"], ["0"])
        apply_recompute(model, rc, rc_comm)
        layer = model.layers[0]
        self.assertIsInstance(layer.region.cell_target, CheckpointExcludeWrapper)
        self.assertIsInstance(
            layer.region._comm_ops["input.allgather"]["fn"], CheckpointExcludeWrapper)

    def test_recompute_comm_and_exclude_runtime_replay_counts(self):
        """Comm recompute and exclude replay counts coexist with gradient parity."""
        model = _ContractModel(num_layers=1)
        reference = _ContractModel(num_layers=1)
        rc = RecomputeConfig(mode="select", select_module={"region": ["0"]},
                             exclude_op={".*reducescatter": ["0"]})
        rc_comm = RecomputeCommConfig(enable=True, select_module={".*allgather": ["0"]})
        apply_recompute(model, rc, rc_comm)
        self._assert_gradients_match(model, reference)
        self.assertEqual(model.calls[(0, "region")], 2)
        self.assertEqual(model.calls[(0, "replay_function")], 2)
        self.assertEqual(model.calls[(0, "allgather")], 2)
        self.assertEqual(model.calls[(0, "reducescatter")], 1)

    def test_full_select_and_exclude_runtime_replay_counts(self):
        """Full, select and exclude layers each replay exactly the expected targets."""
        model = _ContractModel(num_layers=3)
        reference = _ContractModel(num_layers=3)
        rc, rc_comm = _three_way_config(["0"], ["1"], ["0-1"])
        apply_recompute(model, rc, rc_comm)
        self._assert_gradients_match(model, reference)
        for layer_id in (0, 1):
            for target in ("cell", "function", "callable", "allgather", "reducescatter"):
                self.assertEqual(model.calls[(layer_id, target)], 1)
            self.assertEqual(model.calls[(layer_id, "region")], 2)
            self.assertEqual(model.calls[(layer_id, "replay_function")], 2)
        for target in ("cell", "function", "callable", "replay_function",
                       "allgather", "reducescatter", "region", "region_tail", "layer_tail"):
            self.assertEqual(model.calls[(2, target)], 1)


if __name__ == "__main__":
    unittest.main()
