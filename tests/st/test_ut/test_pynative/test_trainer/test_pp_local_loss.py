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
"""Regression coverage for PP loss reporting and local gradient norm monitoring."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, nn, ops
from hyper_parallel.core.dtensor.placement_types import Replicate

from mindformers.pynative.tools import monitor as monitor_module
from mindformers.pynative.tools.monitor import MonitorGroup, TrainStateMonitor
from mindformers.pynative.trainer.trainer import Trainer


def _config(enabled=True, local_norm=False):
    """Enable only the train-state metrics exercised by these tests."""
    return SimpleNamespace(monitor=SimpleNamespace(train_state=SimpleNamespace(
        local_loss=enabled, local_norm=local_norm, device_loss=False, device_norm=False)))


def _trainer(outputs, enabled=True, per_token=False):
    """Use a real monitor with schedule outputs supplied at the PP boundary."""
    trainer = Trainer.__new__(Trainer)
    trainer.model = []
    trainer.monitor = MonitorGroup(_config(enabled))
    trainer.schedule = SimpleNamespace(run=Mock(return_value=outputs))
    trainer.calculate_per_token_loss = per_token
    trainer.compute_loss_func = None
    return trainer


@pytest.mark.parametrize("output_format", ["tuple", "list", "dict", "tensor"])
def test_pp_records_each_micro_loss_and_preserves_mean(monkeypatch, output_format):
    """Micro-step numbering and printed values match the non-PP monitor contract."""
    losses = [Tensor(2.0), Tensor(4.0)]
    formats = {"tuple": lambda x: (x,), "list": lambda x: [x],
               "dict": lambda x: {"loss": x}, "tensor": lambda x: x}
    trainer = _trainer([formats[output_format](x) for x in losses])
    messages = []
    monkeypatch.setattr(monitor_module.logger, "info", messages.append)

    loss = trainer.compute_pp_loss({})
    trainer.monitor.flush(7)

    assert float(loss.asnumpy()) == 3.0
    assert [json.loads(line) for line in messages] == [
        {"step": 7, "micro_step": 1, "local_loss": 2.0},
        {"step": 7, "micro_step": 2, "local_loss": 4.0},
    ]


def test_pp_per_token_reports_mean_but_preserves_sums(monkeypatch):
    """Reporting must not replace the token sums used to normalize gradients."""
    trainer = _trainer([(Tensor(12.0), Tensor(3.0)), (Tensor(28.0), Tensor(7.0))], per_token=True)
    messages = []
    monkeypatch.setattr(monitor_module.logger, "info", messages.append)

    numerator, denominator = trainer.compute_pp_loss({})
    trainer.monitor.flush(9)

    assert float(numerator.asnumpy()) == 40.0
    assert float(denominator.asnumpy()) == 10.0
    assert [json.loads(line) for line in messages] == [
        {"step": 9, "micro_step": 1, "local_loss": 4.0},
        {"step": 9, "micro_step": 2, "local_loss": 4.0},
    ]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
def test_pp_non_last_stage_emits_no_loss(monkeypatch, enabled, per_token):
    """A stage with no loss outputs must neither divide nor print anything."""
    trainer = _trainer([], enabled, per_token)
    info = Mock()
    monkeypatch.setattr(monitor_module.logger, "info", info)
    assert trainer.compute_pp_loss({}) is None
    trainer.monitor.flush(1)
    info.assert_not_called()


@pytest.mark.parametrize("per_token", [False, True])
def test_pp_disabled_monitor_keeps_aggregation_silent(monkeypatch, per_token):
    """Disabled local_loss does not change aggregation or produce records."""
    outputs = [(Tensor(12.0), Tensor(3.0))] if per_token else [(Tensor(2.0),)]
    trainer = _trainer(outputs, enabled=False, per_token=per_token)
    info = Mock()
    monkeypatch.setattr(monitor_module.logger, "info", info)
    result = trainer.compute_pp_loss({})
    if per_token:
        assert [float(x.asnumpy()) for x in result] == [12.0, 3.0]
    else:
        assert float(result.asnumpy()) == 2.0
    trainer.monitor.flush(1)
    info.assert_not_called()


@pytest.mark.parametrize("custom", [False, True])
def test_pp_custom_and_top_level_dict_outputs_keep_contract(custom):
    """Only schedule micro-batch lists supply automatic per-micro loss records."""
    outputs = {"loss": Tensor(5.0)}
    trainer = _trainer(outputs)
    trainer.monitor = Mock()
    if custom:
        trainer.compute_loss_func = Mock(return_value=Tensor(6.0))
    result = trainer.compute_pp_loss({"labels": "labels"})
    assert float(result.asnumpy()) == (6.0 if custom else 5.0)
    trainer.monitor.record.assert_not_called()
    if custom:
        trainer.compute_loss_func.assert_called_once_with(outputs, "labels")


def test_pp_local_loss_record_includes_model_context():
    """Context remains compatible with all MonitorGroup consumers."""
    trainer = _trainer([(Tensor(2.0),)])
    trainer.monitor = Mock()
    trainer.compute_pp_loss({})
    call = trainer.monitor.record.call_args
    assert call.args[0] == "local_loss"
    assert call.kwargs["context"] == {"micro_step": 0, "model": trainer.model}


@pytest.mark.parametrize("values", [[[3.0, 4.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
@pytest.mark.skipif(ms.get_context("device_target") != "Ascend",
                    reason="mint.norm requires the Ascend LinalgVectorNorm kernel")
def test_local_grad_norm_preserves_scalar_interface(values):
    """The fused norm retains the branch's Python scalar return contract."""
    grad = Tensor(values, ms.float32)
    norm = TrainStateMonitor._compute_grad_norm(grad)
    assert isinstance(norm, float)
    assert norm == pytest.approx(float(np.linalg.norm(values)))


@pytest.mark.skipif(ms.get_context("device_target") != "Ascend",
                    reason="mint.norm requires the Ascend LinalgVectorNorm kernel")
def test_local_grad_norm_preserves_dtensor_and_pp_scaling():
    """Placement replicas and PP-shared copies retain the original divisor."""
    grad = SimpleNamespace(
        to_local=lambda: Tensor([3.0, 4.0], ms.float32),
        _pp_replica_count=2, placements=(Replicate(),),
        device_mesh=SimpleNamespace(size=lambda dim: 4),
    )
    assert TrainStateMonitor._compute_grad_norm(grad) == pytest.approx(5.0 / 8.0)


@pytest.mark.skipif(ms.get_context("device_target") != "Ascend",
                    reason="mint.norm requires the Ascend LinalgVectorNorm kernel")
def test_local_norm_real_backward_hook_and_flush(monkeypatch):
    """Exercise backward -> mint.norm -> local_norm record -> log output."""
    class Net(nn.Cell):
        """A parameter with a known backward gradient norm."""
        def __init__(self):
            super().__init__()
            self.weight = ms.Parameter(Tensor([1.0, 1.0], ms.float32), name="weight")

        def construct(self, x):  # pylint: disable=arguments-differ
            return ops.sum(self.weight * x)

    net = Net()
    monitor = TrainStateMonitor(_config(local_norm=True), model=net)
    messages = []
    monkeypatch.setattr(monitor_module.logger, "info", messages.append)
    ms.value_and_grad(net, None, ms.ParameterTuple(net.trainable_params()))(Tensor([3.0, 4.0]))
    assert monitor._hook_local_norms["weight"] == pytest.approx([5.0])
    monitor.flush(2)
    assert any('"local_norm": weight:   5.000000' in line for line in messages)
    monitor._remove_grad_hooks()
