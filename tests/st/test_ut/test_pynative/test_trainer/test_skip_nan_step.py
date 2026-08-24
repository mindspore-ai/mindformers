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
"""Test NaN/Inf step skipping of the pynative trainer."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor

from mindformers.pynative.trainer import trainer as trainer_module
from mindformers.pynative.trainer.trainer import Trainer
from mindformers.pynative.trainer.train_state import TrainerState


def _build_trainer(use_skip_step_on_nan=True, max_consecutive_skipped_steps=0):
    """Build the minimum Trainer state needed by the skip-step logic."""
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(training=SimpleNamespace(max_norm=1.0))
    trainer.use_skip_step_on_nan = use_skip_step_on_nan
    trainer.max_consecutive_skipped_steps = max_consecutive_skipped_steps
    trainer._consecutive_skipped_steps = 0
    trainer.state = TrainerState()
    trainer.enable_parallel = False
    trainer.optimizer = Mock()
    trainer.optimizer.parameters = []
    trainer.model = []
    return trainer


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("norm_value", [float("nan"), float("inf"), float("-inf")])
def test_skip_step_on_non_finite_norm(norm_value):
    """Non-finite global norms are skipped and counted when the switch is on."""
    trainer = _build_trainer()

    assert trainer._should_skip_optimizer_step(Tensor(norm_value, ms.float32)) is True
    assert trainer._consecutive_skipped_steps == 1
    assert trainer.state.skipped_steps == 1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_finite_norm_resets_consecutive_counter():
    """A finite step is never skipped and clears the consecutive skip counter."""
    trainer = _build_trainer()

    assert trainer._should_skip_optimizer_step(Tensor(float("nan"), ms.float32)) is True
    assert trainer._should_skip_optimizer_step(Tensor(1.5, ms.float32)) is False
    assert trainer._consecutive_skipped_steps == 0
    assert trainer.state.skipped_steps == 1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_warning_reports_one_based_step(monkeypatch):
    """The warning must name the step as the loss callback prints it (1-based)."""
    trainer = _build_trainer()
    trainer.state.global_step = 3  # the 4th step, still not incremented at update time
    warning = Mock()
    monkeypatch.setattr(trainer_module.logger, "warning", warning)

    assert trainer._should_skip_optimizer_step(Tensor(float("nan"), ms.float32)) is True

    # args: (fmt, norm_value, step, consecutive, total)
    assert warning.call_args.args[2] == 4


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_switch_off_keeps_legacy_behavior():
    """With the switch off a NaN norm still goes through the optimizer."""
    trainer = _build_trainer(use_skip_step_on_nan=False)

    assert trainer._should_skip_optimizer_step(Tensor(float("nan"), ms.float32)) is False
    assert trainer.state.skipped_steps == 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_max_consecutive_skipped_steps_guard():
    """The guard raises once the configured number of consecutive skips is reached."""
    trainer = _build_trainer(max_consecutive_skipped_steps=2)

    assert trainer._should_skip_optimizer_step(Tensor(float("nan"), ms.float32)) is True
    with pytest.raises(RuntimeError, match="consecutive steps"):
        trainer._should_skip_optimizer_step(Tensor(float("nan"), ms.float32))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    "norm_value,expect_optimizer_called",
    [(float("nan"), False), (2.0, True)],
)
def test_optimizer_update_skips_optimizer_only(monkeypatch, norm_value, expect_optimizer_called):
    """_optimizer_update skips the optimizer on NaN but always zeroes gradients."""
    trainer = _build_trainer()
    grads = (Tensor(np.ones((2,)), ms.float32),)
    monkeypatch.setattr(
        trainer_module,
        "_calculate_global_grad_norm",
        lambda *args, **kwargs: (Tensor(norm_value, ms.float32), grads),
    )
    module = Mock()
    trainer.model = [module]

    global_norm = trainer._optimizer_update()

    assert trainer.optimizer.called is expect_optimizer_called
    module.zero_grad.assert_called_once()
    assert np.isnan(global_norm.asnumpy()) == np.isnan(norm_value)
