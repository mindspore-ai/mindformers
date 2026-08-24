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

import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, Parameter

from mindformers.pynative.trainer import trainer as trainer_module
from mindformers.pynative.trainer import utils as trainer_utils
from mindformers.pynative.trainer.trainer import Trainer
from mindformers.pynative.trainer.train_state import TrainerState
from mindformers.pynative.callback.checkpoint_callback import CheckpointCallback


def _build_trainer(use_skip_step_on_nan=True, max_consecutive_skipped_steps=0):
    """Build the minimum Trainer state needed by the skip-step logic."""
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(training=SimpleNamespace(max_norm=1.0))
    trainer.use_skip_step_on_nan = use_skip_step_on_nan
    trainer.max_consecutive_skipped_steps = max_consecutive_skipped_steps
    trainer.state = TrainerState()
    trainer.enable_parallel = False
    # Off by default: _optimizer_update only walks the parameters for the extra
    # zero-grad pass when DSA per-layer indexer backward is on.
    trainer._dsa_indexer_layerwise_backward = False
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

    assert trainer._should_skip_optimizer_step(Tensor(norm_value, ms.float32), False) is True
    assert trainer.state.consecutive_skipped_steps == 1
    assert trainer.state.skipped_steps == 1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_finite_norm_resets_consecutive_counter():
    """A finite step is never skipped and clears the consecutive skip counter."""
    trainer = _build_trainer()

    assert trainer._should_skip_optimizer_step(Tensor(float("nan"), ms.float32), False) is True
    assert trainer._should_skip_optimizer_step(Tensor(1.5, ms.float32), True) is False
    assert trainer.state.consecutive_skipped_steps == 0
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

    assert trainer._should_skip_optimizer_step(Tensor(float("nan"), ms.float32), False) is True

    # args: (fmt, step, norm_value, consecutive, total)
    assert warning.call_args.args[1] == 4


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_switch_off_keeps_legacy_behavior():
    """With the switch off a NaN norm still goes through the optimizer."""
    trainer = _build_trainer(use_skip_step_on_nan=False)

    assert trainer._should_skip_optimizer_step(Tensor(float("nan"), ms.float32), False) is False
    assert trainer.state.skipped_steps == 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_max_consecutive_skipped_steps_guard():
    """The guard raises once the configured number of consecutive skips is reached."""
    trainer = _build_trainer(max_consecutive_skipped_steps=2)

    assert trainer._should_skip_optimizer_step(Tensor(float("nan"), ms.float32), False) is True
    with pytest.raises(RuntimeError, match="consecutive steps"):
        trainer._should_skip_optimizer_step(Tensor(float("nan"), ms.float32), False)


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
        lambda *args, **kwargs: (
            Tensor(norm_value, ms.float32), grads, math.isfinite(norm_value),
        ),
    )
    module = Mock()
    trainer.model = [module]

    global_norm = trainer._optimizer_update()

    assert trainer.optimizer.called is expect_optimizer_called
    module.zero_grad.assert_called_once()
    assert np.isnan(global_norm.asnumpy()) == np.isnan(norm_value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_create_common_info_persists_skip_counters():
    """The checkpoint callback must write both skip counters into common.json."""
    callback = CheckpointCallback.__new__(CheckpointCallback)
    state = TrainerState(global_step=7, global_batch_size=8, consumed_samples=56,
                         skipped_steps=5, consecutive_skipped_steps=2)

    common_info = callback._create_common_info(state)

    assert common_info.skipped_steps == 5
    assert common_info.consecutive_skipped_steps == 2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    "common_data,expected",
    [
        pytest.param(
            {"global_step": 2, "global_batch_size": 8, "consumed_samples": 16,
             "skipped_steps": 5, "consecutive_skipped_steps": 2},
            (5, 2),
            id="checkpoint-with-skip-counters",
        ),
        pytest.param(
            {"global_step": 2, "global_batch_size": 8, "consumed_samples": 16},
            (0, 0),
            id="legacy-checkpoint-without-skip-counters",
        ),
    ],
)
def test_resume_restores_skip_counters(monkeypatch, tmp_path, common_data, expected):
    """Resume must restore both counters so the guard keeps counting across a restart."""
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    (checkpoint_path / "common.json").write_text(json.dumps(common_data), encoding="utf-8")

    trainer = _build_trainer()
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(max_norm=1.0),
        checkpoint=SimpleNamespace(no_load_optim=False, load_balanced=False),
    )
    trainer.dynamic_batch_enabled = False
    trainer.global_batch_size = 8
    trainer._base_units = 4
    trainer._consumed_samples = 0
    trainer._resumed = False
    trainer.train_dataset = Mock()
    monkeypatch.setattr(trainer_module, "get_checkpoint_path", lambda path, **kwargs: path)
    monkeypatch.setattr(trainer_module, "is_hf_checkpoint", lambda _: False)
    monkeypatch.setattr(trainer_module, "has_optimizer_ckpt", lambda _: True)
    monkeypatch.setattr(trainer_module, "load_checkpoint", Mock(return_value=set()))

    trainer._load_checkpoint(str(checkpoint_path), object(), Mock())

    assert (trainer.state.skipped_steps, trainer.state.consecutive_skipped_steps) == expected


def _param_with_grad(grad_values):
    """A parameter carrying the given gradient, as the norm helper expects."""
    parameter = Parameter(Tensor(np.ones((len(grad_values),)), ms.float32), name="weight")
    parameter.main_grad = Tensor(np.asarray(grad_values, dtype=np.float32))
    return parameter


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    "loss_value,expected_finite",
    [(1.5, True), (float("nan"), False), (float("inf"), False)],
)
def test_non_finite_loss_marks_step_non_finite(loss_value, expected_finite):
    """A NaN/Inf loss must fail the step even when the gradients are finite."""
    global_norm, _, step_is_finite = trainer_utils._calculate_global_grad_norm(
        [_param_with_grad([3.0, 4.0])],
        enable_parallel=False,
        max_norm=1.0e9,
        step_loss=Tensor(loss_value, ms.float32),
        return_finite=True,
    )

    assert step_is_finite is expected_finite
    np.testing.assert_allclose(global_norm.asnumpy(), 5.0, rtol=1.0e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_missing_gradient_fails_loudly_instead_of_shifting_the_tuple():
    """A parameter without a gradient must stop the step, not be quietly dropped.

    The returned tuple is aligned to ``parameters`` **by position**: Muon indexes it with
    ``enumerate(self.param_name_tuple)`` and AdamW zips it against ``self.fp32_params``.
    Dropping one entry shifts every later parameter's gradient by one, and ``zip`` truncates
    without a word -- a weight would then be updated with another weight's gradient.
    """
    good = _param_with_grad([3.0, 4.0])
    bad = Parameter(Tensor(np.ones((2,)), ms.float32), name="no_grad_weight")
    # What a real no-gradient parameter looks like: ``main_grad`` was never allocated and the
    # backward left ``grad`` at None. ``get_grad`` reads both, and a bare Parameter has
    # neither attribute at all, so set them explicitly rather than testing a shape that
    # cannot occur in a run.
    bad.main_grad = None
    bad.grad = None

    with pytest.raises(ValueError, match="no_grad_weight"):
        trainer_utils._calculate_global_grad_norm(
            [good, bad],
            enable_parallel=False,
            max_norm=1.0e9,
        )

    # The count in the message must describe the input, not a re-walked iterator: the loop
    # above has already consumed a generator by the time the message is built.
    with pytest.raises(ValueError, match="1 of 2 optimizer parameters"):
        trainer_utils._calculate_global_grad_norm(
            iter([good, bad]),
            enable_parallel=False,
            max_norm=1.0e9,
        )

    # No parameters at all is not the same thing and must stay allowed.
    norm, _ = trainer_utils._calculate_global_grad_norm(
        [], enable_parallel=False, max_norm=1.0e9)
    np.testing.assert_allclose(norm.asnumpy(), 0.0)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_non_finite_grad_marks_step_non_finite():
    """A NaN gradient fails the step even when the loss is finite."""
    _, _, step_is_finite = trainer_utils._calculate_global_grad_norm(
        [_param_with_grad([float("nan"), 1.0])],
        enable_parallel=False,
        max_norm=1.0e9,
        step_loss=Tensor(1.5, ms.float32),
        return_finite=True,
    )

    assert step_is_finite is False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_return_finite_defaults_to_two_tuple():
    """Existing callers keep the two-value contract."""
    result = trainer_utils._calculate_global_grad_norm(
        [_param_with_grad([3.0, 4.0])], enable_parallel=False, max_norm=1.0e9
    )

    assert len(result) == 2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_empty_parameter_stage_still_checks_loss():
    """A stage without optimizer parameters must still participate in the verdict."""
    norm, grads, finite = trainer_utils._calculate_global_grad_norm(
        [], step_loss=Tensor(float("nan"), ms.float32), return_finite=True)
    assert float(norm) == 0.0 and grads == () and not finite


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_optimizer_rejects_nan_loss_with_finite_gradients():
    """The actual norm helper must reject a bad loss without clipping finite gradients."""
    trainer = _build_trainer()
    parameter = _param_with_grad([3.0, 4.0])
    trainer.optimizer.parameters = [parameter]
    model = Mock()
    trainer.model = [model]
    trainer._optimizer_update(loss=Tensor(float("nan"), ms.float32))
    trainer.optimizer.assert_not_called()
    model.zero_grad.assert_called_once()
    np.testing.assert_array_equal(parameter.main_grad.asnumpy(), [3.0, 4.0])
    assert trainer.state.skipped_steps == 1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("per_token,has_last", [(False, False), (False, True), (True, False), (True, True)])
def test_pipeline_forwards_loss_to_verdict(monkeypatch, per_token, has_last):
    """Both pipeline loss modes forward their loss to the optimizer's finite check."""
    trainer = _build_trainer()
    trainer.num_accumulation_steps = 1
    trainer.calculate_per_token_loss = per_token
    trainer.has_last = has_last
    trainer.metric_reduce_group = None
    trainer.metric_reduce_group_size = 1
    numerator = Tensor(float("nan"), ms.float32)
    denominator = Tensor(8.0, ms.float32)
    loss = (numerator, denominator) if per_token else (numerator if has_last else None)
    trainer._collect_micro_batches = Mock(return_value=[])
    trainer.compute_pp_loss = Mock(return_value=loss)
    trainer._finalize_per_token_grads = Mock(return_value=(numerator, denominator))
    trainer._optimizer_update = Mock(return_value=Tensor(1.0, ms.float32))
    monkeypatch.setattr(trainer_module, "all_reduce", Mock())
    trainer.training_pp_step()
    assert trainer._optimizer_update.call_args.kwargs["loss"] is (numerator if per_token else loss)
