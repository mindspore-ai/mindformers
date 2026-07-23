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
"""Test checkpoint resume behavior of the pynative trainer."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mindformers.pynative.trainer import trainer as trainer_module
from mindformers.pynative.trainer.trainer import Trainer


def _build_trainer(global_batch_size=8):
    """Build the minimum Trainer state needed by _load_checkpoint."""
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(
        checkpoint=SimpleNamespace(
            no_load_optim=False,
            load_balanced=False,
        )
    )
    trainer.dynamic_batch_enabled = False
    trainer.global_batch_size = global_batch_size
    trainer._base_units = 4
    trainer._consumed_samples = 0
    trainer._resumed = False
    trainer.train_dataset = Mock()
    trainer.state = SimpleNamespace(global_step=0, consumed_samples=0)
    return trainer


def _mock_checkpoint_load(monkeypatch):
    """Mock checkpoint IO while exercising the real Trainer resume logic."""
    monkeypatch.setattr(trainer_module, "get_checkpoint_path", lambda path: path)
    monkeypatch.setattr(trainer_module, "is_hf_checkpoint", lambda _: False)
    monkeypatch.setattr(trainer_module, "has_optimizer_ckpt", lambda _: True)
    load_checkpoint = Mock()
    monkeypatch.setattr(trainer_module, "load_checkpoint", load_checkpoint)
    return load_checkpoint


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    "common_data,global_batch_size,expected_global_step",
    [
        pytest.param(
            {"global_step": 2, "global_batch_size": 8, "consumed_samples": 16},
            8,
            2,
            id="checkpoint-with-consumed-samples",
        ),
        pytest.param(
            {"global_step": 2, "global_batch_size": 8},
            8,
            2,
            id="legacy-checkpoint-without-consumed-samples",
        ),
        pytest.param(
            {"global_step": 2, "global_batch_size": 8},
            4,
            4,
            id="legacy-checkpoint-with-changed-global-batch-size",
        ),
    ],
)
def test_static_batch_resume_uses_micro_batch_cursor(
        monkeypatch, tmp_path, common_data, global_batch_size, expected_global_step):
    """Static resume skips consumed micro-batches instead of optimizer steps."""
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    (checkpoint_path / "common.json").write_text(json.dumps(common_data), encoding="utf-8")
    trainer = _build_trainer(global_batch_size)
    load_checkpoint = _mock_checkpoint_load(monkeypatch)
    model = object()
    optimizer = object()

    trainer._load_checkpoint(str(checkpoint_path), model, optimizer)

    trainer.train_dataset.set_init_step.assert_called_once_with(4)
    assert trainer._resumed
    assert trainer._consumed_samples == 16
    assert trainer.state.consumed_samples == 16
    assert trainer.state.global_step == expected_global_step
    load_checkpoint.assert_called_once_with(
        checkpoint=str(checkpoint_path),
        network=model,
        optimizer=optimizer,
        global_step=expected_global_step,
        balanced_load=False,
    )
