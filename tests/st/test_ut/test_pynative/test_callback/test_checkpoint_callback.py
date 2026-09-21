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
"""Test the save interval of CheckpointCallback."""
# pylint: disable=protected-access

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mindformers.pynative.callback.checkpoint_callback import CheckpointCallback
from mindformers.pynative.callback import checkpoint_callback as callback_module


def _collect_saved_steps(tmp_path, start_step, interval, num_steps):
    """Run ``num_steps`` steps from ``start_step`` and return the steps a save was triggered at."""
    callback = CheckpointCallback(save_path=str(tmp_path), save_interleaved_steps=interval)
    saved_steps = []
    callback._save_checkpoint = lambda args, state, **kwargs: saved_steps.append(state.global_step)

    args = MagicMock()
    state = SimpleNamespace(global_step=start_step)
    callback.on_train_begin(args, state)
    for _ in range(num_steps):
        state.global_step += 1
        callback.on_step_end(args, state)
    return saved_steps


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestCheckpointCallbackInterval:
    """Test that save_interleaved_steps keeps its spacing for fresh and resumed runs."""

    def test_save_interval_from_scratch(self, tmp_path):
        """
        Feature: CheckpointCallback save interval.
        Description: Train from step 0 with an interval of 70.
        Expectation: Checkpoints are saved every 70 steps.
        """
        assert _collect_saved_steps(tmp_path, 0, 70, 220) == [70, 140, 210]

    def test_save_interval_on_resume(self, tmp_path):
        """
        Feature: CheckpointCallback save interval.
        Description: Resume at step 300 with an interval of 70.
        Expectation: The interval restarts at the resumed step (370, 440, ...) instead of
            snapping to the absolute step grid (350, 420, ...).
        """
        saved_steps = _collect_saved_steps(tmp_path, 300, 70, 220)
        assert saved_steps == [370, 440, 510]
        assert 350 not in saved_steps


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_trainable_only_checkpoint_selects_lora_and_writes_manifest(tmp_path, monkeypatch):
    """Adapter saves pass a LoRA predicate and persist the base-checkpoint resume contract."""
    base_path = tmp_path / "base"
    base_path.mkdir()
    save_path = tmp_path / "adapter"
    captured = {}
    monkeypatch.setattr(callback_module, "get_real_group_size", lambda: 1)
    monkeypatch.setattr(callback_module, "get_real_rank", lambda: 0)
    monkeypatch.setattr(callback_module, "get_base_checkpoint_fingerprint", lambda _: "base-sha256")
    monkeypatch.setattr(callback_module, "save_checkpoint", lambda **kwargs: captured.update(kwargs))

    callback = CheckpointCallback(
        save_path=str(save_path),
        save_trainable_only=True,
        base_load_path=str(base_path),
        lora_config={"target_modules": r".*experts$", "lora_rank": 4},
    )
    model = SimpleNamespace(parameters_dict=lambda: {
        "model.experts.weight1": object(),
        "model.experts.weight1_lora_a": object(),
        "model.dense.lora_b": object(),
    })
    state = SimpleNamespace(global_step=7, consumed_samples=56)

    callback._save_checkpoint(None, state, model=model)

    choice_func = captured["model_choice_func"]
    assert not choice_func("model.experts.weight1")
    assert choice_func("model.experts.weight1_lora_a")
    assert choice_func("model.dense.lora_b")
    root_manifest = save_path / "mindformers_adapter_config.json"
    iteration_manifest = save_path / "iteration_00000007" / "mindformers_adapter_config.json"
    assert root_manifest.exists()
    assert iteration_manifest.exists()
    manifest = json.loads(root_manifest.read_text(encoding="utf-8"))
    assert manifest["base_fingerprint"] == "base-sha256"
    assert manifest["global_step"] == 7
    assert manifest["consumed_samples"] == 56
    assert manifest["lora_config"]["lora_rank"] == 4
