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
from unittest.mock import Mock, call

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
            base_load_path="",
            reshard_worker_num=1,
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


def _passthrough_checkpoint_path(path, **_):
    """Stand-in for `get_checkpoint_path`, tolerating the keyword arguments the trainer passes."""
    return path


def _mock_checkpoint_load(monkeypatch, get_checkpoint_path=None):
    """Mock checkpoint IO while exercising the real Trainer resume logic.

    `get_checkpoint_path` accepts the keyword arguments the trainer passes (currently
    `require_optimizer`); pass a Mock to assert on them.
    """
    if get_checkpoint_path is None:
        get_checkpoint_path = _passthrough_checkpoint_path
    monkeypatch.setattr(trainer_module, "get_checkpoint_path", get_checkpoint_path)
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
    optimizer = Mock()

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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("no_load_optim", [False, True])
def test_pipeline_checkpoint_load_manages_shared_optimizer_once(monkeypatch, tmp_path, no_load_optim):
    """Snapshot and reload shared fp32 masters once for all pipeline stages."""
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    (checkpoint_path / "common.json").write_text(
        json.dumps({"global_step": 2, "global_batch_size": 8, "consumed_samples": 16}),
        encoding="utf-8",
    )
    trainer = _build_trainer()
    trainer.config.checkpoint.no_load_optim = no_load_optim
    get_checkpoint_path = Mock(side_effect=lambda path, **_: path)
    load_checkpoint = _mock_checkpoint_load(monkeypatch, get_checkpoint_path)
    models = [object(), object()]
    optimizer = Mock()
    events = []
    optimizer.save_main_params_snapshot.side_effect = lambda: events.append("snapshot")
    load_checkpoint.side_effect = lambda **kwargs: events.append(kwargs["network"])
    optimizer.reload_main_params_from_model.side_effect = lambda: events.append("reload")

    trainer._load_checkpoint(str(checkpoint_path), models, optimizer)

    # Weights-only resume must not make the optimizer files part of the integrity check.
    get_checkpoint_path.assert_called_once_with(
        str(checkpoint_path), require_optimizer=not no_load_optim
    )

    expected_events = [models[0], models[1], "reload"]
    if no_load_optim:
        optimizer.save_main_params_snapshot.assert_not_called()
    else:
        expected_events.insert(0, "snapshot")
        optimizer.save_main_params_snapshot.assert_called_once_with()
    assert events == expected_events
    optimizer.reload_main_params_from_model.assert_called_once_with()
    loaded_optimizer = None if no_load_optim else optimizer
    loaded_global_step = None if no_load_optim else 2
    assert load_checkpoint.call_args_list == [
        call(
            checkpoint=str(checkpoint_path),
            network=models[0],
            optimizer=loaded_optimizer,
            global_step=loaded_global_step,
            balanced_load=False,
        ),
        call(
            checkpoint=str(checkpoint_path),
            network=models[1],
            optimizer=loaded_optimizer,
            global_step=loaded_global_step,
            balanced_load=False,
        ),
    ]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adapter_resume_loads_base_before_adapter(monkeypatch, tmp_path):
    """An adapter checkpoint restores its declared base before loading adapter tensors."""
    checkpoint_path = tmp_path / "adapter"
    checkpoint_path.mkdir()
    base_path = tmp_path / "base"
    base_path.mkdir()
    (checkpoint_path / "common.json").write_text(
        json.dumps({"global_step": 2, "global_batch_size": 8, "consumed_samples": 16}),
        encoding="utf-8",
    )
    (checkpoint_path / "mindformers_adapter_config.json").write_text(
        json.dumps({"base_fingerprint": "same-base"}), encoding="utf-8"
    )
    trainer = _build_trainer()
    trainer.config.checkpoint.base_load_path = str(base_path)
    events = []
    trainer._load_base_checkpoint = Mock(side_effect=lambda path, _: events.append(("base", path)))
    monkeypatch.setattr(trainer_module, "get_base_checkpoint_fingerprint", lambda _: "same-base")
    load_checkpoint = _mock_checkpoint_load(monkeypatch)
    load_checkpoint.side_effect = lambda **_: events.append(("adapter", str(checkpoint_path)))

    trainer._load_checkpoint(str(checkpoint_path), object(), Mock())

    assert events == [("base", str(base_path)), ("adapter", str(checkpoint_path))]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adapter_resume_rejects_mismatched_base(monkeypatch, tmp_path):
    """A base fingerprint mismatch fails before either checkpoint is loaded."""
    checkpoint_path = tmp_path / "adapter"
    checkpoint_path.mkdir()
    base_path = tmp_path / "base"
    base_path.mkdir()
    (checkpoint_path / "mindformers_adapter_config.json").write_text(
        json.dumps({"base_fingerprint": "expected-base"}), encoding="utf-8"
    )
    trainer = _build_trainer()
    trainer.config.checkpoint.base_load_path = str(base_path)
    trainer._load_base_checkpoint = Mock()
    load_checkpoint = _mock_checkpoint_load(monkeypatch)
    monkeypatch.setattr(trainer_module, "get_base_checkpoint_fingerprint", lambda _: "different-base")

    with pytest.raises(ValueError, match="fingerprint does not match"):
        trainer._load_checkpoint(str(checkpoint_path), object(), Mock())

    trainer._load_base_checkpoint.assert_not_called()
    load_checkpoint.assert_not_called()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adapter_resume_warns_when_base_fingerprint_is_unavailable(monkeypatch, tmp_path, caplog):
    """Adapter loading warns instead of silently skipping an unavailable fingerprint."""
    checkpoint_path = tmp_path / "adapter"
    checkpoint_path.mkdir()
    base_path = tmp_path / "base"
    base_path.mkdir()
    (checkpoint_path / "mindformers_adapter_config.json").write_text(
        json.dumps({"base_fingerprint": None}), encoding="utf-8"
    )
    trainer = _build_trainer()
    trainer.config.checkpoint.base_load_path = str(base_path)
    trainer._load_base_checkpoint = Mock()
    monkeypatch.setattr(trainer_module, "get_base_checkpoint_fingerprint", lambda _: None)
    model = object()

    assert trainer._prepare_adapter_load(str(checkpoint_path), model) is True
    assert "Cannot fully verify adapter base checkpoint fingerprint" in caplog.text
    trainer._load_base_checkpoint.assert_called_once_with(str(base_path), model)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    "is_adapter,is_hf,expected_loader",
    [
        pytest.param(True, True, "mindformers", id="adapter"),
        pytest.param(False, False, "mindformers", id="mindformers-full"),
        pytest.param(False, True, "huggingface", id="huggingface-full"),
    ],
)
def test_inference_checkpoint_selects_loader(
        monkeypatch, tmp_path, is_adapter, is_hf, expected_loader):
    """Inference uses the adapter-aware loader without duplicating checkpoint IO."""
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    trainer = _build_trainer()
    trainer.config.checkpoint.load_path = str(checkpoint_path)
    trainer._prepare_adapter_load = Mock(return_value=is_adapter)
    model = object()

    monkeypatch.setattr(trainer_module, "is_checkpoint_path_valid", lambda _: True)
    is_hf_checkpoint = Mock(return_value=is_hf)
    monkeypatch.setattr(trainer_module, "is_hf_checkpoint", is_hf_checkpoint)
    monkeypatch.setattr(trainer_module, "get_checkpoint_path", lambda path, **_: path)
    load_checkpoint = Mock()
    load_hf_checkpoint = Mock()
    monkeypatch.setattr(trainer_module, "load_checkpoint", load_checkpoint)
    monkeypatch.setattr(trainer_module, "load_hf_checkpoint", load_hf_checkpoint)

    trainer._load_inference_checkpoint(model)

    trainer._prepare_adapter_load.assert_called_once_with(str(checkpoint_path), model)
    if expected_loader == "huggingface":
        load_hf_checkpoint.assert_called_once_with(
            pretrained_model_dir=str(checkpoint_path),
            network=model,
            balanced_load=False,
            reshard_worker_num=1,
        )
        load_checkpoint.assert_not_called()
    else:
        load_checkpoint.assert_called_once_with(
            checkpoint=str(checkpoint_path),
            network=model,
            balanced_load=False,
            reshard_worker_num=1,
        )
        load_hf_checkpoint.assert_not_called()

    if is_adapter:
        is_hf_checkpoint.assert_not_called()
    else:
        is_hf_checkpoint.assert_called_once_with(str(checkpoint_path))
