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
"""Tests for the pynative max-logits and QK-clip count monitor."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mindspore import Tensor, dtype

from mindformers.pynative.callback.max_logits_monitor import (
    MaxLogitsMonitor,
    MaxLogitsReset,
    configure_max_logits_tracking,
    ensure_max_logits_reset_callback,
)


class _Model:
    """Minimal model surface consumed by ``MaxLogitsMonitor``."""

    def __init__(self, values):
        """Store the per-head max-logit values returned by the model."""
        self.values = values
        self.reset_count = 0

    def get_max_attention_logit(self):
        """Return one MLA layer's tracked per-head max logits."""
        return {
            "decoder.layers.0.self_attention.core_attention.max_logits_val":
                Tensor(self.values, dtype=dtype.float32)
        }

    def reset_max_attention_logit(self):
        """Record that the monitor reset the model's step-local state."""
        self.reset_count += 1


def _muon(enabled=True):
    """Return a minimal Muon-shaped optimizer exposing the QK-clip threshold."""
    optimizer = type("Muon", (), {})()
    optimizer.qk_clip_enabled = enabled
    optimizer.logit_threshold = Tensor([100.0], dtype=dtype.float32)
    return optimizer


def _state():
    state = type("State", (), {})()
    state.global_step = 3
    state.max_steps = 10
    state.consumed_samples = 24
    return state


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestMaxLogitsMonitor:
    """Test logging and reset policies of MaxLogitsMonitor."""

    @staticmethod
    def _model(params=None):
        model = MagicMock(spec=["get_max_attention_logit", "reset_max_attention_logit"])
        model.get_max_attention_logit.return_value = params or {}
        return model

    @staticmethod
    def _state(step):
        return SimpleNamespace(global_step=step, max_steps=10)

    @pytest.mark.parametrize("step_interval", [0, -1, 1.5, "2"])
    def test_invalid_step_interval(self, step_interval):
        """step_interval must remain a positive integer."""
        with pytest.raises(ValueError, match="step_interval must be a positive int"):
            MaxLogitsMonitor(step_interval=step_interval)

    def test_invalid_enable_logging(self):
        """enable_logging accepts booleans only."""
        with pytest.raises(ValueError, match="enable_logging must be a bool"):
            MaxLogitsMonitor(enable_logging="false")

    def test_disabled_logging_still_resets_every_model(self):
        """Disabling output must not disable the per-step reset."""
        models = [self._model({"layer": object()}), self._model({"layer": object()})]
        callback = MaxLogitsMonitor(enable_logging=False)

        with patch.object(callback, "_dump") as mock_dump:
            callback.on_step_end(None, self._state(1), model=models)

        mock_dump.assert_not_called()
        for model in models:
            model.get_max_attention_logit.assert_not_called()
            model.reset_max_attention_logit.assert_called_once_with()

    def test_interval_skips_logging_but_resets_every_step(self):
        """Non-log steps skip collection and still reset all pipeline chunks."""
        models = [self._model({"layer": object()}), self._model({"layer": object()})]
        callback = MaxLogitsMonitor(step_interval=2)

        with patch.object(callback, "_dump") as mock_dump:
            callback.on_step_end(None, self._state(1), model=models)

        mock_dump.assert_not_called()
        for model in models:
            model.get_max_attention_logit.assert_not_called()
            model.reset_max_attention_logit.assert_called_once_with()

    def test_interval_logs_and_resets_every_model(self):
        """Log steps collect, dump, and reset every pipeline chunk."""
        params = {"decoder.layers.0.max_logits_val": object()}
        models = [self._model(params), self._model(params)]
        callback = MaxLogitsMonitor(step_interval=2)

        with patch.object(callback, "_dump") as mock_dump:
            callback.on_step_end(None, self._state(2), model=models)

        assert mock_dump.call_count == 2
        for model in models:
            model.get_max_attention_logit.assert_called_once_with()
            model.reset_max_attention_logit.assert_called_once_with()

    def test_single_model_is_supported(self):
        """Programmatic Trainer use may pass one model instead of a list."""
        model = self._model()
        callback = MaxLogitsMonitor()

        callback.on_step_end(None, self._state(1), model=model)

        model.get_max_attention_logit.assert_called_once_with()
        model.reset_max_attention_logit.assert_called_once_with()

    def test_legacy_reset_callback_is_silent(self):
        """MaxLogitsReset keeps its reset-only behavior."""
        callback = MaxLogitsReset()
        assert callback.enable_logging is False

    def test_disabled_monitor_does_not_enable_unused_tracking(self):
        """A silent monitor alone has no reason to allocate tracking state."""
        config = SimpleNamespace(
            model=SimpleNamespace(track_max_attention_logit=False),
            optimizer=SimpleNamespace(type="AdamW"),
            callbacks=[{"type": "MaxLogitsMonitor", "enable_logging": False}],
        )

        assert configure_max_logits_tracking(config) is False
        assert config.model.track_max_attention_logit is False

    def test_qk_clip_keeps_tracking_with_disabled_logging(self):
        """QK clip still enables tracking when max-logit output is disabled."""
        config = SimpleNamespace(
            model=SimpleNamespace(track_max_attention_logit=False),
            optimizer=SimpleNamespace(type="Muon", qk_clip_enabled=True),
            callbacks=[{"type": "MaxLogitsMonitor", "enable_logging": False}],
        )

        assert configure_max_logits_tracking(config) is True
        assert config.model.track_max_attention_logit is True

    def test_explicit_monitor_prevents_duplicate_auto_callback(self):
        """The reset-capable explicit callback must be reused for QK clip."""
        monitor = MaxLogitsMonitor(enable_logging=False)

        callbacks = ensure_max_logits_reset_callback([monitor], enabled=True)

        assert callbacks == [monitor]

    def test_cached_device_count_skips_quiet_per_layer_collection(self):
        """A Muon-produced device count avoids callback full-tensor/readback work."""
        model = self._model({"layer": object()})
        count_owner = MagicMock(spec=["take_qk_clip_count"])
        cached_count = Tensor([2], dtype=dtype.int32)
        count_owner.take_qk_clip_count.return_value = cached_count
        optimizer = _muon()
        optimizer.model = count_owner
        callback = MaxLogitsMonitor(step_interval=10, enable_logging=False)

        with patch.object(callback, "_dump") as mock_dump, \
                patch("mindformers.pynative.callback.max_logits_monitor._is_last_rank", return_value=True), \
                patch("mindformers.pynative.callback.max_logits_monitor.save_monitor_data") as mock_save, \
                patch("mindformers.pynative.callback.max_logits_monitor.logger") as mock_logger:
            callback.on_step_end(
                None, self._state(1), model=[model], optimizer=optimizer,
                pp_metric_reduce_group=None, pp_metric_reduce_group_size=None,
            )

        count_owner.take_qk_clip_count.assert_called_once_with()
        model.get_max_attention_logit.assert_not_called()
        mock_dump.assert_not_called()
        model.reset_max_attention_logit.assert_called_once_with()
        assert any(call.args[3:] == ("qk_clip_count", 2) for call in mock_logger.info.call_args_list)
        mock_save.assert_called_once_with(
            "qk_clip_count", 2, step=1, consumed_samples=None)

    def test_cached_count_keeps_threshold_on_device_during_logging(self):
        """Per-layer logging must not re-read the threshold or recount on host."""
        params = {"layer": object()}
        state = self._state(1)
        model = self._model(params)
        count_owner = MagicMock(spec=["take_qk_clip_count"])
        count_owner.take_qk_clip_count.return_value = Tensor([2], dtype=dtype.int32)
        optimizer = _muon()
        optimizer.model = count_owner
        callback = MaxLogitsMonitor()

        with patch.object(callback, "_dump", return_value=99) as mock_dump, \
                patch.object(callback, "_get_host_qk_clip_threshold") as mock_threshold, \
                patch("mindformers.pynative.callback.max_logits_monitor._is_last_rank", return_value=True), \
                patch("mindformers.pynative.callback.max_logits_monitor.save_monitor_data"):
            callback.on_step_end(
                None, state, model=[model], optimizer=optimizer,
                pp_metric_reduce_group=None, pp_metric_reduce_group_size=None,
            )

        mock_threshold.assert_not_called()
        mock_dump.assert_called_once_with(
            params, state,
            qk_clip_threshold=None, enable_logging=True,
        )

    def test_non_writer_rank_does_not_read_cached_count_to_host(self):
        """Only the global metric-writer rank may cross the device/host boundary."""
        model = self._model()
        cached_count = MagicMock(spec=["asnumpy"])
        count_owner = MagicMock(spec=["take_qk_clip_count"])
        count_owner.take_qk_clip_count.return_value = cached_count
        optimizer = _muon()
        optimizer.model = count_owner

        with patch(
                "mindformers.pynative.callback.max_logits_monitor._is_last_rank",
                return_value=False,
        ), patch("mindformers.pynative.callback.max_logits_monitor.save_monitor_data") as mock_save:
            MaxLogitsMonitor(enable_logging=False).on_step_end(
                None, self._state(1), model=[model], optimizer=optimizer,
                pp_metric_reduce_group=None, pp_metric_reduce_group_size=None,
            )

        cached_count.asnumpy.assert_not_called()
        mock_save.assert_not_called()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindformers.pynative.callback.max_logits_monitor.save_monitor_data")
@patch("mindformers.pynative.callback.max_logits_monitor._is_last_rank", return_value=True)
@patch("mindformers.pynative.callback.max_logits_monitor.logger")
def test_qk_clip_count_uses_strict_head_threshold(
        mock_logger, mock_is_last_rank, mock_save_monitor_data):
    """Count MLA heads strictly above threshold and publish the per-step value."""
    model = _Model(np.array([99.0, 100.0, 101.0, 150.0], np.float32))
    state = _state()

    MaxLogitsMonitor().on_step_end(
        None, state, model=[model], optimizer=_muon(),
        pp_metric_reduce_group=None, pp_metric_reduce_group_size=None,
    )

    assert model.reset_count == 1
    assert any(call.args[3:] == ("qk_clip_count", 2) for call in mock_logger.info.call_args_list)
    mock_is_last_rank.assert_called_once_with()
    mock_save_monitor_data.assert_called_once_with(
        "qk_clip_count", 2, step=3, consumed_samples=24)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindformers.pynative.callback.max_logits_monitor.save_monitor_data")
@patch("mindformers.pynative.callback.max_logits_monitor.logger")
def test_qk_clip_count_is_independent_of_max_logit_logging(mock_logger, mock_save_monitor_data):
    """Max-logit output controls must not suppress the per-step QK-clip count."""
    model = _Model(np.array([99.0, 101.0], np.float32))

    MaxLogitsMonitor(step_interval=10, enable_logging=False).on_step_end(
        None, _state(), model=[model], optimizer=_muon())

    assert model.reset_count == 1
    assert not any(
        call.args[3].startswith("max_attention_logit/")
        for call in mock_logger.info.call_args_list
    )
    assert any(call.args[3:] == ("qk_clip_count", 1) for call in mock_logger.info.call_args_list)
    mock_save_monitor_data.assert_called_once_with(
        "qk_clip_count", 1, step=3, consumed_samples=24)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindformers.pynative.callback.max_logits_monitor.save_monitor_data")
@patch("mindformers.pynative.callback.max_logits_monitor.all_reduce")
@patch("mindformers.pynative.callback.max_logits_monitor._is_last_rank", return_value=True)
@patch("mindformers.pynative.callback.max_logits_monitor.logger")
def test_qk_clip_count_sums_pipeline_stages(
        mock_logger, mock_is_last_rank, mock_all_reduce, mock_save_monitor_data):
    """Sum full-head local-stage counts over PP before logging."""
    mock_all_reduce.return_value = Tensor([5], dtype=dtype.int32)
    model = _Model(np.array([101.0, 102.0], np.float32))
    state = _state()

    MaxLogitsMonitor().on_step_end(
        None, state, model=[model], optimizer=_muon(),
        pp_metric_reduce_group="pp-group", pp_metric_reduce_group_size=2,
    )

    mock_all_reduce.assert_called_once()
    mock_is_last_rank.assert_called_once_with()
    assert any(call.args[3:] == ("qk_clip_count", 5) for call in mock_logger.info.call_args_list)
    mock_save_monitor_data.assert_called_once_with(
        "qk_clip_count", 5, step=3, consumed_samples=24)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindformers.pynative.callback.max_logits_monitor.save_monitor_data")
@patch("mindformers.pynative.callback.max_logits_monitor.all_reduce")
@patch("mindformers.pynative.callback.max_logits_monitor._is_last_rank", return_value=False)
@patch("mindformers.pynative.callback.max_logits_monitor.logger")
def test_qk_clip_count_non_last_rank_only_contributes(
        mock_logger, mock_is_last_rank, mock_all_reduce, mock_save_monitor_data):
    """Non-last ranks must join the PP SUM without publishing duplicate metrics."""
    mock_all_reduce.return_value = Tensor([5], dtype=dtype.int32)
    model = _Model(np.array([101.0, 102.0], np.float32))

    MaxLogitsMonitor().on_step_end(
        None, _state(), model=[model], optimizer=_muon(),
        pp_metric_reduce_group="pp-group", pp_metric_reduce_group_size=2,
    )

    mock_all_reduce.assert_called_once()
    mock_is_last_rank.assert_called_once_with()
    assert not any(call.args[3:4] == ("qk_clip_count",) for call in mock_logger.info.call_args_list)
    mock_save_monitor_data.assert_not_called()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("mindformers.pynative.callback.max_logits_monitor.save_monitor_data")
@patch("mindformers.pynative.callback.max_logits_monitor.logger")
def test_qk_clip_count_is_hidden_when_clipping_is_disabled(mock_logger, mock_save_monitor_data):
    """Do not report hypothetical triggers for max-logit-only monitoring."""
    model = _Model(np.array([101.0, 150.0], np.float32))

    MaxLogitsMonitor().on_step_end(None, _state(), model=[model], optimizer=_muon(enabled=False))

    assert not any(call.args[3:4] == ("qk_clip_count",) for call in mock_logger.info.call_args_list)
    mock_save_monitor_data.assert_not_called()
