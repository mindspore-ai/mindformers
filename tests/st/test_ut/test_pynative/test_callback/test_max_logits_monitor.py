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
"""Test pynative MaxLogitsMonitor."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mindformers.pynative.callback.max_logits_monitor import (
    MaxLogitsMonitor,
    MaxLogitsReset,
    configure_max_logits_tracking,
    ensure_max_logits_reset_callback,
)


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
