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
"""Max attention logit monitoring callback for the pynative Trainer."""
import numpy as np

from mindformers.tools.logger import logger
from mindformers.tools.register.register import MindFormerRegister, MindFormerModuleType

from .callback import TrainerCallback


def _unwrap_model(model):
    """Strip wrapper Cells (e.g. AMP / FSDP) to reach the GPTModel."""
    while hasattr(model, "network"):
        model = model.network
    return model


def _reset_max_attention_logit(network):
    """Zero every layer's running max so the next training step starts fresh."""
    network = _unwrap_model(network)
    getattr(network, "model", network).reset_max_attention_logit()


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class MaxLogitsMonitor(TrainerCallback):
    """
    Callback that dumps per-layer max attention logits to log and resets them.

    Args:
        step_interval (int, optional): Emit every N steps. Default: ``1``.
    """

    def __init__(
        self,
        step_interval: int = 1,
    ):
        super().__init__()
        if not isinstance(step_interval, int) or step_interval <= 0:
            raise ValueError(
                f"step_interval must be a positive int, got {step_interval!r}."
            )

        self.step_interval = step_interval

    def on_step_end(self, args, state, **kwargs):
        step = getattr(state, "global_step", 0)
        model = _unwrap_model(kwargs.get("model"))
        if model is None:
            return

        if self.step_interval > 1 and step % self.step_interval != 0:
            _reset_max_attention_logit(model)
            return

        for m in model:
            # 1) collect per-layer Parameter values.
            params = m.get_max_attention_logit()
            if not params:
                _reset_max_attention_logit(m)
                return

            # 2) dump.
            self._dump(params, state)

            # 3) reset for the next step.
            _reset_max_attention_logit(m)

    @staticmethod
    def _fmt(v):
        """Format a single number to 4 significant digits."""
        return f"{float(v):.4g}"

    def _fmt_list(self, vs):
        return "[" + ", ".join(self._fmt(x) for x in vs) + "]"

    def _dump(self, params, state):
        """Match TrainingStateMonitor._dump_max_attention_logit print format."""
        vals = []
        for param_name, param in params.items():
            param = param.full_tensor() if hasattr(param, "full_tensor") else param
            v = param.asnumpy()
            self._to_log(f"max_attention_logit/{param_name}", self._fmt_list(v), state)
            vals.extend(v)

        if vals:
            self._to_log('max_attention_logit/mean', self._fmt(np.mean(vals)), state)
            self._to_log('max_attention_logit/max', self._fmt(np.max(vals)), state)

    def _to_log(self, tag, data, state):
        """Mirror TrainingStateMonitor print format: step:[c/d] tag: data."""
        global_step = state.global_step
        max_steps = state.max_steps
        logger.info(
            "step:[%5d/%5d] %s: %s",
            global_step, max_steps, tag, data
        )


# Kept for backward compatibility; MaxLogitsMonitor now already covers reset.
@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class MaxLogitsReset(MaxLogitsMonitor):
    """Deprecated: use MaxLogitsMonitor instead."""

    def __init__(self):
        super().__init__(step_interval=1)


def configure_max_logits_tracking(config, callbacks=None, optimizer=None):
    """Enable model-side max attention logit tracking when optimizer or callbacks need it."""
    model_config = getattr(config, "model", None)
    if model_config is None:
        return False

    needs_tracking = (
        _needs_muon_qk_clip(optimizer or getattr(config, "optimizer", None))
        or _has_enabled_monitor(callbacks)
        or _has_enabled_monitor(getattr(config, "callbacks", None))
    )
    model_config.track_max_attention_logit = needs_tracking
    return needs_tracking


def ensure_max_logits_reset_callback(callbacks, enabled=False):
    """Ensure one MaxLogitsMonitor is present when max-logit tracking is enabled."""
    callback_list = list(callbacks or [])
    has_monitor = any(
        _is_callback_type(callback, MaxLogitsMonitor)
        for callback in callback_list
    )
    if has_monitor:
        # MaxLogitsMonitor already includes reset logic.
        return callback_list

    if not enabled:
        return callback_list

    # Remove legacy MaxLogitsReset callbacks and add a unified MaxLogitsMonitor.
    callback_list = [
        callback for callback in callback_list
        if not _is_callback_type(callback, MaxLogitsReset)
    ]
    callback_list.append(MaxLogitsMonitor())
    return callback_list


def _needs_muon_qk_clip(optimizer):
    """Return whether a Muon optimizer instance/config enables QK clip."""
    return (
        _type_name(optimizer) == "Muon"
        and bool(_get_value(optimizer, "qk_clip_enabled", True))
    )


def _has_enabled_monitor(callbacks):
    """Return whether callbacks contain a MaxLogitsMonitor."""
    return any(
        _is_callback_type(callback, MaxLogitsMonitor)
        for callback in callbacks or []
    )


def _is_callback_type(callback, callback_type):
    """Return whether callback instance, class, dict, or config matches callback_type."""
    if callback is callback_type or isinstance(callback, callback_type):
        return True
    if isinstance(callback, type):
        return issubclass(callback, callback_type)
    return _type_name(callback) == callback_type.__name__


def _type_name(obj):
    """Get type name from config objects, dict configs, classes, or instances."""
    return _get_value(obj, "type") or (
        obj.__name__ if isinstance(obj, type) else obj.__class__.__name__
    )


def _get_value(obj, key, default=None):
    """Read a key from dict-like config or object attributes."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
