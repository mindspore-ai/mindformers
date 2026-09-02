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

from mindspore import Tensor, dtype, ops
from mindspore.mint.distributed import all_reduce, get_rank, get_world_size

from mindformers.pynative.tools.monitor_utils import save_monitor_data
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
    Dump per-layer max attention logits and the per-step QK-clip head count, then reset them.

    ``qk_clip_count`` is emitted only for an active Muon QK-clip optimizer. It
    counts MLA heads whose synchronized max logit is strictly greater than the
    configured threshold, and sums stage-local counts over pipeline parallelism.

    Args:
        step_interval (int, optional): Emit max-logit details every N steps.
            QK-clip counts are still collected every step. Default: ``1``.
        enable_logging (bool, optional): Whether to emit max-logit logs. The
            per-step reset remains enabled when this is ``False``. Default:
            ``True``.
    """

    def __init__(
        self,
        step_interval: int = 1,
        enable_logging: bool = True,
    ):
        super().__init__()
        if not isinstance(step_interval, int) or step_interval <= 0:
            raise ValueError(
                f"step_interval must be a positive int, got {step_interval!r}."
            )
        if not isinstance(enable_logging, bool):
            raise ValueError(
                f"enable_logging must be a bool, got {enable_logging!r}."
            )

        self.step_interval = step_interval
        self.enable_logging = enable_logging
        self._host_threshold_source = None
        self._host_threshold_value = None

    def on_step_end(self, args, state, **kwargs):
        step = getattr(state, "global_step", 0)
        model = kwargs.get("model")
        if model is None:
            return

        should_log = self.enable_logging and step % self.step_interval == 0
        optimizer = kwargs.get("optimizer")
        qk_clip_threshold = _get_qk_clip_threshold(optimizer)
        qk_clip_count = (
            _take_cached_qk_clip_count(optimizer)
            if qk_clip_threshold is not None else None
        )
        needs_fallback_count = qk_clip_threshold is not None and qk_clip_count is None
        should_collect = should_log or needs_fallback_count
        if qk_clip_count is None:
            qk_clip_count = 0
        models = model if isinstance(model, (list, tuple)) else (model,)
        for m in models:
            try:
                if not should_collect:
                    continue

                # 1) collect per-layer Parameter values.
                params = m.get_max_attention_logit()
                if not params:
                    continue

                # 2) dump.
                fallback_count = self._dump(
                    params,
                    state,
                    qk_clip_threshold=(
                        self._get_host_qk_clip_threshold(qk_clip_threshold)
                        if needs_fallback_count else None
                    ),
                    enable_logging=should_log,
                )
                if needs_fallback_count:
                    qk_clip_count += fallback_count
            finally:
                # Reset every step regardless of the logging policy.
                _reset_max_attention_logit(m)

        if qk_clip_threshold is not None:
            qk_clip_count = _reduce_qk_clip_count(
                qk_clip_count,
                kwargs.get("pp_metric_reduce_group"),
                kwargs.get("pp_metric_reduce_group_size"),
            )
            if _is_last_rank():
                qk_clip_count = _to_host_qk_clip_count(qk_clip_count)
                self._to_log("qk_clip_count", qk_clip_count, state)
                save_monitor_data(
                    "qk_clip_count",
                    qk_clip_count,
                    step=step,
                    consumed_samples=getattr(state, "consumed_samples", None),
                )

    def _get_host_qk_clip_threshold(self, threshold):
        """Materialize a fallback host threshold once for legacy model owners."""
        if threshold is self._host_threshold_source:
            return self._host_threshold_value
        value = threshold.asnumpy() if hasattr(threshold, "asnumpy") else threshold
        self._host_threshold_source = threshold
        self._host_threshold_value = float(np.asarray(value).reshape(-1)[0])
        return self._host_threshold_value

    @staticmethod
    def _fmt(v):
        """Format a single number to 4 significant digits."""
        return f"{float(v):.4g}"

    def _fmt_list(self, vs):
        return "[" + ", ".join(self._fmt(x) for x in vs) + "]"

    def _dump(self, params, state, qk_clip_threshold=None, enable_logging=True):
        """Match TrainingStateMonitor._dump_max_attention_logit format, logging this rank's head partition."""
        vals = []
        qk_clip_count = 0
        for param_name, param in params.items():
            # max_logits_val is a per-partition buffer now, so no full_tensor() gather.
            v = param.asnumpy().reshape(-1)
            if enable_logging:
                self._to_log(f"max_attention_logit/{param_name}", self._fmt_list(v), state)
                vals.extend(v)
            if qk_clip_threshold is not None:
                # Match Megatron's QK-clip count: count heads strictly above the
                # threshold, not layers whose clip function was merely called.
                qk_clip_count += int(np.count_nonzero(v > qk_clip_threshold))

        if vals:
            self._to_log('max_attention_logit/mean', self._fmt(np.mean(vals)), state)
            self._to_log('max_attention_logit/max', self._fmt(np.max(vals)), state)
        return qk_clip_count

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
        super().__init__(step_interval=1, enable_logging=False)


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
    """Return whether callbacks contain a MaxLogitsMonitor that emits logs."""
    return any(
        _is_callback_type(callback, MaxLogitsMonitor)
        and bool(_get_value(callback, "enable_logging", True))
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


def _get_qk_clip_threshold(optimizer):
    """Return the active Muon QK-clip threshold without synchronizing it to host."""
    if not _needs_muon_qk_clip(optimizer):
        return None

    threshold = _get_value(optimizer, "logit_threshold")
    return threshold


def _take_cached_qk_clip_count(optimizer):
    """Consume the count cached while Muon synchronized max attention logits."""
    model = _get_value(optimizer, "model")
    if model is None:
        return None
    take_count = getattr(model, "take_qk_clip_count", None)
    if not callable(take_count):
        return None
    return model.take_qk_clip_count()


def _reduce_qk_clip_count(qk_clip_count, pp_group=None, pp_group_size=None):
    """Sum an optimizer-produced stage-local count over pipeline parallelism."""
    if pp_group is None or pp_group_size is None or pp_group_size <= 1:
        return qk_clip_count

    count = (
        qk_clip_count if isinstance(qk_clip_count, Tensor)
        else Tensor([qk_clip_count], dtype=dtype.int32)
    )
    reduced = all_reduce(count, op=ops.ReduceOp.SUM, group=pp_group)
    if reduced is not None:
        count = reduced
    return count


def _to_host_qk_clip_count(qk_clip_count):
    """Read one final scalar only on the single global metric-writer rank."""
    if hasattr(qk_clip_count, "asnumpy"):
        qk_clip_count = qk_clip_count.asnumpy()
    return int(np.asarray(qk_clip_count).reshape(-1)[0])


def _is_last_rank():
    """Return whether this process owns the single global metric writer."""
    try:
        return get_rank() == get_world_size() - 1
    except RuntimeError:
        return True
