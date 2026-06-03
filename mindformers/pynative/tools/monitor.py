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
"""Monitor for collecting and outputting local/device loss and norm during training."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from mindformers.tools import logger

try:
    from hyper_parallel.core.dtensor import DTensor
except ImportError:
    DTensor = None


class Monitor(ABC):
    """Base class for monitors. Subclass to implement specific metric recording.

    Subclasses must implement:
        record(value, context) — collect a metric
    """

    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    @abstractmethod
    def record(self, value: Any = None, context: Optional[Dict[str, Any]] = None):
        """Collect a metric."""

    def reset(self):
        self._records = []

    def flush(self, step: int):
        self._flush_logger(self._records, step)
        self._flush_tensorboard(self._records, step)
        self._records = []

    def _flush_logger(self, records: List[Dict[str, Any]], step: int):  # pylint: disable=W0613
        for rec in records:
            parts = [f"{k}: {v}" for k, v in rec.items()]
            logger.info(f"{{ {', '.join(parts)} }}")

    def _flush_tensorboard(self, records: List[Dict[str, Any]], step: int):  # pylint: disable=W0613
        pass

    @staticmethod
    def _to_scalar(value: Any) -> float:
        if hasattr(value, "asnumpy"):
            return float(value.asnumpy())
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)


class TrainStateMonitor(Monitor):
    """Monitor for local/device loss and norm during training.

    Merges local_loss, local_norm, device_loss, device_norm into one class.
    Each metric is controlled by config and recorded via record(key, value, context).
    """

    def __init__(self, config):
        super().__init__()
        has_monitor = (hasattr(config, 'monitor')
                       and hasattr(config.monitor, 'train_state'))
        train_state = getattr(config.monitor, 'train_state', None) if has_monitor else None

        self._record_config: Dict[str, bool] = {}
        self._local_norm_config: List[str] = []
        self._device_norm_config: List[str] = []
        self._prev_local_norms: Dict[str, float] = {}

        if train_state:
            if getattr(train_state, 'local_loss', False):
                self._record_config["local_loss"] = True

            local_norm_config = getattr(train_state, 'local_norm', False)
            local_norm_active, self._local_norm_config = self._parse_norm_config(
                local_norm_config, "local_norm"
            )
            if local_norm_active:
                self._record_config["local_norm"] = True

            if getattr(train_state, 'device_loss', False):
                self._record_config["device_loss"] = True

            device_norm_config = getattr(train_state, 'device_norm', False)
            device_norm_active, self._device_norm_config = self._parse_norm_config(
                device_norm_config, "device_norm"
            )
            if device_norm_active:
                self._record_config["device_norm"] = True

        self._record_handlers = {
            "local_loss": self._record_local_loss,
            "local_norm": self._record_local_norm,
            "device_loss": self._record_device_loss,
            "device_norm": self._record_device_norm,
        }

    @property
    def active(self):
        return bool(self._record_config)

    def reset(self):
        super().reset()
        self._prev_local_norms = {}

    def record(self, key: str = None, value: Any = None, context: Optional[Dict[str, Any]] = None):
        collect_for_device_norm = key == "local_norm" and self._record_config.get("device_norm", False)
        if not self._record_config.get(key, False) and not collect_for_device_norm:
            return

        handler = self._record_handlers.get(key)
        if handler:
            handler(value, context or {})

    def _record_local_loss(self, value, context):
        micro_step = context.get("micro_step", 0)
        self._records.append({"micro_step": micro_step + 1, "local_loss": self._to_scalar(value)})

    def _record_local_norm(self, value, context):  # pylint: disable=W0613
        """Record local gradient norms for parameters matching the config filter."""
        from mindformers.pynative.trainer.utils import _get_grad_factor
        model = context.get("model")
        micro_step = context.get("micro_step", 0)

        if self._record_config.get("local_norm", False):
            self._records.append({"micro_step": micro_step + 1})
        for param in model.trainable_params():
            if param.grad is None:
                continue
            record_local_norm = self._should_record_norm(param.name, "local_norm")
            record_device_norm = self._should_record_norm(param.name, "device_norm")
            if not record_local_norm and not record_device_norm:
                continue
            grad = param.grad
            local_grad = grad.to_local() if DTensor and isinstance(grad, DTensor) else grad
            accumulated_norm = (float(local_grad.pow(2).sum().sqrt().asnumpy())
                                / _get_grad_factor(grad))
            if micro_step == 0:
                actual_norm = accumulated_norm
            else:
                actual_norm = accumulated_norm - self._prev_local_norms.get(param.name, 0.0)
            self._prev_local_norms[param.name] = accumulated_norm
            if record_local_norm:
                self._records.append({"local_norm": f"{param.name}: {actual_norm:10.6f}"})

    def _record_device_loss(self, value, context):  # pylint: disable=W0613
        self._records.append({"device_loss": self._to_scalar(value)})

    def _record_device_norm(self, value, context):  # pylint: disable=W0613
        for param_name, accumulated_norm in self._prev_local_norms.items():
            if not self._should_record_norm(param_name, "device_norm"):
                continue
            self._records.append({"device_norm": f"{param_name}: {accumulated_norm:10.6f}"})

    @staticmethod
    def _parse_norm_config(config, name: str) -> Tuple[bool, List[str]]:
        """Parse norm monitor config into an enabled flag and parameter-name filters."""
        if isinstance(config, bool):
            return config, []
        if isinstance(config, str):
            return bool(config), [config] if config else []
        if isinstance(config, list):
            if not all(isinstance(item, str) for item in config):
                raise TypeError(f"{name} only supports bool, str, or list[str].")
            config = [item for item in config if item]
            return bool(config), config
        raise TypeError(f"{name} only supports bool, str, or list[str].")

    def _should_record_norm(self, param_name: str, key: str) -> bool:
        """Return whether a parameter should be recorded for the requested norm metric."""
        if not self._record_config.get(key, False):
            return False
        norm_config = self._local_norm_config if key == "local_norm" else self._device_norm_config
        return not norm_config or any(p in param_name for p in norm_config if p)


class MonitorGroup:
    """Manage sub-monitors based on config. Delegates record/flush/reset.

    Extensible: new monitors can be registered here without modifying existing code.

    Usage in trainer:
        self.monitor = MonitorGroup(config)
        self.monitor.reset()
        self.monitor.record("local_loss", micro_loss, context={...})
        self.monitor.record("local_norm", context={...})
        self.monitor.record("device_loss", loss)
        self.monitor.record("device_norm")
        # flush is called in TrainingStateCallback
    """

    def __init__(self, config):
        self._monitors: Dict[str, Monitor] = {}

        train_state_monitor = TrainStateMonitor(config)
        if train_state_monitor.active:
            self._monitors["train_state"] = train_state_monitor

    @property
    def active(self):
        return bool(self._monitors)

    def reset(self):
        for m in self._monitors.values():
            m.reset()

    def record(self, key: str, value: Any = None, context: Optional[Dict[str, Any]] = None):
        for m in self._monitors.values():
            m.record(key, value, context)

    def flush(self, step: int):
        for m in self._monitors.values():
            m.flush(step)
