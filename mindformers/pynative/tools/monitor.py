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
"""Monitor for collecting and outputting training metrics during training."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from mindspore.mint.distributed import get_rank, get_world_size

from mindformers.tools import logger
from mindformers.pynative.tools.monitor_utils import (
    clear_monitor_data_tracker,
    get_monitor_data_tracker,
    reset_monitor_data_tracker,
    save_monitor_data,
)

try:
    from hyper_parallel.core.dtensor import DTensor
except ImportError:
    DTensor = None


def _config_to_dict(config):
    """Convert a training config into tracker-friendly key-value data."""
    if config is None:
        return {}
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if isinstance(config, dict):
        return config
    return vars(config) if hasattr(config, "__dict__") else {"value": config}


class Monitor(ABC):
    """Base class for monitors. Subclass to implement specific metric recording.

    Subclasses must implement:
        record(value, context) — collect a metric
    """

    def __init__(self, model=None):
        self._records: List[Dict[str, Any]] = []
        self._model = model

    @abstractmethod
    def record(self, value: Any = None, context: Optional[Dict[str, Any]] = None):
        """Collect a metric."""

    def reset(self):
        self._records = []

    def setup(self, config=None, state=None):
        """Initialize monitor resources before training starts."""
        _ = config, state

    def close(self):
        """Release monitor resources after training ends."""

    def flush(self, step: int):
        self._flush_logger(self._records, step)
        self._flush_tensorboard(self._records, step)
        self._records = []

    def _flush_logger(self, records: List[Dict[str, Any]], step: int):
        """Output collected records to logger with step/micro_step context."""
        cur_micro_step = None
        pending_norms: List[str] = []
        pending_type: Optional[str] = None

        def flush_pending():
            nonlocal pending_type
            if not pending_norms:
                return
            header_parts = [f'"step": {step}']
            if pending_type == "local_norm" and cur_micro_step is not None:
                header_parts.append(f'"micro_step": {cur_micro_step}')
            logger.info(f'{{ {", ".join(header_parts)} }}')
            for norm_val in pending_norms:
                logger.info(f'{{ "{pending_type}": {norm_val} }}')
            pending_norms.clear()
            pending_type = None

        for rec in records:
            micro_step = rec.pop("micro_step", None)
            if micro_step is not None and micro_step != cur_micro_step:
                flush_pending()
                cur_micro_step = micro_step
            if not rec:
                continue

            norm_key = next((k for k in rec if k in ("local_norm", "device_norm")), None)

            if norm_key is not None:
                if norm_key != pending_type:
                    flush_pending()
                    pending_type = norm_key
                pending_norms.append(rec[norm_key])
            else:
                flush_pending()
                rec_with_ctx = {"step": step}
                if cur_micro_step is not None:
                    rec_with_ctx["micro_step"] = cur_micro_step
                rec_with_ctx.update(rec)
                logger.info(json.dumps(rec_with_ctx))

        flush_pending()

    def _flush_tensorboard(self, records: List[Dict[str, Any]], step: int):  # pylint: disable=W0613
        pass

    def set_model(self, model):
        """Bind a model instance to the monitor."""
        self._model = model

    def _resolve_model(self):
        """Return the bound model."""
        return self._model

    @staticmethod
    def _to_scalar(value: Any) -> float:
        if hasattr(value, "asnumpy"):
            return float(value.asnumpy())
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)


class TensorboardMonitor(Monitor):
    """Write shared monitor data to TensorBoard."""

    def __init__(self, config, model=None):
        super().__init__(model=model)
        tensorboard = getattr(getattr(config, "monitor", None), "tensorboard", None)
        self._output_dir = str(getattr(tensorboard, "output_dir", "") or "").strip()
        self._log_interval = self._positive_int(
            getattr(tensorboard, "log_interval", 1),
            "log_interval",
            1,
        )
        self._queue_size = self._positive_int(
            getattr(tensorboard, "queue_size", 1000),
            "queue_size",
            1000,
        )
        self._writer = None
        self._written_texts: set[str] = set()

    @property
    def configured(self) -> bool:
        """Return whether TensorBoard is configured, independent of rank."""
        return bool(self._output_dir)

    @property
    def active(self) -> bool:
        """Return whether this monitor is configured for the current run."""
        return self.configured

    def setup(self, config=None, state=None):
        """Create the writer on the last rank and write configuration text once."""
        _ = state
        if not self.configured or self._writer is not None:
            return

        tracker = get_monitor_data_tracker()
        if not tracker.config:
            tracker.set_config(_config_to_dict(config))

        try:
            rank = get_rank()
            world_size = get_world_size()
        except RuntimeError:
            rank, world_size = 0, 1
        if rank != world_size - 1:
            return

        from tensorboardX import SummaryWriter

        self._writer = SummaryWriter(log_dir=self._output_dir, max_queue=self._queue_size)
        self._write_config()

    def record(self, key: str = None, value: Any = None,
               context: Optional[Dict[str, Any]] = None):
        """Publish an explicitly recorded value into the shared monitor store."""
        if key is None:
            return
        context = context or {}
        training_state = dict(context.get("training_state") or {})
        for state_name in ("step", "consumed_samples", "micro_step"):
            if state_name in context:
                training_state[state_name] = context[state_name]
        save_monitor_data(key, value, **training_state)

    def flush(self, step: int):
        """Consume current-step data and flush the TensorBoard event queue."""
        if self._writer is None or step % self._log_interval != 0:
            return

        tracker = get_monitor_data_tracker()
        global_step = tracker.training_state.get("step")
        if global_step is None:
            global_step = step
        consumed_samples = tracker.training_state.get("consumed_samples")

        for name, value in tracker.data.items():
            try:
                value = self._to_scalar(value)
            except (TypeError, ValueError, RuntimeError):
                logger.warning("Skip non-scalar TensorBoard data %s with type %s.", name, type(value).__name__)
                continue
            self._writer.add_scalar(name, value, global_step=global_step)

            if consumed_samples is not None:
                self._writer.add_scalar(
                    f"{name} vs samples", value, global_step=consumed_samples
                )
        self._writer.flush()

    def close(self):
        """Flush and close the TensorBoard writer."""
        if self._writer is None:
            return
        self._writer.flush()
        self._writer.close()
        self._writer = None

    def _write_config(self):
        if self._writer is None:
            return
        for key, value in get_monitor_data_tracker().config.items():
            self._writer.add_text(key, str(value), global_step=0)
            self._written_texts.add(key)
        self._writer.flush()

    @staticmethod
    def _positive_int(value, name: str, default: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            logger.warning("%s must be a positive integer, got %s; use %s.", name, value, default)
            return default
        return value


def _iter_models(model):
    """Yield one or more bound model objects."""
    if model is None:
        return ()
    if isinstance(model, (list, tuple)):
        return tuple(m for m in model if m is not None)
    return (model,)


def _first_model(model):
    """Return the first model when a model list is bound."""
    models = _iter_models(model)
    return models[0] if models else None


def _parse_moe_tpe_module_name(module_name: str) -> Optional[Tuple[str, Optional[int], int]]:
    """Parse a MoE MLP module name into ``(block, mtp_idx, layer)``."""
    match = re.fullmatch(r"(?:.*\.)?decoder\.layers\.(\d+)\.mlp", module_name)
    if match:
        return "decoder", None, int(match.group(1))
    match = re.fullmatch(r"(?:.*\.)?mtp\.layers\.(\d+)\.transformer_layer\.mlp", module_name)
    if match:
        return "mtp", int(match.group(1)), 0
    match = re.fullmatch(r"(?:.*\.)?mtp\.layers\.(\d+)\.mtp_model_layer\.layers\.(\d+)\.mlp", module_name)
    if match:
        return "mtp", int(match.group(1)), int(match.group(2))
    return None


class TrainStateMonitor(Monitor):
    """Monitor for local/device loss and norm during training.

    Merges local_loss, local_norm, device_loss, device_norm into one class.
    Each metric is controlled by config and recorded via record(key, value, context).
    """

    def __init__(self, config, model=None):
        super().__init__(model=model)
        has_monitor = (hasattr(config, 'monitor')
                       and hasattr(config.monitor, 'train_state'))
        train_state = getattr(config.monitor, 'train_state', None) if has_monitor else None

        self._record_config: Dict[str, bool] = {}
        self._local_norm_config: List[str] = []
        self._device_norm_config: List[str] = []
        self._prev_local_norms: Dict[str, float] = {}
        self._hook_handles: List[Any] = []
        self._hook_local_norms: Dict[str, List[float]] = {}
        self._cached_trainable_params: List[Tuple[Any, Tuple[Any, ...]]] = []

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
        self._refresh_trainable_params_cache()
        self._register_grad_hooks()

    @property
    def active(self):
        return bool(self._record_config)

    def should_record(self, key: str) -> bool:
        """Return whether the given training-state metric is enabled."""
        return self._record_config.get(key, False)

    def reset(self):
        super().reset()
        self._prev_local_norms = {}
        self._hook_local_norms = {}

    def flush(self, step: int):
        self._record_pending_hook_norms()
        if self._records:
            device_norm_records = [rec for rec in self._records if "device_norm" in rec]
            if device_norm_records:
                self._records = [
                    rec for rec in self._records if "device_norm" not in rec
                ] + device_norm_records
        super().flush(step)

    def _sync_prev_norms_from_hooks(self):
        """Refresh current-step norm cache from parameter hooks without consuming records."""
        if not self._hook_local_norms:
            return
        if not self._cached_trainable_params:
            return

        for _, trainable_params in self._cached_trainable_params:
            for param in trainable_params:
                if not (self._should_record_norm(param.name, "local_norm") or
                        self._should_record_norm(param.name, "device_norm")):
                    continue
                actual_norms = self._hook_local_norms.get(param.name)
                if actual_norms:
                    self._prev_local_norms[param.name] = actual_norms[-1]

    def _record_pending_hook_norms(self):
        """Emit hook-collected local norms not consumed by trainer-side records."""
        if not self._hook_local_norms:
            return

        if not self._cached_trainable_params:
            self._hook_local_norms = {}
            return

        self._sync_prev_norms_from_hooks()
        if not self._record_config.get("local_norm", False):
            self._hook_local_norms = {}
            return

        max_micro_steps = max((len(norms) for norms in self._hook_local_norms.values()), default=0)
        for micro_step in range(max_micro_steps):
            self._records.append({"micro_step": micro_step + 1})
            for _, trainable_params in self._cached_trainable_params:
                for param in trainable_params:
                    if not self._should_record_norm(param.name, "local_norm"):
                        continue
                    actual_norms = self._hook_local_norms.get(param.name)
                    if not actual_norms or micro_step >= len(actual_norms):
                        continue
                    actual_norm = actual_norms[micro_step]
                    self._prev_local_norms[param.name] = actual_norm
                    self._records.append({"local_norm": f"{param.name}: {actual_norm:10.6f}"})

        self._hook_local_norms = {}

    def set_model(self, model):
        """Bind a model instance to the monitor and refresh parameter hooks."""
        self._remove_grad_hooks()
        super().set_model(model)
        self._refresh_trainable_params_cache()
        self._register_grad_hooks()

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
        from mindformers.pynative.trainer.utils import get_grad
        if not self._cached_trainable_params:
            return
        micro_step = context.get("micro_step", 0)

        if self._record_config.get("local_norm", False):
            self._records.append({"micro_step": micro_step + 1})
        for _, trainable_params in self._cached_trainable_params:
            for param in trainable_params:
                grad = get_grad(param)
                if grad is None:
                    continue
                record_local_norm = self._should_record_norm(param.name, "local_norm")
                record_device_norm = self._should_record_norm(param.name, "device_norm")
                if not record_local_norm and not record_device_norm:
                    continue
                actual_norms = self._hook_local_norms.get(param.name)
                if not actual_norms:
                    continue
                actual_norm = actual_norms.pop(0)
                if not actual_norms:
                    self._hook_local_norms.pop(param.name, None)
                self._prev_local_norms[param.name] = actual_norm
                if record_local_norm:
                    self._records.append({"local_norm": f"{param.name}: {actual_norm:10.6f}"})

    def _record_device_loss(self, value, context):  # pylint: disable=W0613
        self._records.append({"device_loss": self._to_scalar(value)})

    def _record_device_norm(self, value, context):  # pylint: disable=W0613
        self._sync_prev_norms_from_hooks()
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

    def _register_grad_hooks(self):
        """Register backward hooks for parameters monitored by local/device norm."""
        if not (self._record_config.get("local_norm") or self._record_config.get("device_norm")):
            return
        if not self._cached_trainable_params:
            return

        for _, trainable_params in self._cached_trainable_params:
            for param in trainable_params:
                if not (self._should_record_norm(param.name, "local_norm") or
                        self._should_record_norm(param.name, "device_norm")):
                    continue
                if not hasattr(param, "register_hook"):
                    logger.warning(
                        "Parameter %s does not support register_hook, skip hook local_norm.",
                        param.name
                    )
                    continue

                def hook_fn(grad, param_name=param.name):
                    self._hook_local_norms.setdefault(param_name, []).append(self._compute_grad_norm(grad))
                    return grad

                self._hook_handles.append(param.register_hook(hook_fn))

    def _refresh_trainable_params_cache(self):
        """Cache trainable parameters for the currently bound model(s)."""
        self._cached_trainable_params = [
            (model, tuple(model.trainable_params()))
            for model in _iter_models(self._resolve_model())
        ]

    def _remove_grad_hooks(self):
        """Remove previously registered backward hooks."""
        for handle in self._hook_handles:
            if hasattr(handle, "remove"):
                handle.remove()
        self._hook_handles = []

    @staticmethod
    def _compute_grad_norm(grad):
        """Compute the local norm of a gradient tensor."""
        from mindformers.pynative.trainer.utils import _get_grad_factor
        local_grad = grad.to_local() if hasattr(grad, "to_local") else grad
        return float(local_grad.pow(2).sum().sqrt().asnumpy()) / _get_grad_factor(grad)


class MoeMonitor(Monitor):
    """Monitor that prints MoE tokens-per-expert records in Megatron's JSON format."""

    def __init__(self, config, model=None):
        super().__init__(model=model)
        moe_monitor = getattr(getattr(config, "monitor", None), "moe_monitor", None)
        self._interval = getattr(moe_monitor, "save_tokens_per_expert_interval", None)
        self._target_layers = self._parse_target_layers(getattr(moe_monitor, "target_layers", None))
        self._module_cache: List[Tuple[Tuple[str, Optional[int], int], Any]] = self._discover_moe_layers(self._model)
        self._prev_tpe: Dict[Tuple[str, Optional[int], int], Any] = {}
        self._step_tpe_records: Dict[Tuple[str, Optional[int], int], List[Tuple[int, int, int, Any]]] = {}

    def set_model(self, model):
        """Bind a model instance and rebuild the cached MoE layer list."""
        super().set_model(model)
        self._module_cache = self._discover_moe_layers(model)

    @property
    def active(self):
        return self._interval is not None

    def reset(self):
        super().reset()
        self._prev_tpe = {}
        self._step_tpe_records = {}

    def record(self, key: str = None, value: Any = None, context: Optional[Dict[str, Any]] = None):
        if self._interval is None:
            return

        context = context or {}

        if key == "moe_tpe_step_begin":
            self._prev_tpe = {
                meta: self._snapshot_tpe(module)
                for meta, module in self._module_cache
            }
            return

        if key != "moe_tpe":
            return

        global_micro_step = context.get("global_micro_step")
        step_id = context.get("step")
        micro_step = context.get("micro_step")
        if global_micro_step is None:
            return

        for meta, module in self._module_cache:
            current = self._snapshot_tpe(module)
            previous = self._prev_tpe.get(meta)
            if previous is None:
                previous = current.copy()
            delta = current - previous
            if global_micro_step % self._interval == 0:
                self._step_tpe_records.setdefault(meta, []).append((global_micro_step, step_id, micro_step, delta))
            self._prev_tpe[meta] = current

    def flush(self, step: int):
        if not self._step_tpe_records:
            return

        for (block, mtp_idx, layer), microbatches in sorted(self._step_tpe_records.items()):
            for iter_id, step_id, micro_step, microbatch in microbatches:
                record = {
                    "iter": iter_id,
                    "step": step_id,
                    "micro_step": micro_step,
                    "block": block,
                    "layer": layer,
                    "tpe": [self._tensor_to_list(microbatch)],
                }
                if block == "mtp":
                    record = {
                        "iter": iter_id,
                        "step": step_id,
                        "micro_step": micro_step,
                        "block": block,
                        "mtp_idx": mtp_idx,
                        "layer": layer,
                        "tpe": [self._tensor_to_list(microbatch)],
                    }
                logger.info(json.dumps(record))

        self._step_tpe_records = {}
        self._prev_tpe = {}

    def _discover_moe_layers(self, model):
        """Discover MoE layers from the bound model."""
        if model is None:
            return []
        modules = []
        moe_decoder_layers = set()
        for module_name, module in model.cells_and_names():
            if not hasattr(module, "tokens_per_expert"):
                continue
            parsed = _parse_moe_tpe_module_name(module_name)
            if parsed is not None:
                block, _, layer = parsed
                if self._target_layers is not None:
                    if block != "decoder" or layer not in self._target_layers:
                        continue
                if block == "decoder":
                    moe_decoder_layers.add(layer)
                modules.append((parsed, module))
        if self._target_layers is not None:
            for layer in sorted(self._target_layers):
                if layer not in moe_decoder_layers:
                    logger.warning(
                        "decoder.layers.%s is not a MoE layer, no tokens_per_expert.",
                        layer,
                    )
        return modules

    @staticmethod
    def _snapshot_tpe(module) -> List[int]:
        return module.tokens_per_expert.copy()

    @staticmethod
    def _tensor_to_list(value):
        if hasattr(value, "asnumpy"):
            return [int(round(float(v))) for v in value.asnumpy().reshape(-1).tolist()]
        if isinstance(value, list):
            return value
        return [int(round(float(value)))]

    @staticmethod
    def _parse_target_layers(target_layers):
        """Normalize target_layers config to a set of decoder layer ids."""
        if target_layers is None:
            return None
        if isinstance(target_layers, int):
            return set(range(target_layers))
        return set(target_layers)


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
        # flush is called by Trainer after all step-end callbacks complete
    """

    def __init__(self, config, model=None):
        self._monitors: Dict[str, Monitor] = {}
        self._model = model

        train_state_monitor = TrainStateMonitor(config, model=model)
        if train_state_monitor.active:
            self._monitors["train_state"] = train_state_monitor

        moe_monitor = MoeMonitor(config, model=_first_model(model))
        if moe_monitor.active:
            self._monitors["moe_monitor"] = moe_monitor

        tensorboard_monitor = TensorboardMonitor(config, model=model)
        if tensorboard_monitor.configured:
            self._monitors["tensorboard"] = tensorboard_monitor

    @property
    def active(self):
        return bool(self._monitors)

    def should_record(self, key: str) -> bool:
        """Return whether any child monitor is configured for the given key."""
        for monitor in self._monitors.values():
            should_record = getattr(monitor, "should_record", None)
            if callable(should_record) and should_record(key):
                return True
        return False

    def reset(self):
        for m in self._monitors.values():
            m.reset()
        clear_monitor_data_tracker()

    def setup(self, config=None, state=None):
        """Initialize all configured monitor resources."""
        tracker = get_monitor_data_tracker()
        tracker.clear()
        tracker.set_config(_config_to_dict(config))
        for m in self._monitors.values():
            m.setup(config, state)

    def set_model(self, model):
        """Bind a model instance to all child monitors."""
        self._model = model
        for name, monitor in self._monitors.items():
            if name == "moe_monitor":
                monitor.set_model(_first_model(model))
                continue
            monitor.set_model(model)

    def record(self, key: str, value: Any = None, context: Optional[Dict[str, Any]] = None):
        """Forward one metric record to every configured monitor."""
        for m in self._monitors.values():
            m.record(key, value, context)

    def flush(self, step: int):
        """Flush every monitor and clear the shared step-scoped data."""
        try:
            for name, monitor in self._monitors.items():
                try:
                    monitor.flush(step)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.error(
                        "Monitor %s flush failed at step %s: %s",
                        name,
                        step,
                        exc,
                        exc_info=True,
                    )
        finally:
            clear_monitor_data_tracker()

    def close(self):
        """Release all monitor resources."""
        try:
            for name, monitor in self._monitors.items():
                try:
                    monitor.close()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.error(
                        "Monitor %s close failed: %s",
                        name,
                        exc,
                        exc_info=True,
                    )
        finally:
            reset_monitor_data_tracker()
