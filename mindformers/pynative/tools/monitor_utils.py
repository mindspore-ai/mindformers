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
"""Shared, backend-agnostic storage for monitor data."""

from __future__ import annotations

from typing import Any, Dict, Optional


class MonitorDataTracker:
    """Process-wide monitor data store with explicit run and step lifecycles."""

    _instance: Optional["MonitorDataTracker"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._storage = {
                "training_state": {},
                "data": {},
                "config": {},
            }
        return cls._instance

    @classmethod
    def get_instance(cls) -> "MonitorDataTracker":
        """Return the process-wide tracker instance."""
        return cls()

    @property
    def training_state(self) -> Dict[str, Any]:
        """Return state shared by the current step's monitor data."""
        return self._storage["training_state"]

    @property
    def data(self) -> Dict[str, Any]:
        """Return backend-agnostic key-value data for the current step."""
        return self._storage["data"]

    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration retained for the current training run."""
        return self._storage["config"]

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return the complete tracker storage without replacing its identity."""
        return self._storage

    def update_training_state(self, **state: Any) -> None:
        """Update non-None training-state values for the current step."""
        self.training_state.update({key: value for key, value in state.items() if value is not None})

    def save_data(self, name: str, value: Any) -> None:
        """Save one backend-agnostic key-value pair for the current step."""
        if value is None:
            return
        if not name:
            raise ValueError("monitor data name must not be empty")
        if hasattr(value, "detach"):
            value = value.detach()
        self.data[name] = value

    def set_config(self, config: Any) -> None:
        """Replace the configuration retained for the current training run."""
        self.config.clear()
        if config is None:
            return
        if isinstance(config, dict):
            self.config.update(config)
            return
        self.config["value"] = config

    def clear_step(self) -> None:
        """Clear step-scoped state and data while retaining run configuration."""
        self.training_state.clear()
        self.data.clear()

    def clear(self) -> None:
        """Clear all data at the end or beginning of a training run."""
        self.clear_step()
        self.config.clear()


def get_monitor_data_tracker() -> MonitorDataTracker:
    """Return the process-wide monitor data tracker."""
    return MonitorDataTracker.get_instance()


def save_monitor_data(
        name: str,
        value: Any,
        **training_state: Any,
) -> None:
    """Save one data item and its shared training state for the current step."""
    tracker = get_monitor_data_tracker()
    tracker.update_training_state(**training_state)
    tracker.save_data(name, value)


def clear_monitor_data_tracker() -> None:
    """Clear current-step state and data while retaining run configuration."""
    get_monitor_data_tracker().clear_step()


def reset_monitor_data_tracker() -> None:
    """Clear the complete tracker at a training-run boundary."""
    get_monitor_data_tracker().clear()
