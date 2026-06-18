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
"""Test module for testing pynative monitor with CPU."""

import pytest

from mindformers.pynative.config.config import (
    TrainConfig,
    TrainStateConfig,
    MonitorConfig,
    MoeMonitorConfig,
)
from mindformers.pynative.tools.monitor import Monitor, TrainStateMonitor, MoeMonitor, MonitorGroup


def _make_config(local_loss=False, local_norm="", device_loss=False, device_norm=False, moe_interval=None):
    """Helper to build a config object with given train_state settings."""
    train_state = TrainStateConfig(
        local_loss=local_loss,
        local_norm=local_norm,
        device_loss=device_loss,
        device_norm=device_norm,
    )
    monitor = MonitorConfig(
        train_state=train_state,
        moe_monitor=MoeMonitorConfig(save_tokens_per_expert_interval=moe_interval),
    )
    config = TrainConfig(monitor=monitor)
    return config


class _FakeHookHandle:
    """Fake hook handle used by monitor unit tests."""

    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class _FakeParam:
    """Fake trainable parameter with register_hook support."""

    def __init__(self, name, grad=None):
        self.name = name
        self.grad = grad
        self.hooks = []

    def register_hook(self, hook):
        self.hooks.append(hook)
        return _FakeHookHandle()


class _FakeModel:
    """Fake model that exposes trainable_params."""

    def __init__(self, params):
        self._params = params

    def trainable_params(self):
        return self._params


class TestMonitorConfigRecognition:
    """Test that TrainStateMonitor correctly reads config."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_all_enabled(self):
        """All train_state options enabled should set all _record_config keys to True."""
        config = _make_config(local_loss=True, local_norm="embedding",
                              device_loss=True, device_norm="mlp")
        monitor = TrainStateMonitor(config)
        assert monitor._record_config["local_loss"] is True
        assert monitor._record_config["local_norm"] is True
        assert monitor._record_config["device_loss"] is True
        assert monitor._record_config["device_norm"] is True
        assert monitor._local_norm_config == ["embedding"]
        assert monitor._device_norm_config == ["mlp"]
        assert monitor.active is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_all_disabled(self):
        """All train_state options disabled should result in inactive monitor."""
        config = _make_config()
        monitor = TrainStateMonitor(config)
        assert monitor._record_config == {}
        assert monitor.active is False

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_norm_monitor_registers_hooks_by_default(self):
        """Norm monitor should register hooks when a model is bound."""
        param = _FakeParam("embedding.weight", grad=100.0)
        model = _FakeModel([param])
        config = _make_config(local_norm="embedding")
        monitor = TrainStateMonitor(config, model=model)
        assert monitor._record_config["local_norm"] is True
        assert monitor._local_norm_config == ["embedding"]
        assert len(param.hooks) == 1

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_norm_config_supported_types(self):
        """local_norm and device_norm should accept bool, str and list[str]."""
        cases = [
            (True, True, True, True, [], []),
            ("embedding", "mlp", True, True, ["embedding"], ["mlp"]),
            (["embedding", "attention"], ["mlp", "attention"], True, True,
             ["embedding", "attention"], ["mlp", "attention"]),
            (False, False, False, False, [], []),
            ("", "", False, False, [], []),
            ([], [], False, False, [], []),
        ]
        for local_norm, device_norm, local_active, device_active, local_config, device_config in cases:
            config = _make_config(local_norm=local_norm, device_norm=device_norm)
            monitor = TrainStateMonitor(config)
            assert monitor._record_config.get("local_norm", False) is local_active
            assert monitor._record_config.get("device_norm", False) is device_active
            assert monitor._local_norm_config == local_config
            assert monitor._device_norm_config == device_config

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_norm_config_invalid_type(self):
        """local_norm and device_norm should reject unsupported types."""
        cases = [
            {"local_norm": 1},
            {"local_norm": ["embedding", 1]},
            {"device_norm": 1},
            {"device_norm": ["mlp", 1]},
        ]
        for case in cases:
            with pytest.raises(TypeError):
                TrainStateMonitor(_make_config(**case))

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_no_monitor_section(self):
        """Config without monitor section should result in inactive monitor."""
        config = TrainConfig()
        monitor = TrainStateMonitor(config)
        assert monitor.active is False


class TestMonitorRecordEffect:
    """Test that record only records when config is enabled."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_record_local_loss(self):
        """record local_loss when enabled should add records."""
        config = _make_config(local_loss=True)
        monitor = TrainStateMonitor(config)
        monitor.record("local_loss", 1.0, {"micro_step": 0})
        assert len(monitor._records) == 1
        assert "local_loss" in monitor._records[0]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_record_device_loss(self):
        """record device_loss when enabled should add records."""
        config = _make_config(device_loss=True)
        monitor = TrainStateMonitor(config)
        monitor.record("device_loss", 2.0)
        assert len(monitor._records) == 1
        assert "device_loss" in monitor._records[0]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_record_disabled_key(self):
        """record with disabled key should not add records."""
        config = _make_config()
        monitor = TrainStateMonitor(config)
        monitor.record("local_loss", 1.0, {"micro_step": 0})
        monitor.record("device_loss", 2.0)
        monitor.record("device_norm")
        assert monitor._records == []

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_hook_local_norm_uses_hook_value(self):
        """hook mode should record the norm captured by register_hook."""
        param = _FakeParam("embedding.weight", grad=100.0)
        model = _FakeModel([param])
        config = _make_config(local_norm="embedding")
        monitor = TrainStateMonitor(config, model=model)
        monitor._compute_grad_norm = float

        assert len(param.hooks) == 1
        assert param.hooks[0](3.5) == 3.5
        monitor.record("local_norm", context={"micro_step": 0})

        assert monitor._records == [
            {"micro_step": 1},
            {"local_norm": "embedding.weight:   3.500000"},
        ]


class TestMonitorReset:
    """Test that reset clears records and state."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reset_clears_all(self):
        """reset should clear records and prev_local_norms."""
        config = _make_config(local_loss=True, local_norm="embedding")
        monitor = TrainStateMonitor(config)
        monitor.record("local_loss", 1.0, {"micro_step": 0})
        monitor._prev_local_norms = {"param1": 1.0}
        monitor.reset()
        assert monitor._records == []
        assert monitor._prev_local_norms == {}


class TestMonitorGroup:
    """Test MonitorGroup delegation."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_group_active(self):
        """MonitorGroup should be active when train_state config is set."""
        config = _make_config(local_loss=True)
        group = MonitorGroup(config)
        assert group.active is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_group_inactive(self):
        """MonitorGroup should be inactive when no train_state config is set."""
        config = _make_config()
        group = MonitorGroup(config)
        assert group.active is False

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_group_delegates(self):
        """MonitorGroup should delegate record/reset/flush to TrainStateMonitor."""
        config = _make_config(local_loss=True)
        group = MonitorGroup(config)
        group.record("local_loss", 1.0, {"micro_step": 0})
        train_state_monitor = group._monitors["train_state"]
        assert len(train_state_monitor._records) == 1
        group.reset()
        assert train_state_monitor._records == []


class TestMoeMonitorConfig:
    """Test MoE monitor config recognition."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_moe_monitor_none_disabled(self):
        """save_tokens_per_expert_interval=None should disable MoeMonitor."""
        config = _make_config(moe_interval=None)
        group = MonitorGroup(config)
        assert "moe_monitor" not in group._monitors
        assert group.active is False

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_moe_monitor_one_enabled(self):
        """save_tokens_per_expert_interval=1 should enable MoeMonitor."""
        config = _make_config(moe_interval=1)
        group = MonitorGroup(config)
        assert "moe_monitor" in group._monitors
        assert isinstance(group._monitors["moe_monitor"], MoeMonitor)
        assert group._monitors["moe_monitor"].active is True
        assert group.active is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_moe_monitor_invalid_interval(self):
        """MoeMonitorConfig should reject bool and non-positive interval values."""
        invalid_cases = [False, 0, -1]
        for value in invalid_cases:
            with pytest.raises((TypeError, ValueError)):
                MoeMonitorConfig(save_tokens_per_expert_interval=value)


class TestMonitorBaseClass:
    """Test Monitor base class."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_cannot_instantiate_base_class(self):
        """Monitor is ABC and should not be instantiated directly."""

        class _ConcreteMonitor(Monitor):  # pylint: disable=W0612
            def record(self, value=None, context=None):
                pass

        _ConcreteMonitor()
        with pytest.raises(TypeError):
            Monitor()  # pylint: disable=E0110

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_to_scalar(self):
        """_to_scalar should convert numeric types to float."""
        assert Monitor._to_scalar(1.5) == 1.5
        assert Monitor._to_scalar(3) == 3.0


class TestYamlConfigLoading:
    """Test loading monitor config from yaml file."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_from_yaml(self):
        """Monitor config should be correctly loaded from yaml."""
        import os
        yaml_path = os.path.join(os.path.dirname(__file__), "monitor_test.yaml")
        config = TrainConfig.load_from_yaml(yaml_path)
        monitor = TrainStateMonitor(config)
        assert monitor._record_config.get("local_loss") is True
        assert monitor._record_config.get("local_norm") is True
        assert monitor._record_config.get("device_loss") is True
        assert monitor._record_config.get("device_norm") is True
        assert monitor._local_norm_config == ["embedding"]
        assert monitor._device_norm_config == []
