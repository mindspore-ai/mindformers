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
"""Test pynative trainer utils."""

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, Parameter, nn

from mindformers.pynative.config.config import OptimizerConfig
from mindformers.pynative.trainer import utils as trainer_utils
from mindformers.pynative.trainer.utils import get_param_groups


class ParamGroupNet(nn.Cell):
    """Tiny network for optimizer parameter group tests."""

    def __init__(self):
        super().__init__()
        # 2D weight — default goes to weight_decay group
        self.weight = Parameter(
            Tensor(np.ones((2, 2)), ms.float32), name="block.weight"
        )
        # 1D + .bias suffix — default goes to no_weight_decay
        self.bias = Parameter(Tensor(np.ones((2,)), ms.float32), name="block.bias")
        # 1D — default goes to no_weight_decay via 1d_param rule (not the name)
        self.norm = Parameter(Tensor(np.ones((2,)), ms.float32), name="norm.weight")
        # 2D weight — default goes to weight_decay group
        self.special = Parameter(
            Tensor(np.ones((2, 2)), ms.float32), name="special.weight"
        )


class ParamGroupNetWithNoWd(ParamGroupNet):
    """Network exposing the optional no_weight_decay interfaces."""

    def no_weight_decay(self):
        return ["block.weight"]

    def no_weight_decay_keywords(self):
        return ["special"]


class _FakeLocalParameter:
    """DTensor-like parameter exposing the local shard used by collectives."""

    def __init__(self):
        self.local = object()
        self._embedding_grad_sync_group = "mtp_embedding_group"
        self._embedding_sync_src_rank = 0

    def to_local(self):
        return self.local


class _FakeModelPart:
    """Minimal model part yielding parameters for post-init synchronization."""

    def __init__(self, parameters):
        self.parameters = parameters

    def parameters_and_names(self):
        return iter(self.parameters)


def _param_weight_decay_map(param_groups):
    """Map parameter names to their assigned weight decay."""
    result = {}
    for group in param_groups:
        for param in group["params"]:
            result[param.name] = group["weight_decay"]
    return result


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_sync_mtp_embedding_weights_after_init(monkeypatch):
    """The canonical PP rank is broadcast into each distinct tagged local shard."""
    tagged = _FakeLocalParameter()
    untagged = object()
    model_parts = [
        _FakeModelPart([("embedding.word_embeddings.weight", tagged)]),
        # Repeating the same object must not launch the collective twice.
        _FakeModelPart([
            ("embedding.word_embeddings.weight", tagged),
            ("decoder.weight", untagged),
        ]),
    ]
    calls = []

    def fake_broadcast(tensor, src, group):
        calls.append((tensor, src, group))

    monkeypatch.setattr(trainer_utils, "broadcast", fake_broadcast)

    synced = trainer_utils._sync_mtp_embedding_weights_after_init(model_parts)

    assert synced == 1
    assert calls == [(tagged.local, 0, "mtp_embedding_group")]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_param_groups_default_rules():
    """Test default weight decay grouping rules."""
    decay_map = _param_weight_decay_map(get_param_groups(ParamGroupNet(), 0.1))

    assert decay_map["block.weight"] == 0.1
    assert decay_map["special.weight"] == 0.1
    assert decay_map["block.bias"] == 0.0
    assert decay_map["norm.weight"] == 0.0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_param_groups_custom_exclude_rule():
    """Test custom rule can force weight decay off."""
    decay_map = _param_weight_decay_map(
        get_param_groups(
            ParamGroupNet(),
            0.1,
            weight_decay_exclude=["special.*"],
        )
    )

    assert decay_map["block.weight"] == 0.1
    assert decay_map["special.weight"] == 0.0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_param_groups_custom_include_overrides_default_no_wd():
    """Test custom include rule can force weight decay on."""
    decay_map = _param_weight_decay_map(
        get_param_groups(
            ParamGroupNet(),
            0.1,
            weight_decay_include=["block.bias", "norm.weight"],
        )
    )

    assert decay_map["block.bias"] == 0.1
    assert decay_map["norm.weight"] == 0.1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_param_groups_include_overrides_exclude():
    """Include rule must win when a param matches both include and exclude."""
    decay_map = _param_weight_decay_map(
        get_param_groups(
            ParamGroupNet(),
            0.1,
            weight_decay_include=["special.weight"],
            weight_decay_exclude=["special.*"],
        )
    )

    assert decay_map["special.weight"] == 0.1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_param_groups_substring_rule_matches():
    """A rule that is a plain substring of a param name should match."""
    decay_map = _param_weight_decay_map(
        get_param_groups(
            ParamGroupNet(),
            0.1,
            # "pecial" is neither exact nor a glob — pure substring match.
            weight_decay_exclude=["pecial"],
        )
    )

    assert decay_map["special.weight"] == 0.0
    assert decay_map["block.weight"] == 0.1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_param_groups_model_no_weight_decay_interfaces():
    """Cover the no_weight_decay() and no_weight_decay_keywords() default paths."""
    decay_map = _param_weight_decay_map(
        get_param_groups(ParamGroupNetWithNoWd(), 0.1)
    )

    # via model.no_weight_decay()
    assert decay_map["block.weight"] == 0.0
    # via model.no_weight_decay_keywords()
    assert decay_map["special.weight"] == 0.0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_param_groups_rejects_bare_string_rule():
    """A bare string is no longer a valid rules container."""
    with pytest.raises(TypeError):
        get_param_groups(
            ParamGroupNet(), 0.1, weight_decay_include="block.weight"
        )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_param_groups_rejects_non_string_items():
    """Non-string items inside the rules list should raise."""
    with pytest.raises(TypeError):
        get_param_groups(ParamGroupNet(), 0.1, weight_decay_include=[123])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_optimizer_config_weight_decay_rules():
    """Test optimizer weight decay rules are loaded from config dict."""
    config = OptimizerConfig.from_dict(
        {
            "type": "AdamW",
            "weight_decay": 0.1,
            "weight_decay_include": ["norm.weight"],
            "weight_decay_exclude": ["special.*"],
        }
    )

    assert config.weight_decay_include == ["norm.weight"]
    assert config.weight_decay_exclude == ["special.*"]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_global_grad_norm_propagates_parameter_replica_count():
    """A replica count tagged on a parameter must reach its FSDP main_grad."""
    parameter = Parameter(Tensor(np.ones((2,)), ms.float32), name="weight")
    parameter.main_grad = Tensor(np.asarray([3.0, 4.0], dtype=np.float32))
    setattr(parameter, "_grad_norm_replica_count", 2)

    calculate_global_grad_norm = getattr(trainer_utils, "_calculate_global_grad_norm")
    global_norm, _ = calculate_global_grad_norm(
        [parameter], enable_parallel=False, max_norm=1.0e9
    )

    assert getattr(parameter.main_grad, "_pp_replica_count") == 2
    np.testing.assert_allclose(
        global_norm.asnumpy(), np.sqrt((3.0 ** 2 + 4.0 ** 2) / 2.0), rtol=1.0e-6
    )
