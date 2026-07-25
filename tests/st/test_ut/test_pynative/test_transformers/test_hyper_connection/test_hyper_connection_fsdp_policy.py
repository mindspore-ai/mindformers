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
"""Tests for configurable transformer-layer and expert FSDP policies."""

import pytest

from mindformers.pynative.base_models.gpt.parallelize import (
    _build_expert_fsdp_policy,
    _build_fsdp_policy,
    _build_hc_head_fsdp_policy,
    _build_layer_fsdp_policy,
    _build_mtp_fsdp_policy,
)


class _FakeParam:
    """Minimal parameter-like object used by the placement policy helpers."""

    def __init__(self, shape):
        self.shape = shape


class _FakeModule:
    """Minimal recursive parameter container."""

    def __init__(self, params):
        self._params = params

    def parameters_and_names(self):
        return iter(self._params.items())


class _FakeLayer(_FakeModule):
    """Minimal transformer-layer shape required by the policy builder."""

    def __init__(self, params):
        super().__init__(params)
        self.self_attention = type("SelfAttention", (), {})()
        self.self_attention.core_attention = type("CoreAttention", (), {})()
        self.mlp = type("MLP", (), {})()


class _FakeMtpLayer(_FakeModule):
    """Minimal outer MTP unit with an inner transformer layer."""

    def __init__(self, params):
        super().__init__(params)
        self.transformer_layer = _FakeLayer({})


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mhc_uses_layer_fsdp_policy():
    """mHC matrices shard on dim 1 while tiny gates remain replicated."""
    matrix = _FakeParam((10, 64))
    rms_weight = _FakeParam((64,))
    alpha_pre = _FakeParam((1,))
    alpha_post = _FakeParam((1,))
    alpha_res = _FakeParam((1,))
    bias = _FakeParam((24,))
    params = {
        "mapping_proj.weight": matrix,
        "rms_weight": rms_weight,
        "alpha_pre": alpha_pre,
        "alpha_post": alpha_post,
        "alpha_res": alpha_res,
        "bias": bias,
    }
    layer = _FakeLayer({f"attn_hc.{name}": param for name, param in params.items()})
    shard_plan, replicate_params = _build_layer_fsdp_policy(layer, 8)

    assert shard_plan(matrix).dim == 1
    assert shard_plan(rms_weight) is None
    assert matrix not in replicate_params
    assert set(map(id, replicate_params)) == {
        id(alpha_pre), id(alpha_post), id(alpha_res), id(bias)
    }


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_layer_fsdp_policy_accepts_custom_rules():
    """Custom rules configure special shard dimensions and replication."""
    matrix = _FakeParam((10, 64))
    bias = _FakeParam((16,))
    layer = _FakeLayer({"special.weight": matrix, "special.bias": bias})
    rules = {
        "special.weight": {"shard_dim": -1},
        "special.bias": {"replicate": True},
    }

    shard_plan, replicate_params = _build_fsdp_policy(layer, 8, rules)

    assert shard_plan(matrix).dim == -1
    assert replicate_params == [bias]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_expert_fsdp_policy_prefers_complete_experts():
    """E-FSDP prefers dim 0 so every shard keeps complete expert matrices."""
    weight1 = _FakeParam((4, 64, 256))
    weight2 = _FakeParam((4, 128, 64))
    experts = _FakeModule({"weight1": weight1, "weight2": weight2})

    shard_plan, replicate_params = _build_expert_fsdp_policy(experts, 2)

    assert shard_plan(weight1).dim == 0
    assert shard_plan(weight2).dim == 0
    assert replicate_params == []


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_expert_fsdp_policy_falls_back_to_matrix_dimension():
    """E-FSDP uses dim 1 only when the local expert dimension cannot be evenly sharded."""
    weight1 = _FakeParam((3, 64, 256))
    weight2 = _FakeParam((3, 128, 64))
    experts = _FakeModule({"weight1": weight1, "weight2": weight2})

    shard_plan, replicate_params = _build_expert_fsdp_policy(experts, 2)

    assert shard_plan(weight1).dim == 1
    assert shard_plan(weight2).dim == 1
    assert replicate_params == []


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_expert_fsdp_policy_replicates_when_no_preferred_dimension_is_divisible():
    """E-FSDP replicates a weight when neither dim 0 nor dim 1 can be evenly sharded."""
    weight = _FakeParam((3, 7, 256))
    experts = _FakeModule({"weight1": weight})

    shard_plan, replicate_params = _build_expert_fsdp_policy(experts, 2)

    assert shard_plan is None
    assert replicate_params == [weight]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_explicit_shard_rule_falls_back_to_replicate():
    """An uneven explicit shard rule must not also return a shard placement."""
    matrix = _FakeParam((4, 7, 256))
    module = _FakeModule({"weight": matrix})

    shard_plan, replicate_params = _build_fsdp_policy(
        module, 4, {"weight": {"shard_dim": 1}}
    )

    assert shard_plan is None
    assert replicate_params == [matrix]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mtp_policy_covers_inner_layer_and_outer_hc_head():
    """The outer MTP wrapper owns both inner mHC and hc_head special placements."""
    layer_weight = _FakeParam((10, 64))
    head_weight = _FakeParam((4, 64))
    head_base = _FakeParam((4,))
    mtp_layer = _FakeMtpLayer({
        "transformer_layer.attn_hc.mapping_proj.weight": layer_weight,
        "hc_head.hc_fn.weight": head_weight,
        "hc_head.hc_base": head_base,
    })

    shard_plan, replicate_params = _build_mtp_fsdp_policy(mtp_layer, 8)

    assert shard_plan(layer_weight).dim == 1
    assert shard_plan(head_weight).dim == 1
    assert replicate_params == [head_base]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_root_hc_head_policy_uses_root_wrapper():
    """Decoder hc_head placements can be passed to root without another wrapper."""
    weight = _FakeParam((4, 64))
    base = _FakeParam((4,))
    scale = _FakeParam((1,))
    hc_head = _FakeModule({"hc_fn.weight": weight, "hc_base": base, "hc_scale": scale})

    shard_plan, replicate_params = _build_hc_head_fsdp_policy(hc_head, 8)

    assert shard_plan(weight).dim == 1
    assert replicate_params == [base, scale]
