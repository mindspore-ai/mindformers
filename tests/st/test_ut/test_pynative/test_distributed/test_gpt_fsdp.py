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
"""Tests for GPT FSDP parameter grouping."""

from contextlib import nullcontext

import pytest

from hyper_parallel.core.dtensor.placement_types import Shard

from mindformers.pynative.base_models.gpt import parallelize


class _FakeParameter:
    """Minimal parameter carrying only the shape used by the sharding plan."""

    def __init__(self, shape):
        self.shape = shape


class _FakeLinear:
    """Minimal Linear-like object with one directly owned weight."""

    def __init__(self, shape=(4, 8)):
        self.weight = _FakeParameter(shape)

    def parameters_and_names(self, expand=True):
        assert expand is False
        return [("weight", self.weight)]


class _FakeOwner:
    """Minimal mHC-like owner with a Linear and a scalar gate parameter."""

    def __init__(self):
        self.proj = _FakeLinear()
        self.scale = _FakeParameter((1,))

    def parameters_and_names(self, expand=True):
        assert expand is False
        return [("scale", self.scale)]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_fp32_linear_and_owner_use_separate_fsdp_groups(monkeypatch):
    """The FP32 projection is wrapped before its replicated scalar owner."""
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    monkeypatch.setattr(parallelize, "fully_shard", fake_fully_shard)
    monkeypatch.setattr(parallelize.ms, "DeviceCtx", lambda *_: nullcontext())

    owner = _FakeOwner()
    parallelize._wrap_fp32_linear_owner(  # pylint: disable=protected-access
        owner, "proj", {}, False, shard_size=2
    )

    assert [module for module, _ in calls] == [owner.proj, owner]
    assert calls[0][1]["replicate_params"] == []
    assert calls[0][1]["shard_placement_fn"](owner.proj.weight) == Shard(1)
    assert calls[1][1]["replicate_params"] == [owner.scale]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_fused_fp32_linear_is_managed_by_owner_hook(monkeypatch):
    """Fused mHC bypasses Linear.construct, so only its callable owner is wrapped."""
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    monkeypatch.setattr(parallelize, "fully_shard", fake_fully_shard)
    monkeypatch.setattr(parallelize.ms, "DeviceCtx", lambda *_: nullcontext())

    owner = _FakeOwner()
    parallelize._wrap_fp32_linear_owner(  # pylint: disable=protected-access
        owner, "proj", {}, False, shard_size=2, wrap_linear=False
    )

    assert [module for module, _ in calls] == [owner]
    assert calls[0][1]["replicate_params"] == [owner.scale]
    assert calls[0][1]["shard_placement_fn"](owner.proj.weight) == Shard(1)
