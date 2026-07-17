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


class _FakeTensor:
    """Small ownership-flag tensor used to isolate collective setup logic."""

    def __init__(self, value, dtype=None):
        del dtype
        self.value = list(value)

    def asnumpy(self):
        return self.value


class _FakeEmbeddingPart:
    """Minimal pipeline model part with one input embedding parameter."""

    def __init__(self):
        self.weight = _FakeParameter((8, 4))

    def parameters_and_names(self):
        return [("model.embedding.word_embeddings.weight", self.weight)]


class _FakePpMesh:
    """Two-rank PP line used by MTP embedding synchronization setup."""

    @staticmethod
    def get_rank_list_along_axis(axis):
        assert axis == "pp"
        return [0, 4]

    @staticmethod
    def get_group():
        return "pp_group"


class _FakeParallelDims:
    """Parallel dimensions with pipeline parallelism enabled."""

    pp_enabled = True

    @staticmethod
    def get_mesh(axis):
        assert axis == "pp"
        return _FakePpMesh()


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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_mtp_embedding_sync_uses_canonical_pp_rank(monkeypatch):
    """MTP embedding tags carry the group and source needed after delayed init."""
    part = _FakeEmbeddingPart()
    config = type("Config", (), {"mtp_num_layers": 1})()
    gpt_model = type(
        "GptModel", (), {"get_gpt_transformer_config": lambda self: config}
    )()
    created_groups = []

    def fake_all_gather(outputs, inputs, group):
        _ = inputs
        assert group == "pp_group"
        for output in outputs:
            output.value[0] = 1

    def fake_new_group(ranks):
        created_groups.append(ranks)
        return "mtp_embedding_group"

    monkeypatch.setattr(parallelize, "Tensor", _FakeTensor)
    monkeypatch.setattr(parallelize, "all_gather", fake_all_gather)
    monkeypatch.setattr(parallelize, "get_rank", lambda: 4)
    monkeypatch.setattr(parallelize, "new_group", fake_new_group)
    monkeypatch.setattr(parallelize, "_unwrap_gptmodel", lambda _part: gpt_model)

    parallelize._setup_mtp_embedding_grad_sync([part], _FakeParallelDims())

    assert created_groups == [[0, 4]]
    assert part.weight._embedding_grad_sync_group == "mtp_embedding_group"
    assert part.weight._embedding_grad_sync_size == 2
    assert part.weight._embedding_sync_src_rank == 0
