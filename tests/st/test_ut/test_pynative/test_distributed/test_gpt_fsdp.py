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
"""Tests for GPT FSDP wrapper boundaries."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from mindformers.pynative.base_models.gpt import parallelize


class _FakeModule:
    """Minimal parameter-free module used to exercise FSDP orchestration."""

    def parameters_and_names(self):
        return iter(())


class _FakeTransformerLayer(_FakeModule):
    """Transformer-layer attributes consumed by the policy builder."""

    def __init__(self, max_logits_val=None):
        self.self_attention = SimpleNamespace(core_attention=SimpleNamespace())
        if max_logits_val is not None:
            self.self_attention.core_attention.max_logits_val = max_logits_val
        self.mlp = SimpleNamespace()


class _FakeMtpLayer(_FakeModule):
    """Outer MTP unit containing one transformer layer."""

    def __init__(self):
        self.transformer_layer = _FakeTransformerLayer()


class _FakeMesh:
    def size(self):
        return 2


class _FakeParameter:
    """Minimal parameter carrying only the shape used by the sync setup."""

    def __init__(self, shape):
        self.shape = shape


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
def test_apply_fsdp_wraps_each_layer_once(monkeypatch):
    """Only decoder layers and outer MTP units are wrapped; their children are not."""
    decoder_max_logits = _FakeParameter((8,))
    mtp_max_logits = _FakeParameter((8,))
    decoder_layer = _FakeTransformerLayer(decoder_max_logits)
    mtp_layer = _FakeMtpLayer()
    mtp_layer.transformer_layer.self_attention.core_attention.max_logits_val = mtp_max_logits
    decoder = SimpleNamespace(layers=[decoder_layer], final_layernorm=None, hc_head=None)
    gpt_model = SimpleNamespace(
        embedding=None,
        decoder=decoder,
        mtp=SimpleNamespace(layers=[mtp_layer]),
        output_layer=None,
    )
    model = _FakeModule()
    parallel_dims = SimpleNamespace(
        dp_replicate_enabled=False,
        ep_enabled=False,
        pp_enabled=False,
        fsdp=2,
        get_mesh=lambda *_: _FakeMesh(),
    )
    parallelism = SimpleNamespace(
        dense_fsdp_shard_size=None,
        reshard_after_forward_policy="default",
        cpu_offload=False,
        disable_gradient_division=False,
    )
    wrapped = []

    monkeypatch.setattr(parallelize, "_unwrap_gptmodel", lambda _: gpt_model)
    monkeypatch.setattr(parallelize, "get_fsdp_reshard_after_forward_policy", lambda *_: False)
    monkeypatch.setattr(parallelize, "MixedPrecisionPolicy", lambda **_: object())
    monkeypatch.setattr(parallelize, "OffloadPolicy", object)
    monkeypatch.setattr(parallelize.ms, "DeviceCtx", lambda *_: nullcontext())
    monkeypatch.setattr(parallelize, "_setup_gpt_prefetch", lambda *_: None)
    monkeypatch.setattr(
        parallelize,
        "fully_shard",
        lambda module, **kwargs: wrapped.append((module, kwargs)),
    )

    parallelize.apply_fsdp(model, parallel_dims, parallelism)

    assert [module for module, _ in wrapped] == [decoder_layer, mtp_layer, model]
    assert mtp_layer.transformer_layer not in [module for module, _ in wrapped]
    assert all(call[1]["comm_fusion"] is False for call in wrapped)
    expected_ignored = {decoder_max_logits, mtp_max_logits}
    assert wrapped[0][1]["ignored_params"] == expected_ignored
    assert wrapped[1][1]["ignored_params"] == expected_ignored
    assert wrapped[2][1]["ignored_params"] == expected_ignored
    assert decoder_max_logits not in wrapped[0][1]["replicate_params"]
    assert mtp_max_logits not in wrapped[1][1]["replicate_params"]


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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_qk_clip_reduce_group_skips_collective_for_pure_pp(monkeypatch):
    """Pure PP has a size-one loss mesh even though the world has many ranks."""
    calls = []
    gpt_model = SimpleNamespace(
        set_qk_clip_reduce_group=lambda group, size: calls.append((group, size))
    )
    parallel_dims = SimpleNamespace(get_optional_mesh=lambda name: None)

    monkeypatch.setattr(parallelize, "_unwrap_gptmodel", lambda _: gpt_model)

    parallelize._setup_qk_clip_reduce_group(object(), parallel_dims)

    assert calls == [(None, 1)]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_qk_clip_reduce_group_uses_loss_mesh(monkeypatch):
    """DP/CP QK-clip synchronization uses the exact loss-mesh group."""
    calls = []
    gpt_model = SimpleNamespace(
        set_qk_clip_reduce_group=lambda group, size: calls.append((group, size))
    )
    loss_mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: "loss_group")
    parallel_dims = SimpleNamespace(
        get_optional_mesh=lambda name: loss_mesh if name == "loss" else None
    )

    monkeypatch.setattr(parallelize, "_unwrap_gptmodel", lambda _: gpt_model)

    parallelize._setup_qk_clip_reduce_group(object(), parallel_dims)

    assert calls == [("loss_group", 2)]
