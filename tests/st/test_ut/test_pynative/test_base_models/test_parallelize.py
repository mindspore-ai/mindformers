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
"""Tests for GPT model parallelization helpers."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from mindformers.pynative.base_models.gpt import parallelize


def _build_moe_model():
    """Build the minimal model tree consumed by the MoE parallelize helpers."""
    layer = SimpleNamespace(mlp=SimpleNamespace(experts=object()))
    config = SimpleNamespace(moe_permute_fusion=False)
    inner_model = SimpleNamespace(
        config=config,
        decoder=SimpleNamespace(layers=[layer]),
        mtp=None,
    )
    return SimpleNamespace(model=inner_model)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("dispatcher", ["alltoall", "alltoall_deredundancy"])
def test_apply_moe_ep_tp_propagates_use_safe_tokens(monkeypatch, dispatcher):
    """The safe-token setting reaches both PyNative EP dispatchers."""
    captured = {}

    def _capture_strategy(_, **kwargs):
        captured["use_safe_tokens"] = kwargs["use_safe_tokens"]
        return object()

    monkeypatch.setattr(parallelize, "ExpertParallel", _capture_strategy)
    monkeypatch.setattr(parallelize, "DeredundancyExpertParallel", _capture_strategy)
    monkeypatch.setattr(parallelize, "set_comm_ops_inplace", lambda _: None)
    monkeypatch.setattr(parallelize, "parallelize_module", lambda **_: None)

    parallelize.apply_moe_ep_tp(
        _build_moe_model(),
        ep_mesh=object(),
        moe_token_dispatcher_type=dispatcher,
        use_safe_tokens=False,
    )

    assert captured["use_safe_tokens"] is False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_moe_ep_overlap_tp_propagates_use_safe_tokens(monkeypatch):
    """The overlap EP strategy receives the same safe-token config."""
    captured = {}

    def _capture_strategy(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(parallelize, "OverlapExpertParallel", _capture_strategy)
    monkeypatch.setattr(parallelize, "parallelize_module", lambda **_: None)
    monkeypatch.setattr(parallelize.ms, "DeviceCtx", lambda *_: nullcontext())

    parallelize.apply_moe_ep_overlap_tp(
        _build_moe_model(),
        overlap=SimpleNamespace(coordinator=object()),
        ep_mesh=object(),
        use_safe_tokens=False,
    )

    assert captured["use_safe_tokens"] is False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tag_dsv4_tp_replicated_grad_norm_params_excludes_sharded_and_indexer_weights():
    """Only already-global full-attention gradients are TP replica-counted."""
    parameters = {
        "linear_proj.weight": SimpleNamespace(),
        "linear_q_up_proj.weight": SimpleNamespace(),
        "core_attention.attn_sink": SimpleNamespace(),
        "core_attention.indexer.weight": SimpleNamespace(),
    }
    unrelated = SimpleNamespace()

    marked_attention = SimpleNamespace(
        _tp_full_attention_replica_count=2,
        parameters_and_names=parameters.items,
    )
    unmarked_cell = SimpleNamespace(
        parameters_and_names=lambda: (("weight", unrelated),)
    )
    model = SimpleNamespace(
        cells_and_names=lambda: (
            ("self_attention", marked_attention),
            ("unrelated", unmarked_cell),
        )
    )
    tag_replicated_params = getattr(
        parallelize, "_tag_dsv4_tp_replicated_grad_norm_params"
    )
    tag_replicated_params(model)

    assert getattr(parameters["linear_proj.weight"], "_grad_norm_replica_count") == 2
    assert getattr(parameters["core_attention.attn_sink"], "_grad_norm_replica_count") == 2
    assert not hasattr(parameters["linear_q_up_proj.weight"], "_grad_norm_replica_count")
    assert not hasattr(parameters["core_attention.indexer.weight"], "_grad_norm_replica_count")
    assert not hasattr(unrelated, "_grad_norm_replica_count")
