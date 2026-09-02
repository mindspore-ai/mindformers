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
"""Tests for pynative GPT model-level behavior."""

from types import SimpleNamespace
import numpy as np
import pytest
from mindspore import Tensor, dtype

from mindformers.pynative.base_models.gpt import gpt_model as gpt_model_module
from mindformers.pynative.base_models.gpt.gpt_model import GPTModel


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_qk_clip_caches_strict_device_count():
    """The clip pass reuses its own synced logits for a strict-``>`` device count."""
    stacked = Tensor([[99.0, 100.0], [101.0, 150.0]], dtype=dtype.float32)
    applied = []

    def make_layer(row):
        core = SimpleNamespace(max_logits_val=stacked[row])
        layer = SimpleNamespace(core_attention=core)
        layer.apply_qk_clip = lambda scales, fp32, _l=layer: applied.append((_l, scales))
        return layer

    layers = [make_layer(0), make_layer(1)]
    model = SimpleNamespace(
        config=SimpleNamespace(multi_latent_attention=True),
        _iter_self_attentions=lambda: [(i, "", l) for i, l in enumerate(layers)],
        _all_reduce_max_logits=lambda t: stacked,
        _qk_clip_count_cache=[None],
    )

    GPTModel.apply_qk_clip_scaling(model, Tensor([100.0], dtype=dtype.float32), {})

    # 100.0 is not strictly greater than the threshold; 101.0 and 150.0 are.
    count = GPTModel.take_qk_clip_count(model)
    assert np.array_equal(count.asnumpy(), np.array([2], np.int32))
    assert GPTModel.take_qk_clip_count(model) is None
    # every tracked layer got its own row of scales, in order
    assert [l for l, _ in applied] == layers


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_use_rotary_position_ids_updates_model_config_and_rope():
    """The model-level switch is the single source for rotary position behavior."""
    model = SimpleNamespace(
        config=SimpleNamespace(use_rotary_position_ids=False),
        use_rotary_position_ids=False,
        rotary_pos_emb=SimpleNamespace(use_rotary_position_ids=False),
    )

    GPTModel.set_use_rotary_position_ids(model, True)

    assert model.use_rotary_position_ids is True
    assert model.config.use_rotary_position_ids is True
    assert model.rotary_pos_emb.use_rotary_position_ids is True


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_explicit_rotary_positions_require_position_ids():
    """An enabled explicit-position path must not silently fall back to arange."""
    model = SimpleNamespace(use_rotary_position_ids=True)

    with pytest.raises(ValueError, match="position_ids must be provided"):
        GPTModel.construct(model, input_ids=None, position_ids=None)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fused_rope_rejects_per_batch_position_ids():
    """Fused RoPE fails early when explicit positions have a non-broadcast batch dimension."""
    model = SimpleNamespace(
        use_rotary_position_ids=True,
        config=SimpleNamespace(apply_rope_fusion=True),
    )
    position_ids = SimpleNamespace(shape=(2, 4))

    with pytest.raises(ValueError, match="fused RoPE does not support"):
        GPTModel.construct(model, input_ids=None, position_ids=position_ids)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_qk_clip_explicit_size_one_skips_world_all_reduce(monkeypatch):
    """A pure-PP loss mesh has size one and must not use the world group."""
    model = SimpleNamespace()
    tensor = object()

    monkeypatch.setattr(gpt_model_module, "get_world_size", lambda *_: 16)

    def fail_all_reduce(*_args, **_kwargs):
        pytest.fail("size-one QK-clip synchronization must not call all_reduce")

    monkeypatch.setattr(gpt_model_module, "all_reduce", fail_all_reduce)

    GPTModel.set_qk_clip_reduce_group(model, None, 1)

    assert GPTModel._all_reduce_max_logits(model, tensor) is tensor


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_qk_clip_uses_exact_configured_loss_group(monkeypatch):
    """A multi-rank loss mesh must pass its group to the max reduction."""
    model = SimpleNamespace()
    tensor = object()
    reduced = object()
    calls = []

    def fake_all_reduce(value, *, op, group=None):
        calls.append((value, op, group))
        return reduced

    monkeypatch.setattr(gpt_model_module, "all_reduce", fake_all_reduce)

    GPTModel.set_qk_clip_reduce_group(model, "loss_group", 16)

    assert GPTModel._all_reduce_max_logits(model, tensor) is reduced
    assert calls == [(tensor, gpt_model_module.ops.ReduceOp.MAX, "loss_group")]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_qk_clip_rejects_missing_multi_rank_loss_group():
    """A missing multi-rank group must fail instead of reducing over world."""
    model = SimpleNamespace()
    GPTModel.set_qk_clip_reduce_group(model, None, 16)

    with pytest.raises(RuntimeError, match="reduce group is missing"):
        GPTModel._all_reduce_max_logits(model, object())
