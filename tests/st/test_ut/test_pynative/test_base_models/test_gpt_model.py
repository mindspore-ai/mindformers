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

import pytest

from mindformers.pynative.base_models.gpt.gpt_model import GPTModel


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
