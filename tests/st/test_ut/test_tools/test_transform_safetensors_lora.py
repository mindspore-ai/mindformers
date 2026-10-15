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
"""Tests for LoRA safetensors merge behavior."""
import numpy as np
import pytest
import mindspore as ms

from mindformers.tools import transform_safetensors_lora as lora_transform


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_separate_adapter_merge_preserves_moe_routing_state(tmp_path, monkeypatch):
    """Adapter routing state overrides stale base values in a merged export."""
    weight_prefix = "model.layers.0.mlp.linear"
    weight_name = f"{weight_prefix}.weight"
    lora_a_name = f"{weight_prefix}.lora_A.weight"
    lora_b_name = f"{weight_prefix}.lora_B.weight"
    expert_bias_name = "model.layers.0.mlp.router.expert_bias"
    tokens_name = "model.layers.0.mlp.router.tokens_per_expert"
    base_params = {
        weight_name: np.zeros((2, 2), dtype=np.float32),
        expert_bias_name: np.zeros(2, dtype=np.float32),
        tokens_name: np.zeros(2, dtype=np.int64),
    }
    adapter_params = {
        lora_a_name: np.ones((1, 2), dtype=np.float32),
        lora_b_name: np.ones((2, 1), dtype=np.float32),
        expert_bias_name: np.array([0.2, -0.2], dtype=np.float32),
        tokens_name: np.array([7, 9], dtype=np.int64),
    }
    dtypes = {
        weight_name: ms.float32,
        lora_a_name: ms.float32,
        lora_b_name: ms.float32,
        expert_bias_name: ms.float32,
        tokens_name: ms.int64,
    }

    def fake_load(path):
        params = base_params if path == "base" else adapter_params
        return params.copy(), {name: dtypes[name] for name in params}, [path]

    saved = {}
    monkeypatch.setattr(lora_transform, "_read_adapter_config", lambda _: lora_transform.AdapterConfig(
        {"lora_alpha": 1}, None, None))
    monkeypatch.setattr(lora_transform, "_load_checkpoint_params", fake_load)
    monkeypatch.setattr(lora_transform, "_atomic_save_checkpoint", lambda values, _: saved.update(
        {item["name"]: item["data"].asnumpy() for item in values}))

    lora_transform.merge_lora(
        base_ckpt_path="base",
        adapter_ckpt_path="adapter",
        dst_ckpt_path=str(tmp_path / "merged"),
    )

    np.testing.assert_allclose(saved[expert_bias_name], adapter_params[expert_bias_name])
    np.testing.assert_array_equal(saved[tokens_name], adapter_params[tokens_name])
    assert lora_a_name not in saved
    assert lora_b_name not in saved
