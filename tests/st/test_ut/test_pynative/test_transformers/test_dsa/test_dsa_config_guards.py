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
"""What the DSA config guards accept, reject and merely ignore."""

import pytest

from mindformers.parallel_core.transformer_config import MLATransformerConfig


def _dsa_config(**kwargs):
    base = {
        "num_layers": 8, "hidden_size": 512, "num_attention_heads": 8,
        "experimental_attention_variant": "dsa",
        "kv_lora_rank": 512, "qk_pos_emb_head_dim": 64,
        "dsa_indexer_head_dim": 128, "dsa_indexer_n_heads": 64,
    }
    base.update(kwargs)
    return MLATransformerConfig(**base)


def _non_dsa_config(**kwargs):
    base = {"num_layers": 8, "hidden_size": 512, "num_attention_heads": 8}
    base.update(kwargs)
    return MLATransformerConfig(**base)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("n_heads", [8, 16, 32, 64])
def test_indexer_head_count_accepts_the_supported_set(n_heads):
    """The indexer head count is not fixed at 64; the set matches the static-graph DSA branch."""
    assert _dsa_config(dsa_indexer_n_heads=n_heads).dsa_indexer_n_heads == n_heads


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("n_heads", [1, 63, 128])
def test_indexer_head_count_rejects_values_outside_the_set(n_heads):
    """A head count no operator supports still fails, by name."""
    with pytest.raises(ValueError, match="dsa_indexer_n_heads"):
        _dsa_config(dsa_indexer_n_heads=n_heads)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_indexer_head_dim_must_be_128():
    """The per-head dimension is still fixed."""
    with pytest.raises(ValueError, match="dsa_indexer_head_dim"):
        _dsa_config(dsa_indexer_head_dim=64)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("knob", [
    {"dsa_indexer_topk_freq": 4},
    {"dsa_indexer_types": ["full", "shared", "shared", "shared",
                           "full", "shared", "shared", "shared"]},
    {"dsa_indexer_layerwise_backward": True},
])
def test_dsa_knobs_are_inert_without_dsa(knob):
    """A DSA knob on a non-DSA model is ignored with a warning, not rejected.

    yaml templates are shared across models and may carry a leftover DSA key; that must not
    stop an unrelated model from starting.
    """
    config = _non_dsa_config(**knob)
    assert config.experimental_attention_variant != "dsa"
    for key, value in knob.items():
        assert getattr(config, key) == value
