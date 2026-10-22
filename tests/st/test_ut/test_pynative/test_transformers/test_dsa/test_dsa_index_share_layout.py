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
"""Config-level tests for the DSA indexer Top-K sharing (IndexShare) layout."""

import pytest

from mindformers.models.deepseek3.config_converter_deepseek_v3 import DeepseekV3ConfigConverter
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.parallel_core.transformer_config_utils import (
    COMMON_CONFIG_MAPPING,
    scatter_multi_mapping_keys_to_mapping,
)


def _megatron_is_skip_topk_layer(layer_number, skip_topk_offset, topk_freq):
    """Megatron main's ``is_dsa_skip_topk_layer``, copied verbatim as the reference.

    The point of ``dsa_indexer_skip_topk_offset`` sharing Megatron's name is that a value
    copied from a Megatron config gives the same layout, so the expected layout comes from
    their formula rather than from ours.
    """
    skip_topk_offset = max(skip_topk_offset, 1)
    return (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0


def _config(**kwargs):
    base = {
        "num_layers": 8, "hidden_size": 512, "num_attention_heads": 8,
        "experimental_attention_variant": "dsa",
        "kv_lora_rank": 512, "qk_pos_emb_head_dim": 64,
        "dsa_indexer_head_dim": 128, "dsa_indexer_n_heads": 64,
    }
    base.update(kwargs)
    return MLATransformerConfig(**base)


def _pattern(config):
    return "".join("F" if full else "S" for full in config.resolve_dsa_index_share_leaders())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("num_layers", [4, 8, 12, 16, 18, 61])
@pytest.mark.parametrize("freq", [2, 3, 4, 8])
@pytest.mark.parametrize("offset", [0, 1, 2, 3, 5])
def test_uniform_layout_matches_megatron(num_layers, freq, offset):
    """The same (freq, offset) gives Megatron's layout, layer for layer."""
    config = _config(num_layers=num_layers, dsa_indexer_topk_freq=freq,
                     dsa_indexer_skip_topk_offset=offset)
    expected = tuple(
        not _megatron_is_skip_topk_layer(i + 1, offset, freq) for i in range(num_layers)
    )
    assert config.resolve_dsa_index_share_leaders() == expected


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_default_offset_keeps_the_original_layout():
    """offset 0 (and its Megatron equivalent 1) keep one Full layer every freq layers."""
    assert _pattern(_config(num_layers=12, dsa_indexer_topk_freq=4)) == "FSSSFSSSFSSS"
    assert _pattern(_config(num_layers=12, dsa_indexer_topk_freq=4,
                            dsa_indexer_skip_topk_offset=1)) == "FSSSFSSSFSSS"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_layer_count_need_not_be_a_multiple_of_freq():
    """Like Megatron, a remainder ends on a shorter group instead of being rejected."""
    assert _pattern(_config(num_layers=18, dsa_indexer_topk_freq=4)) == "FSSSFSSSFSSSFSSSFS"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("offset", [-1, True])
def test_offset_must_be_a_non_negative_integer(offset):
    """A negative offset is rejected by name rather than yielding another layout."""
    with pytest.raises(ValueError, match="dsa_indexer_skip_topk_offset"):
        _config(dsa_indexer_topk_freq=4, dsa_indexer_skip_topk_offset=offset)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("kwargs, pattern", [
    ({"num_layers": 8, "dsa_indexer_topk_freq": 2, "pipeline_model_parallel_size": 4}, "FSFSFSFS"),
    ({"num_layers": 16, "dsa_indexer_topk_freq": 4, "pipeline_model_parallel_size": 2,
      "virtual_pipeline_model_parallel_size": 2}, "FSSSFSSSFSSSFSSS"),
    # Groups of unequal length whose boundaries still land on Full layers: a group-size
    # divisibility test would reject this.
    ({"num_layers": 12, "dsa_indexer_topk_freq": 4, "dsa_indexer_skip_topk_offset": 3,
      "pipeline_model_parallel_size": 2}, "FFFSSSFSSSFS"),
])
def test_chunk_boundaries_on_full_layers_are_accepted(kwargs, pattern):
    """Every pipeline chunk starts on a Full layer, so no group is split."""
    assert _pattern(_config(**kwargs)) == pattern


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("kwargs, boundary", [
    ({"num_layers": 8, "dsa_indexer_topk_freq": 4, "pipeline_model_parallel_size": 4}, 2),
    ({"num_layers": 18, "dsa_indexer_topk_freq": 4, "pipeline_model_parallel_size": 2}, 9),
    # A group-size divisibility test would accept this (8 % 4 == 0) although the chunk
    # boundary at layer 8 falls inside a group.
    ({"num_layers": 16, "dsa_indexer_topk_freq": 4, "dsa_indexer_skip_topk_offset": 3,
      "pipeline_model_parallel_size": 2}, 8),
])
def test_group_spanning_a_chunk_boundary_is_rejected(kwargs, boundary):
    """The error names the boundary and the group count, not just a group size."""
    with pytest.raises(ValueError, match=rf"boundary at layer {boundary}, which is Shared") as err:
        _config(**kwargs)
    assert "size of a group, not the number of groups" in str(err.value)


def _model_config(**kwargs):
    base = {
        "num_hidden_layers": 8, "hidden_size": 512, "num_attention_heads": 8,
        "experimental_attention_variant": "dsa",
        "kv_lora_rank": 512, "qk_rope_head_dim": 64,
        "dsa_indexer_head_dim": 128, "dsa_indexer_n_heads": 64,
    }
    base.update(kwargs)
    return base


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("model_kwargs, pattern", [
    ({"dsa_indexer_topk_freq": 4}, "FSSSFSSS"),
    ({"dsa_indexer_topk_freq": 4, "dsa_indexer_skip_topk_offset": 2}, "FFSSSFSS"),
    ({"dsa_indexer_types": ["full", "shared", "shared", "full", "shared", "shared", "full",
                            "shared"]}, "FSSFSSFS"),
])
def test_model_config_converter_reaches_the_layout(model_kwargs, pattern):
    """The Megatron names reach the layout through the converter model configs actually use."""
    config = DeepseekV3ConfigConverter.convert(_model_config(**model_kwargs), is_mla_model=True)
    assert _pattern(config) == pattern


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_checkpoint_converter_mapping_accepts_glm5_names():
    """GLM-5's HF spellings are carried only by the mapping used for checkpoint conversion:
    no upstreamed model's HF config declares them, so they stay out of the converter base."""
    mapping = scatter_multi_mapping_keys_to_mapping(COMMON_CONFIG_MAPPING)
    assert mapping["index_topk_freq"] == "dsa_indexer_topk_freq"
    assert mapping["index_skip_topk_offset"] == "dsa_indexer_skip_topk_offset"
    assert mapping["indexer_types"] == "dsa_indexer_types"
