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
"""Functional tests for the PYNATIVE DeepSeek Sparse Attention modules."""

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, mint, nn, ops

from mindformers.pynative.base_models.gpt.experimental_attention_variant_module_specs import (
    get_attention_variant_module_spec,
)
from mindformers.pynative.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from mindformers.pynative.transformers.experimental_attention_variant.dsa import (
    DSAttention,
    DSASoftmaxConverter,
    DSASparseFlashAttention,
)
from mindformers.pynative.transformers.experimental_attention_variant.dsa_attention import (
    DSASelfAttention,
)
from mindformers.pynative.transformers.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttention,
)
from mindformers.pynative.transformers.experimental_attention_variant.indexer import (
    CSAIndexer,
    UnfusedCSAIndexerLoss,
)
from mindformers.pynative.transformers.multi_latent_attention import MLASelfAttention


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_dsa_uses_registered_top_level_attention_class():
    """DSA is selected as a complete attention variant instead of an MLA branch."""
    dsa_layer_spec = get_gpt_layer_local_spec(
        multi_latent_attention=True,
        attention_variant="dsa",
    )
    dsa_spec = dsa_layer_spec.submodules.self_attention

    assert dsa_spec.module is DSASelfAttention
    assert dsa_spec.submodules.core_attention.module is DSAttention

    mla_layer_spec = get_gpt_layer_local_spec(multi_latent_attention=True)
    assert mla_layer_spec.submodules.self_attention.module is MLASelfAttention

    legacy_mla_layer_spec = get_gpt_layer_local_spec(
        multi_latent_attention=True,
        attention_variant="mla",
    )
    assert legacy_mla_layer_spec.submodules.self_attention.module is MLASelfAttention

    dsv4_spec = get_attention_variant_module_spec("dsv4_hybrid")
    assert dsv4_spec.module is DSv4HybridSelfAttention


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_legacy_mla_variant_requires_multi_latent_attention():
    """The legacy MLA alias must not silently select standard attention."""
    with pytest.raises(ValueError, match="multi_latent_attention must be True"):
        get_gpt_layer_local_spec(attention_variant="mla")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_unknown_attention_variant_fails_at_spec_build_time():
    """Unregistered variants fail once at the registry boundary."""
    with pytest.raises(ValueError, match="Unsupported experimental attention variant"):
        get_attention_variant_module_spec("not_registered")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_dsa_rejects_unsupported_attention_options():
    """DSA fails early for options that FlashAttention does not implement."""
    for kwargs in (
        {"attn_mask_type": "causal"},
        {"attention_type": "self"},
        {"cp_comm_type": "all_gather"},
    ):
        with pytest.raises(NotImplementedError):
            DSAttention(None, None, layer_number=1, **kwargs)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_sparse_flash_attention_forwards_layout_and_sequence_lengths(monkeypatch):
    """The sparse FlashAttention boundary forwards all runtime metadata."""
    calls = {}

    def fake_sparse_flash_attention(query, key, value, topk_indices, scale, **kwargs):
        calls.update({"query": query, "key": key, "value": value,
                      "topk_indices": topk_indices, "scale": scale, **kwargs})
        return query, mint.ones((1, 1, 1), dtype=ms.float32), mint.zeros((1, 1, 1), dtype=ms.float32)

    monkeypatch.setattr("mindformers.pynative.transformers.experimental_attention_variant.dsa.ops"
                        ".sparse_flash_attention", fake_sparse_flash_attention)
    module = DSASparseFlashAttention("TND", 0.5, attention_mode=2)
    query = mint.zeros((2, 1, 4), dtype=ms.float32)
    key = mint.zeros((3, 1, 4), dtype=ms.float32)
    value = mint.zeros((3, 1, 4), dtype=ms.float32)
    topk = Tensor([[0, 1]], ms.int32)
    qlen = Tensor([0, 2], ms.int32)
    kvlen = Tensor([0, 3], ms.int32)

    module(query, key, value, topk, actual_seq_qlen=qlen, actual_seq_kvlen=kvlen)

    assert calls["scale"] == 0.5
    assert calls["layout_query"] == "TND"
    assert calls["layout_kv"] == "TND"
    assert calls["attention_mode"] == 2
    assert calls["actual_seq_lengths_query"] is qlen
    assert calls["actual_seq_lengths_kv"] is kvlen
    assert calls["return_softmax_lse"] is True


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_softmax_converter_non_tnd_is_passthrough():
    """BSND dense-attention statistics do not need TND reordering."""
    converter = DSASoftmaxConverter("BSND")
    softmax_max = mint.ones((1, 2, 1), dtype=ms.float32)
    softmax_sum = mint.ones((1, 2, 1), dtype=ms.float32) * 2

    actual_max, actual_sum = converter(softmax_max, softmax_sum)

    assert actual_max is softmax_max
    assert actual_sum is softmax_sum


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_unfused_indexer_returns_selected_topk_scores():
    """The unfused indexer returns logits aligned with its selected indices."""
    indexer = nn.Cell.__new__(CSAIndexer)
    indexer.index_topk = 2
    indexer.apply_dsa_kernel_fusion = False
    indexer.input_layout = "BSND"
    indexer.compress_ratio = 1
    indexer.cast = ops.cast
    indexer.reshape = mint.reshape
    indexer.permute = mint.permute
    indexer.bmm = mint.matmul
    indexer.relu = mint.nn.functional.relu
    indexer.unsqueeze = mint.unsqueeze
    indexer.sum = mint.sum
    indexer.topk = mint.topk

    query = Tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], ms.float32)
    key = Tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]], ms.float32)
    weights = Tensor([[[1.0], [1.0]]], ms.float32)
    indices, scores = CSAIndexer.construct(indexer, query, key, weights)

    np.testing.assert_array_equal(indices.asnumpy(), [[[0, 2], [1, 2]]])
    np.testing.assert_allclose(scores.asnumpy(), [[[1.0, 1.0], [1.0, 1.0]]])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_unfused_indexer_loss_masks_invalid_prefix_without_nan():
    """All-invalid prefix rows produce a finite KL loss for stable training."""
    loss_fn = UnfusedCSAIndexerLoss(softmax_scale=1.0)
    index_scores = Tensor([[[float("-inf"), float("-inf")], [1.0, 0.0]]], ms.float32)
    topk_indices = Tensor([[[-1, -1], [0, 1]]], ms.int32)
    query = mint.ones((2, 1, 1, 2), dtype=ms.float32)
    key = mint.ones((3, 1, 1, 2), dtype=ms.float32)

    loss = loss_fn(index_scores, topk_indices, query, key)

    assert loss.shape == ()
    assert np.isfinite(loss.asnumpy()).all()
