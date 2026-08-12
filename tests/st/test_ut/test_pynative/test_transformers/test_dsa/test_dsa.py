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

from types import SimpleNamespace

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, mint, nn, ops

from mindformers.pynative.base_models.gpt.experimental_attention_variant_module_specs import (
    get_attention_variant_module_spec,
)
from mindformers.pynative.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from mindformers.pynative.base_models.gpt import parallelize as gpt_parallelize
from mindformers.pynative.distributed.style import AllGather, SequenceParallel
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


class _FakeTPMesh:
    """Minimal mesh used to inspect DSA's declarative TP plan."""

    def __init__(self, size):
        self._size = size

    def size(self):
        """Return the configured TP degree."""
        return self._size


def _input_transform(plan, name, index):
    """Return one positional input transform from a plan entry."""
    return plan[name].input_transforms[index]


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
@pytest.mark.parametrize(
    "use_tnd,sparse_loss,query_dim,softmax_dim",
    ((False, True, 2, 3), (True, True, 1, 2),
     (False, False, 2, 1), (True, False, 1, 0)),
)
def test_dsa_tp_plan_uses_local_heads_and_loss_only_head_gathers(
        monkeypatch, use_tnd, sparse_loss, query_dim, softmax_dim
):
    """DSA TP gathers through hooks on real Indexer and fused-loss boundaries."""
    configured = []
    indexer_weight = object()
    indexer_cell = SimpleNamespace(
        parameters_and_names=lambda expand=False: (("weight", indexer_weight),)
    )
    indexer = SimpleNamespace(cells_and_names=lambda: (("linear", indexer_cell),))
    core_attention = SimpleNamespace(indexer=indexer)
    attention = SimpleNamespace(
        num_attention_heads=8,
        use_tnd=use_tnd,
        sparse_loss=sparse_loss,
        core_attention=core_attention,
    )
    projection_plan = {
        "self_attention.linear_qb": object(),
        "self_attention.linear_kvb": object(),
        "self_attention.linear_proj": object(),
    }
    monkeypatch.setattr(
        gpt_parallelize,
        "_mla_attention_layer_plan",
        lambda *_args, **_kwargs: projection_plan.copy(),
    )
    monkeypatch.setattr(
        gpt_parallelize,
        "_configure_dsa_local_fa",
        lambda core, world: configured.append((core, world)),
    )

    plan, param_plan = gpt_parallelize._dsa_attention_layer_plan(attention, _FakeTPMesh(2))

    assert configured == [(core_attention, 2)]
    assert "self_attention" not in plan
    assert projection_plan.keys() <= plan.keys()
    indexer_path = "self_attention.core_attention.indexer"
    assert _input_transform(plan, indexer_path, 0).dim == 0
    assert _input_transform(plan, indexer_path, 1).dim == 0
    assert _input_transform(plan, indexer_path, 0).reduce_grad is False
    assert _input_transform(plan, indexer_path, 1).reduce_grad is False

    loss_path = "self_attention.core_attention.indexer_loss.compute_indexer_loss"
    loss_transforms = plan[loss_path].input_transforms
    assert loss_transforms[0].dim == query_dim
    assert loss_transforms[6].dim == softmax_dim
    assert loss_transforms[7].dim == softmax_dim
    assert loss_transforms[8].dim == query_dim
    assert all(
        transform.reduce_grad is False
        for transform in (loss_transforms[0], loss_transforms[6],
                          loss_transforms[7], loss_transforms[8])
    )
    assert isinstance(loss_transforms[1], AllGather) is (not sparse_loss)
    assert isinstance(loss_transforms[9], AllGather) is (not sparse_loss)
    if not sparse_loss:
        assert loss_transforms[1].dim == query_dim
        assert loss_transforms[9].dim == query_dim

    removed_handoffs = (
        "dsa_indexer_x_handoff", "dsa_indexer_q_handoff",
        "dsa_sparse_kv_handoff", "dsa_loss_query_handoff",
        "dsa_loss_key_handoff", "dsa_loss_softmax_max_handoff",
        "dsa_loss_softmax_sum_handoff",
    )
    assert all(f"self_attention.{name}" not in plan for name in removed_handoffs)
    if sparse_loss:
        k_layernorm_style = plan["self_attention.k_layernorm"]
        assert isinstance(k_layernorm_style, SequenceParallel)
    assert param_plan == [[indexer_cell, "weight", (gpt_parallelize.Replicate(),)]]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_dsa_tp_plan_requires_complete_local_heads(monkeypatch):
    """A TP shard may not split the rows belonging to one DSA attention head."""
    attention = SimpleNamespace(num_attention_heads=6)
    monkeypatch.setattr(
        gpt_parallelize,
        "_mla_attention_layer_plan",
        lambda *_args, **_kwargs: {},
    )

    with pytest.raises(ValueError, match="num_attention_heads.*TP degree"):
        gpt_parallelize._dsa_attention_layer_plan(attention, _FakeTPMesh(4))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_configure_dsa_local_fa_rebuilds_dense_kernel(monkeypatch):
    """The parallel layer rebuilds dense DSA FlashAttention with local heads."""
    built = {}

    def _fake_flash_attention(**kwargs):
        built.update(kwargs)
        return object()

    dense_attention = SimpleNamespace(
        head_num=8,
        softmax_scale=0.125,
        input_layout="BSND",
        sparse_mode=0,
        flash_attention=None,
    )
    core_attention = nn.Cell.__new__(DSAttention)
    core_attention.head_num = 8
    core_attention.sparse_loss = False
    core_attention.dense_flash_attention = dense_attention
    monkeypatch.setattr(gpt_parallelize, "FlashAttentionScore", _fake_flash_attention)

    gpt_parallelize._configure_dsa_local_fa(core_attention, 2)

    assert core_attention.head_num == 4
    assert dense_attention.head_num == 4
    assert built["head_num"] == 4
    assert dense_attention.flash_attention is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_dsa_sparse_head_parallel_math_matches_single_partition():
    """Local absorb plus row-parallel output is algebraically identical to TP=1."""
    rng = np.random.default_rng(42)
    tokens, heads, qk_dim, value_dim, latent_dim, hidden_size = 5, 4, 3, 2, 6, 7
    query = rng.normal(size=(tokens, heads, qk_dim)).astype(np.float32)
    compressed_kv = rng.normal(size=(tokens, latent_dim)).astype(np.float32)
    kv_up = rng.normal(
        size=(heads, qk_dim + value_dim, latent_dim)
    ).astype(np.float32)
    output_weight = rng.normal(size=(hidden_size, heads * value_dim)).astype(np.float32)

    q_absorb = kv_up[:, :qk_dim, :]
    v_absorb = kv_up[:, qk_dim:, :]
    absorbed_query = np.einsum("thd,hdr->thr", query, q_absorb)
    scores = np.einsum("thr,sr->ths", absorbed_query, compressed_kv)
    scores -= scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    latent_output = np.einsum("ths,sr->thr", probabilities, compressed_kv)
    value_output = np.einsum("thr,hvr->thv", latent_output, v_absorb)
    full_output = value_output.reshape(tokens, -1) @ output_weight.T

    partial_outputs = []
    for head_slice in (slice(0, 2), slice(2, 4)):
        local_value = value_output[:, head_slice, :].reshape(tokens, -1)
        start = head_slice.start * value_dim
        stop = head_slice.stop * value_dim
        partial_outputs.append(local_value @ output_weight[:, start:stop].T)

    np.testing.assert_allclose(sum(partial_outputs), full_output, rtol=1.e-5, atol=1.e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_dsa_teacher_head_aggregation_matches_single_partition():
    """Summing TP-local head probabilities recovers the TP=1 indexer teacher."""
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(2, 4, 5, 5)).astype(np.float32)
    logits -= logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)

    full_teacher = probabilities.sum(axis=1)
    full_teacher /= full_teacher.sum(axis=-1, keepdims=True)
    local_teacher_sum = probabilities[:, :2].sum(axis=1) + probabilities[:, 2:].sum(axis=1)
    local_teacher_sum /= local_teacher_sum.sum(axis=-1, keepdims=True)

    np.testing.assert_allclose(local_teacher_sum, full_teacher, rtol=1.e-6, atol=1.e-6)


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
