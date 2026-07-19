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
"""Regression tests for sparse Router auxiliary-loss statistics."""
# pylint: disable=protected-access

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, context, mint

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.transformers.moe.moe_utils import (
    compute_routing_scores_for_aux_loss,
    get_moe_layer_wise_logging_tracker,
)
from mindformers.pynative.transformers.moe.router import TopKRouter


def _build_router() -> TopKRouter:
    """Build a small Router for auxiliary-statistics tests."""
    config = TransformerConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_layers=1,
        num_moe_experts=4,
        moe_router_topk=2,
        add_bias_linear=False,
    )
    return TopKRouter(config)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_aux_score_helper_returns_compact_indices():
    """Auxiliary routing keeps top-k indices instead of a dense token-expert map."""
    context.set_context(mode=context.PYNATIVE_MODE)
    raw_scores = Tensor(
        [
            [0.2, 0.1, 0.8, 0.3],
            [0.8, 0.3, 0.2, 0.1],
        ],
        dtype=ms.float32,
    )

    topk_indices, normalized_scores = compute_routing_scores_for_aux_loss(
        raw_scores, topk=2, score_function="sigmoid"
    )

    expected_scores = raw_scores.asnumpy()
    expected_scores /= expected_scores.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(normalized_scores.asnumpy(), expected_scores, rtol=0.0, atol=1e-7)
    assert topk_indices.shape == (2, 2)
    np.testing.assert_array_equal(
        topk_indices.asnumpy(), np.array([[2, 3], [0, 1]], dtype=np.int64)
    )


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_sparse_seq_counts_preserve_sequence_major_order():
    """Sparse sequence counts match the original sequence-major dense reduction."""
    context.set_context(mode=context.PYNATIVE_MODE)
    router = _build_router()
    scores = Tensor(np.arange(24, dtype=np.float32).reshape(6, 4))
    topk_indices = Tensor(
        [
            [0, 2],
            [1, 3],
            [0, 1],
            [1, 2],
            [2, 3],
            [0, 3],
        ],
        dtype=ms.int64,
    )

    aggregated_scores, tokens_per_expert = router._reduce_seq_sum_pair(
        scores, topk_indices, bsz=2
    )

    expected_scores = scores.asnumpy().reshape(3, 2, 4).sum(axis=0)
    expected_counts = np.array([[2, 1, 2, 1], [1, 2, 1, 2]], dtype=np.float32)
    np.testing.assert_array_equal(aggregated_scores.asnumpy().reshape(2, 4), expected_scores)
    np.testing.assert_array_equal(tokens_per_expert.asnumpy().reshape(2, 4), expected_counts)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_sparse_token_counts_match_dense_histogram():
    """Token/global auxiliary paths count compact indices exactly once."""
    context.set_context(mode=context.PYNATIVE_MODE)
    router = _build_router()
    scores = Tensor(np.arange(24, dtype=np.float32).reshape(6, 4))
    topk_indices = Tensor(
        [[0, 2], [1, 3], [0, 1], [1, 2], [2, 3], [0, 3]], dtype=ms.int64
    )

    aggregated_scores, tokens_per_expert = router._reduce_token_sum_pair(
        scores, topk_indices
    )

    expected_counts = np.bincount(
        topk_indices.asnumpy().reshape(-1), minlength=4
    ).astype(np.float32)
    np.testing.assert_array_equal(aggregated_scores.asnumpy(), scores.asnumpy().sum(axis=0))
    np.testing.assert_array_equal(tokens_per_expert.asnumpy(), expected_counts)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_sparse_seq_aux_matches_dense_value_and_gradient():
    """Sparse statistics preserve the dense auxiliary-loss value and gradient."""
    context.set_context(mode=context.PYNATIVE_MODE)
    router = _build_router()
    logits = Tensor(
        [
            [0.2, -0.1, 0.7, 0.3],
            [0.6, 0.1, -0.2, 0.4],
            [-0.3, 0.8, 0.2, 0.1],
            [0.5, -0.4, 0.3, 0.9],
            [0.1, 0.4, 0.6, -0.2],
            [0.9, 0.2, -0.1, 0.5],
        ],
        dtype=ms.float32,
    )

    def dense_aux_loss(router_logits):
        scores = mint.sigmoid(router_logits)
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        _, topk_indices = mint.topk(scores, k=2, dim=1)
        routing_map = mint.zeros_like(router_logits).int()
        routing_map.scatter_(1, topk_indices, 1)
        aggregated_scores = scores.reshape((3, 2, 4)).sum(dim=0)
        tokens_per_expert = routing_map.reshape((3, 2, 4)).sum(dim=0)
        return mint.sum(aggregated_scores * tokens_per_expert)

    def sparse_aux_loss(router_logits):
        scores = mint.sigmoid(router_logits)
        topk_indices, normalized_scores = compute_routing_scores_for_aux_loss(
            scores, topk=2, score_function="sigmoid"
        )
        aggregated_scores, tokens_per_expert = router._reduce_seq_sum_pair(
            normalized_scores, topk_indices, bsz=2
        )
        return mint.sum(aggregated_scores * tokens_per_expert)

    dense_value, dense_gradient = ms.value_and_grad(dense_aux_loss)(logits)
    sparse_value, sparse_gradient = ms.value_and_grad(sparse_aux_loss)(logits)

    np.testing.assert_allclose(sparse_value.asnumpy(), dense_value.asnumpy(), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        sparse_gradient.asnumpy(), dense_gradient.asnumpy(), rtol=0.0, atol=1e-6
    )


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_force_balance_aux_uses_overridden_indices():
    """Force-balance auxiliary loss counts the overridden round-robin route."""
    context.set_context(mode=context.PYNATIVE_MODE)
    config = TransformerConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_layers=1,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_router_score_function="softmax",
        moe_router_force_expert_balance=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=0.01,
        add_bias_linear=False,
    )
    router = TopKRouter(config)
    scores = Tensor(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.7, 0.1, 0.1, 0.1],
            [0.7, 0.1, 0.1, 0.1],
            [0.7, 0.1, 0.1, 0.1],
        ],
        dtype=ms.float32,
    )
    selected_experts_indices = Tensor(
        [[0, 1], [2, 3], [0, 1], [2, 3]], dtype=ms.int64
    )
    tracker = get_moe_layer_wise_logging_tracker()
    tracker.clear()

    router._compute_aux_loss(
        scores,
        mint.ones((4, 2), dtype=ms.float32),
        seq_length=2,
        bsz=2,
        selected_experts_indices=selected_experts_indices,
    )

    np.testing.assert_allclose(tracker["values"][0].asnumpy(), 1.0, rtol=0.0, atol=1e-6)
    tracker.clear()
