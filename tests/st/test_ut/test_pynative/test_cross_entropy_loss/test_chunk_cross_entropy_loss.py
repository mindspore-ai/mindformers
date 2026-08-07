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
"""Tests for chunked PyNative cross entropy loss."""

from types import SimpleNamespace

import numpy as np
import pytest

import mindspore as ms
from mindspore import mint

import mindformers.pynative.loss.loss as loss_module
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.loss import ChunkCrossEntropyLoss, CrossEntropyLoss
from mindformers.pynative.transformers.multi_token_prediction import (
    get_mtp_layer_wise_logging_tracker,
    process_mtp_loss,
)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_chunk_per_token_loss_matches_non_chunked_loss_and_gradient(monkeypatch):
    """Chunk loss should preserve the per-token numerator/denominator contract."""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    logits = ms.Tensor(
        np.array(
            [
                [
                    [1.0, 0.0, -1.0, 2.0],
                    [0.5, 1.5, -0.5, 0.0],
                    [2.0, -1.0, 0.0, 1.0],
                    [0.2, 0.7, 1.2, -0.3],
                    [-0.4, 0.1, 0.9, 1.4],
                ],
                [
                    [-0.5, 0.5, 1.0, 0.0],
                    [1.0, 2.0, 0.0, -1.0],
                    [0.0, -0.5, 1.5, 0.5],
                    [1.1, -0.2, 0.3, 0.8],
                    [0.6, 0.4, -0.8, 1.0],
                ],
            ],
            dtype=np.float32,
        )
    )
    labels = ms.Tensor(
        np.array([[3, 1, 0, 2, 3], [2, 1, 2, 0, 3]], dtype=np.int32)
    )
    loss_mask = ms.Tensor(
        np.array([[1, 1, 0, 1, 0], [1, 0, 1, 1, 1]], dtype=np.float32)
    )
    chunk_loss = ChunkCrossEntropyLoss(
        calculate_per_token_loss=True,
        chunk_loss_num=3,
    )
    full_loss = CrossEntropyLoss(calculate_per_token_loss=True)

    chunk_numerator, chunk_denominator = chunk_loss(logits, labels, loss_mask)
    full_numerator, full_denominator = full_loss(
        logits.reshape((-1, logits.shape[-1])),
        labels.reshape((-1,)),
        loss_mask.reshape((-1,)),
    )

    np.testing.assert_allclose(chunk_numerator.asnumpy(), full_numerator.asnumpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(chunk_denominator.asnumpy(), full_denominator.asnumpy(), rtol=0, atol=0)

    def chunk_mean(input_logits):
        numerator, denominator = chunk_loss(input_logits, labels, loss_mask)
        return numerator / denominator

    def full_mean(input_logits):
        numerator, denominator = full_loss(
            input_logits.reshape((-1, input_logits.shape[-1])),
            labels.reshape((-1,)),
            loss_mask.reshape((-1,)),
        )
        return numerator / denominator

    chunk_mean_value, chunk_grad = ms.value_and_grad(chunk_mean)(logits)
    full_mean_value, full_grad = ms.value_and_grad(full_mean)(logits)
    np.testing.assert_allclose(chunk_mean_value.asnumpy(), full_mean_value.asnumpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(chunk_grad.asnumpy(), full_grad.asnumpy(), rtol=1e-6, atol=1e-6)

    # Exercise the vocab-parallel chunk implementation without requiring a
    # distributed launch. TP=1 makes each collective an identity operation.
    monkeypatch.setattr(loss_module, "_tp_all_reduce", lambda tensor, _op, _group: tensor)
    vocab_chunk_loss = ChunkCrossEntropyLoss(
        calculate_per_token_loss=True,
        chunk_loss_num=3,
    )
    vocab_chunk_loss.enable_vocab_parallel(group="mock_tp", rank=0, size=1)

    def vocab_chunk_mean(input_logits):
        numerator, denominator = vocab_chunk_loss(input_logits, labels, loss_mask)
        return numerator / denominator

    vocab_numerator, vocab_denominator = vocab_chunk_loss(logits, labels, loss_mask)
    vocab_mean_value, vocab_grad = ms.value_and_grad(vocab_chunk_mean)(logits)
    np.testing.assert_allclose(vocab_numerator.asnumpy(), full_numerator.asnumpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(vocab_denominator.asnumpy(), full_denominator.asnumpy(), rtol=0, atol=0)
    np.testing.assert_allclose(vocab_mean_value.asnumpy(), full_mean_value.asnumpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(vocab_grad.asnumpy(), full_grad.asnumpy(), rtol=1e-6, atol=1e-6)

    # The existing normalized-scalar contract must remain unchanged when the
    # per-token option is disabled.
    legacy_chunk_loss = ChunkCrossEntropyLoss(
        calculate_per_token_loss=False,
        chunk_loss_num=3,
    )

    def legacy_chunk_mean(input_logits):
        return legacy_chunk_loss(input_logits, labels, loss_mask)

    legacy_mean_value, legacy_grad = ms.value_and_grad(legacy_chunk_mean)(logits)
    np.testing.assert_allclose(legacy_mean_value.asnumpy(), full_mean_value.asnumpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(legacy_grad.asnumpy(), full_grad.asnumpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mtp_chunk_per_token_loss_uses_numerator_for_backward():
    """MTP should log a mean while injecting the unnormalized chunk numerator."""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    tracker = get_mtp_layer_wise_logging_tracker()
    tracker.clear()
    try:
        main_hidden = ms.Tensor(np.zeros((5, 2, 4), dtype=np.float32))
        mtp_logits = ms.Tensor(
            np.array(
                [
                    [[1.0, 0.0, -1.0, 2.0], [-0.5, 0.5, 1.0, 0.0]],
                    [[0.5, 1.5, -0.5, 0.0], [1.0, 2.0, 0.0, -1.0]],
                    [[2.0, -1.0, 0.0, 1.0], [0.0, -0.5, 1.5, 0.5]],
                    [[0.2, 0.7, 1.2, -0.3], [1.1, -0.2, 0.3, 0.8]],
                    [[-0.4, 0.1, 0.9, 1.4], [0.6, 0.4, -0.8, 1.0]],
                ],
                dtype=np.float32,
            )
        )
        hidden_states = mint.cat((main_hidden, mtp_logits), dim=0)
        labels = ms.Tensor(
            np.array([[3, 1, 0, 2, 3], [2, 1, 2, 0, 3]], dtype=np.int32)
        )
        loss_mask = ms.Tensor(
            np.array([[1, 1, 0, 1, 0], [1, 0, 1, 1, 1]], dtype=np.float32)
        )
        config = SimpleNamespace(
            mtp_num_layers=1,
            chunk_loss_num=3,
            mtp_loss_scaling_factor=0.25,
            calculate_per_token_loss=True,
        )
        captured_losses = []

        def output_layer(hidden, weight=None):
            del weight
            return hidden

        def capture_loss(output, loss):
            captured_losses.append(loss)
            return output

        chunk_loss = ChunkCrossEntropyLoss(
            calculate_per_token_loss=True,
            chunk_loss_num=3,
        )

        def compute_language_model_loss(target, logits, mask):
            return chunk_loss(logits, target, mask)

        output = process_mtp_loss(
            hidden_states_list=hidden_states,
            labels=labels,
            loss_mask=loss_mask,
            output_layer=output_layer,
            output_weight=None,
            compute_language_model_loss=compute_language_model_loss,
            config=config,
            mtp_loss_auto_scaler=capture_loss,
        )

        assert len(captured_losses) == 1
        np.testing.assert_allclose(output.asnumpy(), main_hidden.asnumpy(), rtol=0, atol=0)
        token_count = loss_mask.asnumpy()[:, 1:].sum()
        expected_logged_loss = captured_losses[0] / config.mtp_loss_scaling_factor / token_count
        np.testing.assert_allclose(
            tracker["values"][0].asnumpy(), expected_logged_loss.asnumpy(), rtol=1e-6, atol=1e-6)
    finally:
        tracker.clear()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_transformer_config_allows_chunk_per_token_loss():
    """The supported chunk/per-token combination should pass config validation."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        chunk_loss_num=2,
        calculate_per_token_loss=True,
    )

    assert config.chunk_loss_num == 2
    assert config.calculate_per_token_loss is True
