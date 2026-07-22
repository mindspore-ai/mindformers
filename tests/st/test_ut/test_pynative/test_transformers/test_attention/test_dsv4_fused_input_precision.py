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
"""Regression tests for DSV4 fused-attention input precision boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import hadamard

import mindspore as ms
from mindspore import Tensor, context, mint, ops


ms.set_context(mode=context.PYNATIVE_MODE)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_hadamard_scales_after_bf16_linear():
    """Lock MindSpeed's unscaled-matrix GEMM followed by output scaling order."""
    from mindformers.pynative.transformers.experimental_attention_variant.utils import Hadamard

    rng = np.random.default_rng(20260713)
    input_value = Tensor(rng.standard_normal((2, 3, 128)).astype(np.float32), ms.bfloat16)
    matrix = Tensor(hadamard(128), ms.bfloat16)
    scale = 128 ** -0.5

    expected = mint.nn.functional.linear(input_value, matrix) * scale
    prescaled = mint.nn.functional.linear(input_value, matrix * scale)
    actual = Hadamard(128)(input_value)

    actual_f32 = ops.cast(actual, ms.float32).asnumpy()
    expected_f32 = ops.cast(expected, ms.float32).asnumpy()
    prescaled_f32 = ops.cast(prescaled, ms.float32).asnumpy()
    np.testing.assert_array_equal(actual_f32, expected_f32)
    assert np.max(np.abs(actual_f32 - prescaled_f32)) > 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_hadamard_fp32_matrix_casts_back_to_input_dtype():
    """Lock Megatron-style BF16 I/O with FP32 Hadamard arithmetic."""
    from mindformers.pynative.transformers.experimental_attention_variant.utils import Hadamard

    rng = np.random.default_rng(20260713)
    input_value = Tensor(rng.standard_normal((2, 3, 128)).astype(np.float32), ms.bfloat16)
    matrix = Tensor(hadamard(128), ms.float32)
    scale = 128 ** -0.5

    expected = mint.nn.functional.linear(ops.cast(input_value, ms.float32), matrix) * scale
    expected = ops.cast(expected, ms.bfloat16)
    actual = Hadamard(128, use_fp32_matrix=True)(input_value)

    assert actual.dtype == input_value.dtype
    np.testing.assert_array_equal(
        ops.cast(actual, ms.float32).asnumpy(),
        ops.cast(expected, ms.float32).asnumpy(),
    )


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_sparse_flash_preparation_rejects_nonpositive_compress_ratio():
    """A compressed-KV branch must never reach residual modulo with a zero ratio."""
    from mindformers.pynative.transformers.experimental_attention_variant.csa import (
        _prepare_sparse_flash_mla,
    )

    ori_kv = mint.zeros((1, 4, 1, 2), dtype=ms.bfloat16)
    cmp_kv = mint.zeros((1, 1, 1, 2), dtype=ms.bfloat16)
    with pytest.raises(ValueError, match="cmp_ratio must be positive"):
        _prepare_sparse_flash_mla(ori_kv, cmp_kv, None, 0)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_fused_indexer_backward_applies_loss_coefficient(monkeypatch):
    """Indexer gradients include both the main-loss scale and configured loss coefficient."""
    from mindformers.pynative.transformers.experimental_attention_variant import csa

    query = mint.zeros((1, 2, 1, 2), dtype=ms.bfloat16)
    ori_kv = mint.zeros((1, 2, 1, 2), dtype=ms.bfloat16)
    cmp_kv = mint.zeros((1, 1, 1, 2), dtype=ms.bfloat16)
    query_index = mint.zeros((1, 2, 1, 2), dtype=ms.bfloat16)
    key_index = mint.zeros((1, 1, 1, 2), dtype=ms.bfloat16)
    weights = mint.zeros((1, 2, 1), dtype=ms.float32)
    sinks = mint.zeros((1,), dtype=ms.float32)
    cmp_softmax_l1 = Tensor([[[[0.25, 0.75]]]], dtype=ms.float32)

    def fake_sparse_flash_grad(*args, **kwargs):
        del args, kwargs
        return query, ori_kv, cmp_kv, sinks, None, cmp_softmax_l1

    def fake_indexer_grad(*args, **kwargs):
        del args, kwargs
        return (
            mint.ones_like(query_index),
            mint.ones_like(key_index),
            mint.ones_like(weights),
            cmp_softmax_l1,
        )

    monkeypatch.setattr(csa, "npu_sparse_flash_mla_grad", fake_sparse_flash_grad)
    monkeypatch.setattr(csa, "npu_sparse_lightning_indexer_kl_loss_grad", fake_indexer_grad)
    monkeypatch.setattr(csa, "save_to_indexer_losses_tracker", lambda *args: None)
    monkeypatch.setattr(csa._IndexerLossAutoScaler, "main_loss_backward_scale", 4.0)

    ctx = SimpleNamespace(
        saved_tensors=(
            query,
            ori_kv,
            cmp_kv,
            query_index,
            key_index,
            weights,
            Tensor([0], dtype=ms.int32),
            sinks,
            query,
            mint.zeros((1, 2, 1), dtype=ms.float32),
        ),
        has_cmp_kv=True,
        has_sparse_indices=False,
        softmax_scale=1.0,
        cmp_ratio=4,
        ori_mask_mode=3,
        cmp_mask_mode=3,
        ori_win_left=1,
        ori_win_right=0,
        loss_coeff=0.25,
        layer_number=1,
        num_layers=1,
    )
    gradients = csa.FusedSparseFlashMlaWithIndexerLoss.backward(
        ctx, mint.ones_like(query)
    )

    expected_scale = 4.0 * 0.25 / 2
    np.testing.assert_array_equal(
        gradients[4].asnumpy(),
        np.full(query_index.shape, expected_scale, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        gradients[5].asnumpy(),
        np.full(key_index.shape, expected_scale, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        gradients[6].asnumpy(),
        np.full(weights.shape, expected_scale, dtype=np.float32),
    )
