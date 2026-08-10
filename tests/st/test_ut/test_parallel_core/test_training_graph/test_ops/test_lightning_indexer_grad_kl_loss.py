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
"""Test the DSA lightning-indexer fused operators against the small-operator implementation."""
import os

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, context, mint, ops
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from tests.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator

from mindformers.parallel_core.training_graph.ops.dense_lightning_indexer_softmax_lse import (
    DenseLightningIndexerSoftmaxLse, INT64_MAX
)
from mindformers.parallel_core.training_graph.ops.dense_lightning_indexer_grad_kl_loss import (
    DenseLightningIndexerGradKLLoss
)
from mindformers.parallel_core.training_graph.ops.sparse_lightning_indexer_grad_kl_loss import (
    SparseLightningIndexerGradKLLoss
)


DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))
SPARSE_MODE_RIGHT_DOWN_CAUSAL = 3

# Matches DSAIndexerLoss.eps / DSAIndexer.inf on master.
EPS = 1e-8
INDEXER_MASK_FILL = -10000.0

MS_DTYPE = {"fp16": ms.float16, "bf16": ms.bfloat16}
BENCHMARK_DTYPE = {"fp16": "float16", "bf16": "bfloat16"}

GRAD_CV = {"max_re_cv": 3000, "avg_re_cv": 10, "rmse_cv": 5, "small_err_ratio_cv": 10}


def _setup_context():
    """Static graph at O0, which is what the ops.Custom aot wrappers require."""
    context.set_context(device_target="Ascend", device_id=DEVICE_ID, deterministic="ON")
    context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})


def _numpy(tensor):
    """Tensor -> fp32 numpy, for the final comparison."""
    if tensor.dtype == ms.bfloat16:
        return tensor.astype(ms.float32).asnumpy()
    return tensor.asnumpy().astype(np.float32)


def _check(name, fused, reference, dtype):
    """Compare the fused output against the small-op reference via the double benchmark."""
    golden = _numpy(reference)
    print(f"[{name}]")
    DoubleBenchmarkComparator.check_pass_or_not(
        npu_data=_numpy(fused),
        gpu_data=_numpy(Tensor(golden, MS_DTYPE[dtype])),
        golden_data=golden,
        standard=DoubleBenchmarkStandard(dtype=BENCHMARK_DTYPE[dtype], **GRAD_CV),
    )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _gen_inputs_np(seed, b, s1, s2, n1, n2, nidx1, nidx2, dim, d_index, d_rope):
    """Deterministic fp16 inputs, generated with numpy only."""
    rng = np.random.default_rng(seed)

    def _n(*shape):
        return rng.standard_normal(shape).astype(np.float16)

    return {
        "query": _n(b, s1, n1, dim),
        "key": _n(b, s2, n2, dim),
        "value": _n(b, s2, n2, dim),
        "query_rope": _n(b, s1, n1, d_rope),
        "key_rope": _n(b, s2, n2, d_rope),
        "query_index": _n(b, s1, nidx1, d_index),
        "key_index": _n(b, s2, nidx2, d_index),
        "weights": (rng.standard_normal((b, s1, nidx1)) * 0.05).astype(np.float16),
    }


def _as_tensors(arrays, ms_dtype):
    """numpy inputs -> Tensors; everything downstream stays a Tensor."""
    return {k: Tensor(v, ms_dtype) for k, v in arrays.items()}


def _right_down_causal_mask(b, s1, s2):
    """1 where the key is *not* visible, matching sparse_mode=3: row ``t`` sees ``[0, s2-s1+t]``."""
    j = np.arange(s2)[None, :]
    t = np.arange(s1)[:, None]
    mask = (j > (s2 - s1 + t)).astype(np.float32)
    return Tensor(np.broadcast_to(mask, (b, s1, s2)).copy())


# ---------------------------------------------------------------------------
# Small-operator reference
# ---------------------------------------------------------------------------
def _index_scores(query_index, key_index, weights, mask):
    """Index score [B, S1, S2], masked, as DSAIndexer.compute_dense_indices."""
    bsz, seqlen_q = query_index.shape[0], query_index.shape[1]
    q = query_index.astype(ms.float32)
    # [B, S2, 1, D] -> [B, 1, D, S2] so the matmul broadcasts over the query heads.
    k = key_index.astype(ms.float32).transpose(0, 2, 3, 1)
    # [B, S1, Nidx1, D] @ [B, 1, D, S2] -> [B, S1, Nidx1, S2]
    scores = ops.relu(mint.matmul(q, k))
    scores = scores * weights.astype(ms.float32).reshape(bsz, seqlen_q, -1, 1)
    # Sum over the indexer heads -> [B, S1, 1, S2]
    scores = mint.sum(scores, dim=2, keepdim=True).reshape(bsz, seqlen_q, -1)
    return scores + mask * INDEXER_MASK_FILL


def _kl_loss(index_scores, p_target):
    """KL(p || softmax(I)) summed over every token."""
    y = mint.softmax(index_scores, -1)
    kl = p_target * (mint.log(p_target + EPS) - mint.log(y + EPS))
    return mint.sum(mint.sum(kl, dim=-1))


def _dense_kl_loss(query_index, key_index, weights, p_target, mask):
    """Loss over the full causal context."""
    return _kl_loss(_index_scores(query_index, key_index, weights, mask), p_target)


def _sparse_kl_loss(query_index, key_index, weights, p_target, mask, sparse_indices):
    """Loss over the selected top-k columns."""
    index_scores = _index_scores(query_index, key_index, weights, mask)
    # [B, S1, S2] -> [B, S1, K] on the selected keys.
    return _kl_loss(mint.gather(index_scores, -1, sparse_indices), p_target)


# ---------------------------------------------------------------------------
# Main-attention target, rebuilt from the attention kernel's own statistics
# ---------------------------------------------------------------------------
def _attention_logits(query, key, query_rope, key_rope, scale):
    """Main-attention logits [B, N1, S1, S2], fp32."""
    q = mint.cat([query, query_rope], dim=-1).astype(ms.float32).transpose(0, 2, 1, 3)
    k = mint.cat([key, key_rope], dim=-1).astype(ms.float32).transpose(0, 2, 3, 1)
    return mint.matmul(q, k) * scale


def _target_from_stats(logits, softmax_max, softmax_sum):
    """p = (1/N1) * sum_h softmax_h(logits), i.e. head sum then L1 normalise."""
    return mint.mean(mint.exp(logits - softmax_max) / softmax_sum, dim=1)


def _dense_attention_stats(inputs, n1, scale):
    """softmax_max / softmax_sum from FlashAttentionScore (dense stage)."""
    q = mint.cat([inputs["query"], inputs["query_rope"]], dim=-1)
    k = mint.cat([inputs["key"], inputs["key_rope"]], dim=-1)
    fa = FlashAttentionScore(head_num=n1, scale_value=scale, inner_precise=0,
                             input_layout="BSND", sparse_mode=SPARSE_MODE_RIGHT_DOWN_CAUSAL)
    # sparse_mode=3 takes the compressed 2048x2048 upper-triangular mask.
    attn_mask = Tensor(np.triu(np.ones((2048, 2048), np.uint8), 1))
    softmax_max, softmax_sum, _, _ = fa(q, k, inputs["value"], None, None, None, attn_mask)
    return softmax_max[..., :1], softmax_sum[..., :1]


def _sparse_attention_stats(inputs, sparse_indices, scale):
    """softmax_max / softmax_sum from SparseFlashAttention (sparse stage)."""
    _, softmax_max, softmax_sum = ops.sparse_flash_attention(
        inputs["query"], inputs["key"], inputs["value"], sparse_indices, scale,
        query_rope=inputs["query_rope"], key_rope=inputs["key_rope"],
        layout_query="BSND", layout_kv="BSND", attention_mode=2, return_softmax_lse=True)
    return softmax_max, softmax_sum


# ---------------------------------------------------------------------------
# Dense: SoftmaxLse chained into GradKLLoss
# ---------------------------------------------------------------------------
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_dense_softmax_lse_chained_into_grad_kl_loss(dtype):
    """
    Feature: DSA dense lightning-indexer fused operators.
    Description: Run DenseLightningIndexerSoftmaxLse and feed its softmax statistics straight into
        DenseLightningIndexerGradKLLoss, the order the two ops execute in during the dense warmup
        stage, and compare against the small-operator implementation.
    Expectation: loss and d_query_index / d_key_index / d_weights pass the double benchmark against
        the small-op reference.
    """
    _setup_context()
    ms_dtype = MS_DTYPE[dtype]

    b, s1, s2 = 1, 128, 128
    n1 = n2 = 64                      # dense requires N2 == N1, so G == 1
    nidx1, nidx2 = 32, 1
    dim, d_index, d_rope = 128, 128, 64
    scale = float(dim + d_rope) ** -0.5

    arrays = _gen_inputs_np(20260810, b, s1, s2, n1, n2, nidx1, nidx2, dim, d_index, d_rope)
    inputs = _as_tensors(arrays, ms_dtype)
    ref_inputs = _as_tensors(arrays, ms.float32)
    mask = _right_down_causal_mask(b, s1, s2)

    # --- main-attention statistics straight out of FlashAttentionScore -----
    softmax_max, softmax_sum = _dense_attention_stats(inputs, n1, scale)

    # Rebuild the target the fused op reconstructs internally, from those very statistics.  Masked
    # keys get a large negative logit so their exp underflows to zero, as in the small-op path.
    logits = _attention_logits(inputs["query"], inputs["key"],
                               inputs["query_rope"], inputs["key_rope"], scale)
    logits = logits + mask.reshape(b, 1, s1, s2) * INDEXER_MASK_FILL
    p_target = _target_from_stats(logits, softmax_max, softmax_sum)

    # --- fused path: SoftmaxLse -> GradKLLoss ------------------------------
    softmax_max_index, softmax_sum_index = DenseLightningIndexerSoftmaxLse()(
        inputs["query_index"], inputs["key_index"], inputs["weights"],
        None, None, "BSND", SPARSE_MODE_RIGHT_DOWN_CAUSAL, INT64_MAX, INT64_MAX)

    d_query_index, d_key_index, d_weights, loss = DenseLightningIndexerGradKLLoss()(
        inputs["query"], inputs["key"], inputs["query_index"], inputs["key_index"],
        inputs["weights"], softmax_max, softmax_sum, softmax_max_index, softmax_sum_index,
        scale_value=scale, query_rope=inputs["query_rope"], key_rope=inputs["key_rope"],
        layout="BSND", sparse_mode=SPARSE_MODE_RIGHT_DOWN_CAUSAL,
        pre_tokens=INT64_MAX, next_tokens=INT64_MAX)

    # --- small-operator reference -----------------------------------------
    ref_args = (ref_inputs["query_index"], ref_inputs["key_index"], ref_inputs["weights"],
                p_target, mask)
    ref_loss = _dense_kl_loss(*ref_args)
    ref_grads = ops.grad(_dense_kl_loss, grad_position=(0, 1, 2))(*ref_args)

    _check("loss", loss.reshape(()), ref_loss, dtype)
    _check("d_query_index", d_query_index, ref_grads[0], dtype)
    _check("d_key_index", d_key_index, ref_grads[1], dtype)
    _check("d_weights", d_weights, ref_grads[2], dtype)


# ---------------------------------------------------------------------------
# Sparse
# ---------------------------------------------------------------------------
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_sparse_grad_kl_loss(dtype):
    """
    Feature: DSA sparse lightning-indexer fused operator.
    Description: Run SparseLightningIndexerGradKLLoss on a genuinely sparse selection (S2 > S1 so
        every row picks K keys out of a larger causal window) and compare against the
        small-operator implementation.
    Expectation: loss and d_query_index / d_key_index / d_weights pass the double benchmark against
        the small-op reference.
    """
    _setup_context()
    ms_dtype = MS_DTYPE[dtype]

    b, s1, s2 = 1, 512, 2048
    n1, n2 = 32, 1                    # SparseFA is MQA
    nidx1, nidx2 = 16, 1
    dim, d_index, d_rope = 512, 128, 64
    topk = 1024
    scale = float(dim + d_rope) ** -0.5

    arrays = _gen_inputs_np(20260811, b, s1, s2, n1, n2, nidx1, nidx2, dim, d_index, d_rope)
    inputs = _as_tensors(arrays, ms_dtype)
    ref_inputs = _as_tensors(arrays, ms.float32)

    # Row t may see keys [0, s2 - s1 + t]; that window is always wider than topk here, so every row
    # selects a strict subset and no padding is needed.
    rng = np.random.default_rng(20260812)
    indices = np.empty((s1, topk), np.int32)
    for t in range(s1):
        window = s2 - s1 + t + 1
        assert window >= topk
        indices[t] = np.sort(rng.choice(window, topk, replace=False))
    sparse_indices = Tensor(indices[None])                 # int32, [B, S1, K]
    sfa_indices = sparse_indices.reshape(b, s1, nidx2, topk)

    # --- main-attention statistics straight out of SparseFlashAttention ----
    softmax_max, softmax_sum = _sparse_attention_stats(inputs, sfa_indices, scale)

    # Rebuild the target from those statistics, over the selected columns only.  SFA emits
    # (B, N2, S1, G) with N2 = 1 and G = N1; transpose to (B, N1, S1, 1) so it broadcasts over the
    # context axis like the dense case.
    logits = _attention_logits(inputs["query"], inputs["key"],
                               inputs["query_rope"], inputs["key_rope"], scale)
    logits = mint.gather(
        logits, -1, mint.broadcast_to(sparse_indices.reshape(b, 1, s1, topk), (b, n1, s1, topk)))
    p_target = _target_from_stats(logits, softmax_max.transpose(0, 3, 2, 1),
                                  softmax_sum.transpose(0, 3, 2, 1))

    # --- fused path --------------------------------------------------------
    d_query_index, d_key_index, d_weights, loss = SparseLightningIndexerGradKLLoss()(
        inputs["query"], inputs["key"], inputs["query_index"], inputs["key_index"],
        inputs["weights"], sfa_indices, softmax_max, softmax_sum,
        scale_value=scale, query_rope=inputs["query_rope"], key_rope=inputs["key_rope"],
        layout="BSND", sparse_mode=SPARSE_MODE_RIGHT_DOWN_CAUSAL,
        pre_tokens=INT64_MAX, next_tokens=INT64_MAX)

    # --- small-operator reference -----------------------------------------
    ref_args = (ref_inputs["query_index"], ref_inputs["key_index"], ref_inputs["weights"],
                p_target, _right_down_causal_mask(b, s1, s2), sparse_indices)
    ref_loss = _sparse_kl_loss(*ref_args)
    ref_grads = ops.grad(_sparse_kl_loss, grad_position=(0, 1, 2))(*ref_args)

    _check("loss", loss.reshape(()), ref_loss, dtype)
    _check("d_query_index", d_query_index, ref_grads[0], dtype)
    _check("d_key_index", d_key_index, ref_grads[1], dtype)
    _check("d_weights", d_weights, ref_grads[2], dtype)
