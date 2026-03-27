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
"""MindSpore vs PTA (torch_npu) bit-exact comparison tests for
aclnnSparseLightningIndexerGradKLLoss.

Run from mindformers/ root:
    pytest -sv mindformers/parallel_core/training_graph/ops/\
test_sparse_lightning_indexer_grad_kl_loss.py

Single case:
    pytest -sv mindformers/parallel_core/training_graph/ops/\
test_sparse_lightning_indexer_grad_kl_loss.py::test_bsnd_with_rope
"""
import numpy as np
import pytest
import torch
import torch_npu

import mindspore as ms
from mindspore import Tensor, context
from mindformers.parallel_core.training_graph.ops.sparse_lightning_indexer_grad_kl_loss import (
    SparseLightningIndexerGradKLLoss,
)


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)) or np.any(np.isnan(data_me)):
        assert np.allclose(data_expected, data_me, rtol, atol,
                           equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol,
                         equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape

# ---------------------------------------------------------------------------
# Global device / context setup
# ---------------------------------------------------------------------------
DEVICE_ID = 7
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
ms.context.set_context(device_target="Ascend", device_id=DEVICE_ID)

ms.context.set_context(deterministic="ON", pynative_synchronize=False)
torch.use_deterministic_algorithms(True)

OUTPUT_NAMES = ("d_query_index", "d_key_index", "d_weights", "loss")


# =========================================================================
# Helper: input generation (numpy-based, deterministic)
# =========================================================================
def _gen_inputs_np(seed, batch, s1, s2, n1, n2, nidx1, nidx2,  # pylint: disable=too-many-arguments,too-many-locals
                   dim, d_index, dr, top_k,
                   is_tnd=False, seqlens_q=None, seqlens_kv=None,
                   pta_weights=False):
    """Generate all inputs as numpy fp16 arrays with deterministic seed.

    Both PTA and MS convert from the same numpy arrays:
      PTA: torch.from_numpy(arr).npu()  then optionally .to(bf16)
      MS:  Tensor(arr)                  then optionally .astype(bf16)

    IMPORTANT: The random number consumption order must match
    compare_pta_ms_sparse_lightning.py exactly so that the same seed
    produces the same data.  Order: query, key, query_index, key_index,
    query_rope, key_rope, weights, sparse_indices, softmax_max,
    softmax_sum.
    """
    np.random.seed(seed)

    scale = 1.0 / np.sqrt(dim)
    idx_scale = 1.0 / np.sqrt(d_index)

    # --- generate in BSND shape first (matches golden gen_inputs_np) ---
    query = (np.random.randn(batch, s1, n1, dim) * scale).astype(np.float16)
    key = (np.random.randn(batch, s2, n2, dim) * scale).astype(np.float16)
    query_index = (np.random.randn(batch, s1, nidx1, d_index)
                   * idx_scale).astype(np.float16)
    key_index = (np.random.randn(batch, s2, nidx2, d_index)
                 * idx_scale).astype(np.float16)

    # rope consumed before weights (matches golden order)
    query_rope, key_rope = None, None
    if dr > 0:
        rope_scale = 1.0 / np.sqrt(dr)
        query_rope = (np.random.randn(batch, s1, n1, dr)
                      * rope_scale).astype(np.float16)
        key_rope = (np.random.randn(batch, s2, n2, dr)
                    * rope_scale).astype(np.float16)

    raw_w = np.random.randn(batch, s1, nidx1).astype(np.float16)
    if pta_weights:
        a, b_val, kk = -0.05, 0.05, 3.0
        sc = (b_val - a) / (2 * kk)
        sh = (a + b_val) / 2
        weights = (raw_w * sc + sh).astype(np.float16)
    else:
        weights = (raw_w * 0.1 / 6.0).astype(np.float16)

    if is_tnd:
        t1 = sum(seqlens_q) if seqlens_q else s1
        sparse = np.zeros((t1, nidx2, top_k), dtype=np.int32)
        t_idx = 0
        for b_idx in range(batch):
            sq = seqlens_q[b_idx] if seqlens_q else s1
            sk = seqlens_kv[b_idx] if seqlens_kv else s2
            for s_idx in range(sq):
                s2_real = int(sk - sq + s_idx + 1)
                if s2_real <= 0:
                    s2_real = sk
                s2_len = min(s2_real, top_k)
                sparse[t_idx, :, :s2_len] = np.random.randint(
                    0, s2_real, (nidx2, s2_len)).astype(np.int32)
                sparse[t_idx, :, s2_len:] = -1
                t_idx += 1
        if batch == 1:
            query = query.reshape(s1, n1, dim)
            key = key.reshape(s2, n2, dim)
            query_index = query_index.reshape(s1, nidx1, d_index)
            key_index = key_index.reshape(s2, nidx2, d_index)
            weights = weights.reshape(s1, nidx1)
            if query_rope is not None:
                query_rope = query_rope.reshape(s1, n1, dr)
                key_rope = key_rope.reshape(s2, n2, dr)
        sm_shape = (n2, t1, n1)
    else:
        sparse = np.zeros((batch, s1, nidx2, top_k), dtype=np.int32)
        for s1_idx in range(s1):
            s2_real = max(int(s2 - s1 + s1_idx + 1), 1)
            s2_len = min(s2_real, top_k)
            sparse[:, s1_idx, 0, :s2_len] = np.random.randint(
                0, s2_real, (s2_len,)).astype(np.int32)
            sparse[:, s1_idx, 0, s2_len:] = -1
        sm_shape = (batch, n2, s1, n1)

    # Consume random state for backward-compat with golden script seeds.
    np.random.randn(*sm_shape)
    np.random.randn(*sm_shape)

    # softmax_max/softmax_sum are forward-pass online-softmax intermediates.
    # Random values are physically meaningless and cause fp16 overflow in the
    # backward kernel.  Use small random perturbations around safe baselines:
    #   max ∈ [-0.05, 0.05]  (realistic for scaled dot-product scores)
    #   sum ∈ [1.0, 2.0]     (positive, avoids div-by-zero)
    np.random.seed(seed + 9999)
    softmax_max = (np.random.rand(*sm_shape) * 0.1 - 0.05).astype(np.float32)
    softmax_sum = (np.random.rand(*sm_shape) + 1.0).astype(np.float32)

    return {
        "query": query, "key": key,
        "query_index": query_index, "key_index": key_index,
        "weights": weights, "sparse_indices": sparse,
        "softmax_max": softmax_max, "softmax_sum": softmax_sum,
        "query_rope": query_rope, "key_rope": key_rope,
    }


def _gen_tnd_flattened_inputs_np(  # pylint: disable=too-many-arguments,too-many-locals
        seed, seqlens_q, seqlens_kv, n1, n2, nidx1, nidx2,
        dim, d_index, dr, top_k, pta_weights=False):
    """Generate flattened TND inputs for multi-batch/ragged cases."""
    np.random.seed(seed)

    t1 = sum(seqlens_q)
    t2 = sum(seqlens_kv)
    scale = 1.0 / np.sqrt(dim)
    idx_scale = 1.0 / np.sqrt(d_index)

    query = (np.random.randn(t1, n1, dim) * scale).astype(np.float16)
    key = (np.random.randn(t2, n2, dim) * scale).astype(np.float16)
    query_index = (np.random.randn(t1, nidx1, d_index)
                   * idx_scale).astype(np.float16)
    key_index = (np.random.randn(t2, nidx2, d_index)
                 * idx_scale).astype(np.float16)

    query_rope, key_rope = None, None
    if dr > 0:
        rope_scale = 1.0 / np.sqrt(dr)
        query_rope = (np.random.randn(t1, n1, dr)
                      * rope_scale).astype(np.float16)
        key_rope = (np.random.randn(t2, n2, dr)
                    * rope_scale).astype(np.float16)

    raw_w = np.random.randn(t1, nidx1).astype(np.float16)
    if pta_weights:
        a, b_val, kk = -0.05, 0.05, 3.0
        sc = (b_val - a) / (2 * kk)
        sh = (a + b_val) / 2
        weights = (raw_w * sc + sh).astype(np.float16)
    else:
        weights = (raw_w * 0.1 / 6.0).astype(np.float16)

    sparse = np.zeros((t1, nidx2, top_k), dtype=np.int32)
    t_idx = 0
    for sq, sk in zip(seqlens_q, seqlens_kv):
        for s_idx in range(sq):
            s2_real = int(sk - sq + s_idx + 1)
            if s2_real <= 0:
                s2_real = sk
            s2_len = min(s2_real, top_k)
            sparse[t_idx, :, :s2_len] = np.random.randint(
                0, s2_real, (nidx2, s2_len)).astype(np.int32)
            sparse[t_idx, :, s2_len:] = -1
            t_idx += 1

    sm_shape = (n2, t1, n1)
    np.random.randn(*sm_shape)
    np.random.randn(*sm_shape)

    np.random.seed(seed + 9999)
    softmax_max = (np.random.rand(*sm_shape) * 0.1 - 0.05).astype(np.float32)
    softmax_sum = (np.random.rand(*sm_shape) + 1.0).astype(np.float32)

    return {
        "query": query, "key": key,
        "query_index": query_index, "key_index": key_index,
        "weights": weights, "sparse_indices": sparse,
        "softmax_max": softmax_max, "softmax_sum": softmax_sum,
        "query_rope": query_rope, "key_rope": key_rope,
    }


# =========================================================================
# Helper: run PTA (torch_npu)
# =========================================================================
def _run_pta(inputs_np, scale_value, layout, sparse_mode,
             pre_tokens, next_tokens,
             seqlens_q=None, seqlens_kv=None, use_bf16=False):
    """Run torch_npu op, return 4-tuple of numpy arrays.

    bf16 path: fp16 numpy -> torch fp16 -> NPU -> .to(bf16) on device,
    matching the golden (compare_pta_ms) conversion order exactly.
    """
    def _to_npu(arr):
        if arr is None:
            return None
        t = torch.from_numpy(arr).npu()
        if use_bf16 and t.dtype == torch.float16:
            t = t.to(torch.bfloat16)
        return t

    results = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
        _to_npu(inputs_np["query"]),
        _to_npu(inputs_np["key"]),
        _to_npu(inputs_np["query_index"]),
        _to_npu(inputs_np["key_index"]),
        _to_npu(inputs_np["weights"]),
        torch.from_numpy(inputs_np["sparse_indices"]).npu(),
        torch.from_numpy(inputs_np["softmax_max"]).npu(),
        torch.from_numpy(inputs_np["softmax_sum"]).npu(),
        scale_value,
        query_rope=_to_npu(inputs_np["query_rope"]),
        key_rope=_to_npu(inputs_np["key_rope"]),
        actual_seq_qlen=seqlens_q,
        actual_seq_klen=seqlens_kv,
        layout=layout,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
    )

    def _out_np(t):
        if t.dtype == torch.bfloat16:
            return t.float().cpu().numpy()
        return t.cpu().numpy()

    return tuple(_out_np(r) for r in results)


# =========================================================================
# Helper: run MindSpore
# =========================================================================
def _run_ms(inputs_np, scale_value, layout, sparse_mode,
            pre_tokens, next_tokens,
            seqlens_q=None, seqlens_kv=None, use_bf16=False):
    """Run MS SparseLightningIndexerGradKLLoss, return 4-tuple of numpy.

    bf16 path: fp16 numpy -> Tensor(fp16) -> .astype(bf16) on device,
    matching the golden (compare_pta_ms) conversion order exactly.
    """
    context.set_context(mode=ms.GRAPH_MODE,
                        jit_config={"jit_level": "O0"})

    def _to_ms(arr):
        if arr is None:
            return None
        t = Tensor(arr)
        if use_bf16 and t.dtype == ms.float16:
            t = t.astype(ms.bfloat16)
        return t

    op = SparseLightningIndexerGradKLLoss()
    results = op(
        _to_ms(inputs_np["query"]),
        _to_ms(inputs_np["key"]),
        _to_ms(inputs_np["query_index"]),
        _to_ms(inputs_np["key_index"]),
        _to_ms(inputs_np["weights"]),
        Tensor(inputs_np["sparse_indices"]),
        Tensor(inputs_np["softmax_max"]),
        Tensor(inputs_np["softmax_sum"]),
        scale_value=scale_value,
        query_rope=_to_ms(inputs_np["query_rope"]),
        key_rope=_to_ms(inputs_np["key_rope"]),
        actual_seq_qlen=seqlens_q,
        actual_seq_klen=seqlens_kv,
        layout=layout,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
    )

    def _out_np(t):
        if t.dtype == ms.bfloat16:
            return t.astype(ms.float32).asnumpy()
        return t.asnumpy()

    return tuple(_out_np(r) for r in results)


INT64_MAX = 9223372036854775807


# =========================================================================
# Test: BSND layout with rope, fp16 and bf16
# =========================================================================
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_bsnd_with_rope(dtype):
    """
    Feature: MS vs PTA bit-exact comparison
    Description: BSND layout, B=1, S1=S2=128, N1=64, N2=1, Nidx1=64,
        Nidx2=1, D=512, DIndex=128, DR=64, topK=2048, with rope.
    Expectation: All 4 outputs match PTA bit-for-bit (rtol=0, atol=0).
    """
    use_bf16 = dtype == "bf16"
    inputs = _gen_inputs_np(
        seed=42, batch=1, s1=128, s2=128, n1=64, n2=1,
        nidx1=64, nidx2=1, dim=512, d_index=128, dr=64,
        top_k=2048, is_tnd=False)

    pta_out = _run_pta(
        inputs, scale_value=1.0, layout="BSND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536,
        use_bf16=use_bf16)
    ms_out = _run_ms(
        inputs, scale_value=1.0, layout="BSND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536,
        use_bf16=use_bf16)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)
    allclose_nparray(pta_out[2], ms_out[2], 0, 0)
    allclose_nparray(pta_out[3], ms_out[3], 0, 0)


# =========================================================================
# Test: BSND, Nidx1 doc matrix
# =========================================================================
@pytest.mark.parametrize(
    'nidx1,seed',
    [(8, 170), (16, 171), (32, 177), (64, 178)],
    ids=[
        'nidx1_8_seed_170',
        'nidx1_16_seed_171',
        'nidx1_32_seed_177',
        'nidx1_64_seed_178',
    ],
)
def test_bsnd_nidx1_matrix(nidx1, seed):
    """
    Feature: MS vs PTA bit-exact comparison
    Description: BSND, doc-supported Nidx1 values (8/16/32/64).
    Expectation: All 4 outputs match PTA bit-for-bit.
    """
    inputs = _gen_inputs_np(
        seed=seed, batch=1, s1=128, s2=128, n1=64, n2=1,
        nidx1=nidx1, nidx2=1, dim=512, d_index=128, dr=64,
        top_k=2048, is_tnd=False)

    pta_out = _run_pta(
        inputs, scale_value=1.0, layout="BSND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536)
    ms_out = _run_ms(
        inputs, scale_value=1.0, layout="BSND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)
    allclose_nparray(pta_out[2], ms_out[2], 0, 0)
    allclose_nparray(pta_out[3], ms_out[3], 0, 0)


# =========================================================================
# Test: TND layout with rope, fp16 and bf16
# =========================================================================
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_tnd_with_rope(dtype):
    """
    Feature: MS vs PTA bit-exact comparison
    Description: TND layout, B=1, S1=S2=128, N1=64, N2=1, Nidx1=64,
        Nidx2=1, D=512, DIndex=128, DR=64, topK=2048, with rope.
    Expectation: All 4 outputs match PTA bit-for-bit (rtol=0, atol=0).
    """
    use_bf16 = dtype == "bf16"
    inputs = _gen_inputs_np(
        seed=42, batch=1, s1=128, s2=128, n1=64, n2=1,
        nidx1=64, nidx2=1, dim=512, d_index=128, dr=64,
        top_k=2048, is_tnd=True, seqlens_q=[128], seqlens_kv=[128])

    pta_out = _run_pta(
        inputs, scale_value=1.0, layout="TND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536,
        seqlens_q=[128], seqlens_kv=[128], use_bf16=use_bf16)
    ms_out = _run_ms(
        inputs, scale_value=1.0, layout="TND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536,
        seqlens_q=[128], seqlens_kv=[128], use_bf16=use_bf16)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)
    allclose_nparray(pta_out[2], ms_out[2], 0, 0)
    allclose_nparray(pta_out[3], ms_out[3], 0, 0)


# =========================================================================
# Test: TND, multi-batch ragged actual_seq lengths
# =========================================================================
def test_tnd_multi_batch_ragged_actual_seq():
    """
    Feature: MS vs PTA bit-exact comparison
    Description: TND, B=2 with ragged actual_seq_qlen/actual_seq_klen.
    Expectation: All 4 outputs match PTA bit-for-bit.
    """
    seqlens_q = [64, 128]
    seqlens_kv = [96, 128]
    inputs = _gen_tnd_flattened_inputs_np(
        seed=7778, seqlens_q=seqlens_q, seqlens_kv=seqlens_kv,
        n1=64, n2=1, nidx1=64, nidx2=1,
        dim=512, d_index=128, dr=64, top_k=2048)

    pta_out = _run_pta(
        inputs, scale_value=1.0, layout="TND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536,
        seqlens_q=seqlens_q, seqlens_kv=seqlens_kv)
    ms_out = _run_ms(
        inputs, scale_value=1.0, layout="TND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536,
        seqlens_q=seqlens_q, seqlens_kv=seqlens_kv)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)
    allclose_nparray(pta_out[2], ms_out[2], 0, 0)
    allclose_nparray(pta_out[3], ms_out[3], 0, 0)


# =========================================================================
# Test: TND, scale=0.5, Nidx1 doc matrix
# =========================================================================
@pytest.mark.parametrize(
    'nidx1,seed',
    [(8, 70), (16, 71), (32, 77), (64, 78)],
    ids=[
        'nidx1_8_seed_70',
        'nidx1_16_seed_71',
        'nidx1_32_seed_77',
        'nidx1_64_seed_78',
    ],
)
def test_tnd_scale_0p5_nidx1_matrix(nidx1, seed):
    """
    Feature: MS vs PTA bit-exact comparison
    Description: TND, scale=0.5, doc-supported Nidx1 values (8/16/32/64).
    Expectation: All 4 outputs match PTA bit-for-bit.
    """
    inputs = _gen_inputs_np(
        seed=seed, batch=1, s1=128, s2=128, n1=64, n2=1,
        nidx1=nidx1, nidx2=1, dim=512, d_index=128, dr=64,
        top_k=2048, is_tnd=True, seqlens_q=[128], seqlens_kv=[128])

    pta_out = _run_pta(
        inputs, scale_value=0.5, layout="TND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536,
        seqlens_q=[128], seqlens_kv=[128])
    ms_out = _run_ms(
        inputs, scale_value=0.5, layout="TND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536,
        seqlens_q=[128], seqlens_kv=[128])

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)
    allclose_nparray(pta_out[2], ms_out[2], 0, 0)
    allclose_nparray(pta_out[3], ms_out[3], 0, 0)


# =========================================================================
# Test: TND, N2/Nidx2 > 1
# =========================================================================
def test_tnd_multi_head_kv_and_index():
    """
    Feature: MS vs PTA bit-exact comparison
    Description: TND, N2=4 and Nidx2=4 to cover multi-head KV/index paths.
    Expectation: All 4 outputs match PTA bit-for-bit.
    """
    inputs = _gen_inputs_np(
        seed=333, batch=1, s1=128, s2=128, n1=64, n2=4,
        nidx1=64, nidx2=4, dim=512, d_index=128, dr=64,
        top_k=2048, is_tnd=True, seqlens_q=[128], seqlens_kv=[128])

    pta_out = _run_pta(
        inputs, scale_value=1.0, layout="TND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536,
        seqlens_q=[128], seqlens_kv=[128])
    ms_out = _run_ms(
        inputs, scale_value=1.0, layout="TND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536,
        seqlens_q=[128], seqlens_kv=[128])

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)
    allclose_nparray(pta_out[2], ms_out[2], 0, 0)
    allclose_nparray(pta_out[3], ms_out[3], 0, 0)


# =========================================================================
# Test: TND, scale=2.0
# =========================================================================
def test_tnd_scale_2p0():
    """
    Feature: MS vs PTA bit-exact comparison
    Description: TND, scale=2.0, seed=88.
    Expectation: All 4 outputs match PTA bit-for-bit.
    """
    inputs = _gen_inputs_np(
        seed=88, batch=1, s1=128, s2=128, n1=64, n2=1,
        nidx1=64, nidx2=1, dim=512, d_index=128, dr=64,
        top_k=2048, is_tnd=True, seqlens_q=[128], seqlens_kv=[128])

    pta_out = _run_pta(
        inputs, scale_value=2.0, layout="TND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536,
        seqlens_q=[128], seqlens_kv=[128])
    ms_out = _run_ms(
        inputs, scale_value=2.0, layout="TND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536,
        seqlens_q=[128], seqlens_kv=[128])

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)
    allclose_nparray(pta_out[2], ms_out[2], 0, 0)
    allclose_nparray(pta_out[3], ms_out[3], 0, 0)


# =========================================================================
# Test: TND, default tokens (INT64_MAX)
# =========================================================================
def test_tnd_default_tokens():
    """
    Feature: MS vs PTA bit-exact comparison
    Description: TND, pre_tokens=next_tokens=INT64_MAX, seed=42.
    Expectation: All 4 outputs match PTA bit-for-bit.
    """
    inputs = _gen_inputs_np(
        seed=42, batch=1, s1=128, s2=128, n1=64, n2=1,
        nidx1=64, nidx2=1, dim=512, d_index=128, dr=64,
        top_k=2048, is_tnd=True, seqlens_q=[128], seqlens_kv=[128])

    pta_out = _run_pta(
        inputs, scale_value=1.0, layout="TND",
        sparse_mode=3, pre_tokens=INT64_MAX, next_tokens=INT64_MAX,
        seqlens_q=[128], seqlens_kv=[128])
    ms_out = _run_ms(
        inputs, scale_value=1.0, layout="TND",
        sparse_mode=3, pre_tokens=INT64_MAX, next_tokens=INT64_MAX,
        seqlens_q=[128], seqlens_kv=[128])

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)
    allclose_nparray(pta_out[2], ms_out[2], 0, 0)
    allclose_nparray(pta_out[3], ms_out[3], 0, 0)


# =========================================================================
# Test: BSND, scale=0.5
# =========================================================================
def test_bsnd_scale_0p5():
    """
    Feature: MS vs PTA bit-exact comparison
    Description: BSND, scale=0.5, seed=99.
    Expectation: All 4 outputs match PTA bit-for-bit.
    """
    inputs = _gen_inputs_np(
        seed=99, batch=1, s1=128, s2=128, n1=64, n2=1,
        nidx1=64, nidx2=1, dim=512, d_index=128, dr=64,
        top_k=2048, is_tnd=False)

    pta_out = _run_pta(
        inputs, scale_value=0.5, layout="BSND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536)
    ms_out = _run_ms(
        inputs, scale_value=0.5, layout="BSND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)
    allclose_nparray(pta_out[2], ms_out[2], 0, 0)
    allclose_nparray(pta_out[3], ms_out[3], 0, 0)


# =========================================================================
# Test: BSND, scale=2.0
# =========================================================================
def test_bsnd_scale_2p0():
    """
    Feature: MS vs PTA bit-exact comparison
    Description: BSND, scale=2.0, seed=188.
    Expectation: All 4 outputs match PTA bit-for-bit.
    """
    inputs = _gen_inputs_np(
        seed=188, batch=1, s1=128, s2=128, n1=64, n2=1,
        nidx1=64, nidx2=1, dim=512, d_index=128, dr=64,
        top_k=2048, is_tnd=False)

    pta_out = _run_pta(
        inputs, scale_value=2.0, layout="BSND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536)
    ms_out = _run_ms(
        inputs, scale_value=2.0, layout="BSND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)
    allclose_nparray(pta_out[2], ms_out[2], 0, 0)
    allclose_nparray(pta_out[3], ms_out[3], 0, 0)


# =========================================================================
# Test: BSND, default tokens (INT64_MAX)
# =========================================================================
def test_bsnd_default_tokens():
    """
    Feature: MS vs PTA bit-exact comparison
    Description: BSND, pre_tokens=next_tokens=INT64_MAX, seed=142.
    Expectation: All 4 outputs match PTA bit-for-bit.
    """
    inputs = _gen_inputs_np(
        seed=142, batch=1, s1=128, s2=128, n1=64, n2=1,
        nidx1=64, nidx2=1, dim=512, d_index=128, dr=64,
        top_k=2048, is_tnd=False)

    pta_out = _run_pta(
        inputs, scale_value=1.0, layout="BSND",
        sparse_mode=3, pre_tokens=INT64_MAX, next_tokens=INT64_MAX)
    ms_out = _run_ms(
        inputs, scale_value=1.0, layout="BSND",
        sparse_mode=3, pre_tokens=INT64_MAX, next_tokens=INT64_MAX)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)
    allclose_nparray(pta_out[2], ms_out[2], 0, 0)
    allclose_nparray(pta_out[3], ms_out[3], 0, 0)


# =========================================================================
# Test: TND, PTA weight distribution
# =========================================================================
def test_tnd_pta_weights():
    """
    Feature: MS vs PTA bit-exact comparison
    Description: TND, pta_weights=True (narrow range), seed=42.
    Expectation: All 4 outputs match PTA bit-for-bit.
    """
    inputs = _gen_inputs_np(
        seed=42, batch=1, s1=128, s2=128, n1=64, n2=1,
        nidx1=64, nidx2=1, dim=512, d_index=128, dr=64,
        top_k=2048, is_tnd=True, seqlens_q=[128], seqlens_kv=[128],
        pta_weights=True)

    pta_out = _run_pta(
        inputs, scale_value=1.0, layout="TND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536,
        seqlens_q=[128], seqlens_kv=[128])
    ms_out = _run_ms(
        inputs, scale_value=1.0, layout="TND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536,
        seqlens_q=[128], seqlens_kv=[128])

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)
    allclose_nparray(pta_out[2], ms_out[2], 0, 0)
    allclose_nparray(pta_out[3], ms_out[3], 0, 0)


# =========================================================================
# Test: BSND, PTA weight distribution
# =========================================================================
def test_bsnd_pta_weights():
    """
    Feature: MS vs PTA bit-exact comparison
    Description: BSND, pta_weights=True (narrow range), seed=242.
    Expectation: All 4 outputs match PTA bit-for-bit.
    """
    inputs = _gen_inputs_np(
        seed=242, batch=1, s1=128, s2=128, n1=64, n2=1,
        nidx1=64, nidx2=1, dim=512, d_index=128, dr=64,
        top_k=2048, is_tnd=False, pta_weights=True)

    pta_out = _run_pta(
        inputs, scale_value=1.0, layout="BSND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536)
    ms_out = _run_ms(
        inputs, scale_value=1.0, layout="BSND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)
    allclose_nparray(pta_out[2], ms_out[2], 0, 0)
    allclose_nparray(pta_out[3], ms_out[3], 0, 0)


# =========================================================================
# Test: TND topK doc matrix (1024 ~ 8192)
# =========================================================================
@pytest.mark.parametrize('topk,seed', [
    (1024, 210), (2048, 42), (3072, 214), (4096, 211),
    (5120, 216), (6144, 217), (7168, 218), (8192, 219),
])
def test_tnd_topk_doc_matrix(topk, seed):
    """
    Feature: MS vs PTA bit-exact comparison
    Description: TND, doc-declared topK values (CANN 9.0).
    Expectation: All 4 outputs match PTA bit-for-bit.
    """
    inputs = _gen_inputs_np(
        seed=seed, batch=1, s1=128, s2=128, n1=64, n2=1,
        nidx1=64, nidx2=1, dim=512, d_index=128, dr=64,
        top_k=topk, is_tnd=True, seqlens_q=[128], seqlens_kv=[128])

    pta_out = _run_pta(
        inputs, scale_value=1.0, layout="TND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536,
        seqlens_q=[128], seqlens_kv=[128])
    ms_out = _run_ms(
        inputs, scale_value=1.0, layout="TND",
        sparse_mode=3, pre_tokens=65536, next_tokens=65536,
        seqlens_q=[128], seqlens_kv=[128])

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)
    allclose_nparray(pta_out[2], ms_out[2], 0, 0)
    allclose_nparray(pta_out[3], ms_out[3], 0, 0)


# =========================================================================
# Test: rope=None (skipped — framework limitation)
# =========================================================================
@pytest.mark.skip(reason="rope=None not supported: MS ops.Custom passes "
                         "nullptr; PTA rejects both-None (block_dim invalid)")
def test_bsnd_no_rope():
    """
    Feature: MS vs PTA bit-exact comparison
    Description: BSND, DR=0 (no rope).
    Expectation: Skipped — both frameworks reject rope=None.
    """
