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
aclnnDenseLightningIndexerSoftmaxLse.

Run from mindformers/ root:
    pytest -sv mindformers/parallel_core/training_graph/ops/\
test_dense_lightning_indexer_softmax_lse.py

Single case:
    pytest -sv mindformers/parallel_core/training_graph/ops/\
test_dense_lightning_indexer_softmax_lse.py::test_bsnd_base
"""
import os
import numpy as np
import pytest
import torch
import torch_npu

import mindspore as ms
from mindspore import Tensor, context
from mindformers.parallel_core.training_graph.ops.dense_lightning_indexer_softmax_lse import (
    DenseLightningIndexerSoftmaxLse, INT64_MAX
)


def _count_unequal_element(data_expected, data_me, rtol, atol):
    """Count elements failing |expected - actual| <= atol + rtol * |actual| and assert.

    Used as a secondary check when np.allclose fails, to surface mismatched values.
    Raises AssertionError if the fraction of failing elements is >= rtol.
    """
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    """Assert two numpy arrays are close (same semantics as np.allclose).

    If either side contains NaNs, uses np.allclose(..., equal_nan=...).
    Otherwise, on mismatch calls _count_unequal_element for a detailed failure message.
    """
    if np.any(np.isnan(data_expected)) or np.any(np.isnan(data_me)):
        assert np.allclose(data_expected, data_me, rtol, atol,
                           equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol,
                         equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape

# ---------------------------------------------------------------------------
# Global setup
# ---------------------------------------------------------------------------
DEVICE_ID = 7

# PyTorch setup
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
torch.use_deterministic_algorithms(True)

# MindSpore setup
ms.context.set_context(device_target="Ascend", device_id=DEVICE_ID, mode=ms.GRAPH_MODE,
                       jit_config={"jit_level": "O0"})
ms.context.set_context(deterministic="ON", pynative_synchronize=False)

# ---------------------------------------------------------------------------
# Inputs generation
# ---------------------------------------------------------------------------
def _gen_inputs_np(seed, b, s1, s2, nidx1, nidx2, d_index,
                   is_tnd=False, seqlens_q=None, seqlens_kv=None,
                   pta_weights=False):
    """Build random float16 numpy inputs for DenseLightningIndexerSoftmaxLse / PTA.

    Returns dict: query_index, key_index, weight, actual_seq_qlen, actual_seq_klen.
    BSND: shapes (B,S1,Nidx1,D), (B,S2,Nidx2,D), (B,S1,Nidx1); seqlens are None.
    TND (is_tnd=True, b==1): reshapes to (S1,Nidx1,D), (S2,Nidx2,D), (S1,Nidx1) and
    passes through seqlens_q / seqlens_kv as actual_seq_qlen / actual_seq_klen.
    pta_weights=True maps raw weights into a tighter range to match PTA-style init.
    """
    np.random.seed(seed)
    query_index = np.random.randn(b, s1, nidx1, d_index).astype(np.float16)
    key_index = np.random.randn(b, s2, nidx2, d_index).astype(np.float16)

    raw_w = np.random.randn(b, s1, nidx1).astype(np.float16)
    if pta_weights:
        a, b_val, kk = -0.05, 0.05, 3.0
        sc = (b_val - a) / (2 * kk)
        sh = (a + b_val) / 2
        weight = (raw_w * sc + sh).astype(np.float16)
    else:
        weight = (raw_w * 0.1 / 6.0).astype(np.float16)

    if is_tnd:
        if b == 1:
            query_index = query_index.reshape(s1, nidx1, d_index)
            key_index = key_index.reshape(s2, nidx2, d_index)
            weight = weight.reshape(s1, nidx1)

        final_seqlens_q = seqlens_q
        final_seqlens_kv = seqlens_kv
    else:
        final_seqlens_q = None
        final_seqlens_kv = None

    return {
        "query_index": query_index,
        "key_index": key_index,
        "weight": weight,
        "actual_seq_qlen": final_seqlens_q,
        "actual_seq_klen": final_seqlens_kv,
    }


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def _run_pta(inputs, layout="BSND", sparse_mode=3, pre_tokens=INT64_MAX, next_tokens=INT64_MAX, use_bf16=False):
    """Run torch_npu.npu_dense_lightning_indexer_softmax_lse on NPU; return outputs as numpy.

    Converts inputs from the dict produced by _gen_inputs_np to NPU tensors (optional bf16),
    calls the PTA custom op with layout/sparse/pre_tokens/next_tokens, and returns a tuple
    of float32 numpy arrays (bf16 outputs are cast to float32 on CPU).
    """
    def _to_npu(arr):
        if arr is None:
            return None
        t = torch.from_numpy(arr).npu()
        if use_bf16 and t.dtype == torch.float16:
            t = t.to(torch.bfloat16)
        return t

    results = torch_npu.npu_dense_lightning_indexer_softmax_lse(
        _to_npu(inputs["query_index"]),
        _to_npu(inputs["key_index"]),
        _to_npu(inputs["weight"]),
        actual_seq_qlen=inputs.get("actual_seq_qlen"),
        actual_seq_klen=inputs.get("actual_seq_klen"),
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

def _run_ms(inputs, layout="BSND", sparse_mode=3, pre_tokens=INT64_MAX, next_tokens=INT64_MAX, use_bf16=False):
    """Run DenseLightningIndexerSoftmaxLse on MindSpore; return outputs as numpy.

    Converts inputs from the dict produced by _gen_inputs_np to MS tensors (optional bf16),
    calls the MindSpore custom op with layout/sparse/pre_tokens/next_tokens, and returns a tuple
    of float32 numpy arrays (bf16 outputs are cast to float32). Uses a cache to prevent
    tensors from early GC.
    """
    op = DenseLightningIndexerSoftmaxLse()
    
    def _to_ms(arr):
        if arr is None:
            return None
        t = Tensor(arr)
        if use_bf16 and t.dtype == ms.float16:
            t = t.astype(ms.bfloat16)
        return t
        
    tensors = {
        "query_index": _to_ms(inputs["query_index"]),
        "key_index": _to_ms(inputs["key_index"]),
        "weight": _to_ms(inputs["weight"]),
    }
    # Bind to prevent early GC causing 0x25
    inputs["_tensors_cache"] = tensors

    results = op(
        tensors["query_index"],
        tensors["key_index"],
        tensors["weight"],
        actual_seq_qlen=inputs.get("actual_seq_qlen"),
        actual_seq_klen=inputs.get("actual_seq_klen"),
        layout=layout,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
    )

    def _out_np(t):
        if t.dtype == ms.bfloat16:
            return t.astype(ms.float32).asnumpy()
        return t.asnumpy()

    outputs_np = tuple(_out_np(r) for r in results)
    
    # Optional explicit cleanup if called in a loop
    inputs.pop("_tensors_cache", None)
    
    return outputs_np

# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_bsnd_base(dtype):
    """
    Feature: MS vs PTA bit-exact comparison
    Description: BSND layout, fp16/bf16 types
    Expectation: Outputs match PTA
    """
    use_bf16 = dtype == 'bf16'
    inputs = _gen_inputs_np(seed=42, b=1, s1=128, s2=128, nidx1=64, nidx2=1, d_index=128, is_tnd=False)
    
    pta_out = _run_pta(inputs, layout="BSND", use_bf16=use_bf16)
    ms_out = _run_ms(inputs, layout="BSND", use_bf16=use_bf16)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)


@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
@pytest.mark.parametrize('nidx1', [8, 16, 32, 64])
def test_bsnd_different_nidx1(dtype, nidx1):
    """
    Feature: MS vs PTA bit-exact comparison
    Description: BSND layout with various Nidx1 sizes (actual_seq lengths None)
    Expectation: Outputs match PTA
    """
    use_bf16 = dtype == 'bf16'
    inputs = _gen_inputs_np(seed=300 + nidx1, b=1, s1=128, s2=128, nidx1=nidx1, nidx2=1, d_index=128,
                            is_tnd=False)

    pta_out = _run_pta(inputs, layout="BSND", use_bf16=use_bf16)
    ms_out = _run_ms(inputs, layout="BSND", use_bf16=use_bf16)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)


@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_tnd_base(dtype):
    """
    Feature: MS vs PTA bit-exact comparison
    Description: TND layout, fp16/bf16 types
    Expectation: Outputs match PTA
    """
    use_bf16 = dtype == 'bf16'
    inputs = _gen_inputs_np(seed=100, b=1, s1=128, s2=128, nidx1=64, nidx2=1, d_index=128, 
                            is_tnd=True, seqlens_q=[128], seqlens_kv=[128])
    
    pta_out = _run_pta(inputs, layout="TND", use_bf16=use_bf16)
    ms_out = _run_ms(inputs, layout="TND", use_bf16=use_bf16)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)


@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
@pytest.mark.parametrize('nidx1', [8, 16, 32, 64])
def test_tnd_different_nidx1(dtype, nidx1):
    """
    Feature: MS vs PTA bit-exact comparison
    Description: TND layout with various Nidx1 sizes
    Expectation: Outputs match PTA
    """
    use_bf16 = dtype == 'bf16'
    inputs = _gen_inputs_np(seed=200 + nidx1, b=1, s1=128, s2=128, nidx1=nidx1, nidx2=1, d_index=128, 
                            is_tnd=True, seqlens_q=[128], seqlens_kv=[128])
    
    pta_out = _run_pta(inputs, layout="TND", use_bf16=use_bf16)
    ms_out = _run_ms(inputs, layout="TND", use_bf16=use_bf16)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)


@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
@pytest.mark.parametrize("layout", ["BSND", "TND"])
def test_pta_weights_distribution(dtype, layout):
    """
    Feature: MS vs PTA bit-exact comparison
    Description: Test with PTA-style truncated normal weight distribution
    Expectation: Outputs match PTA
    """
    use_bf16 = dtype == 'bf16'
    is_tnd = layout == "TND"
    seqlens_q = [128] if is_tnd else None
    seqlens_kv = [128] if is_tnd else None
    
    inputs = _gen_inputs_np(seed=300, b=1, s1=128, s2=128, nidx1=64, nidx2=1, d_index=128, 
                            is_tnd=is_tnd, seqlens_q=seqlens_q, seqlens_kv=seqlens_kv, pta_weights=True)
    
    pta_out = _run_pta(inputs, layout=layout, use_bf16=use_bf16)
    ms_out = _run_ms(inputs, layout=layout, use_bf16=use_bf16)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)
