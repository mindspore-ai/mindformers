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
"""
Custom operator wrapper for aclnnDenseLightningIndexerGradKLLoss.

Provides a ready-to-use MindSpore Cell for static graph (GRAPH_MODE) execution
via ops.Custom + CustomRegOp. No C++ compilation needed.

Usage:
    import mindspore as ms
    from mindspore import Tensor
    from custom_aclnn_dense_lightning import DenseLightningIndexerGradKLLoss

    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

    op = DenseLightningIndexerGradKLLoss()
    d_query_index, d_key_index, d_weights, loss = op(
        query, key, query_index, key_index, weights,
        softmax_max, softmax_sum, softmax_max_index, softmax_sum_index,
        scale_value=1.0,
        query_rope=query_rope,
        key_rope=key_rope,
        layout="BSND",
        sparse_mode=3,
        pre_tokens=65536,
        next_tokens=65536,
    )

Requirements:
    - MindSpore >= 2.3.0
    - Ascend 910B or 910C with CANN (aclnnDenseLightningIndexerGradKLLoss symbol available)
    - GRAPH_MODE with jit_level O0
"""
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops import DataType, CustomRegOp
import mindspore.common.dtype as mstype

INT64_MAX = 9223372036854775807

_ACLNN_WORKSPACE_SIGNATURE = (
    "const aclTensor* query, const aclTensor* key, "
    "const aclTensor* queryIndex, const aclTensor* keyIndex, "
    "const aclTensor* weights, const aclTensor* softmaxMax, "
    "const aclTensor* softmaxSum, const aclTensor* softmaxMaxIndex, "
    "const aclTensor* softmaxSumIndex, const aclTensor* queryRope, "
    "const aclTensor* keyRope, const aclIntArray* actualSeqQlen, "
    "const aclIntArray* actualSeqKlen, double scaleValue, "
    "const char* layout, int64_t sparseMode, int64_t preTokens, "
    "int64_t nextTokens, "
    "aclTensor* dQueryIndex, aclTensor* dKeyIndex, "
    "aclTensor* dWeights, aclTensor* loss, "
    "uint64_t* workspaceSize, aclOpExecutor** executor"
)

def _build_reg_info():
    """Build CustomRegOp registration info.

    Input order follows the aclnn kernel signature:
      11 required tensors, then scalar/array attrs, then 4 output tensors.
    """
    return CustomRegOp("aclnnDenseLightningIndexerGradKLLoss") \
        .input(0, "query", "required") \
        .input(1, "key", "required") \
        .input(2, "query_index", "required") \
        .input(3, "key_index", "required") \
        .input(4, "weights", "required") \
        .input(5, "softmax_max", "required") \
        .input(6, "softmax_sum", "required") \
        .input(7, "softmax_max_index", "required") \
        .input(8, "softmax_sum_index", "required") \
        .input(9, "query_rope", "required") \
        .input(10, "key_rope", "required") \
        .attr("actual_seq_qlen", "optional", "listInt") \
        .attr("actual_seq_klen", "optional", "listInt") \
        .attr("scale_value", "required", "float") \
        .attr("layout", "required", "str") \
        .attr("sparse_mode", "required", "int") \
        .attr("pre_tokens", "required", "int") \
        .attr("next_tokens", "required", "int") \
        .output(0, "d_query_index", "required") \
        .output(1, "d_key_index", "required") \
        .output(2, "d_weights", "required") \
        .output(3, "loss", "required") \
        .dtype_format(
            DataType.F16_Default, DataType.F16_Default,
            DataType.F16_Default, DataType.F16_Default,
            DataType.F16_Default, DataType.F32_Default,
            DataType.F32_Default, DataType.F32_Default,
            DataType.F32_Default, DataType.F16_Default,
            DataType.F16_Default, DataType.F16_Default,
            DataType.F16_Default, DataType.F16_Default,
            DataType.F32_Default,
        ) \
        .dtype_format(
            DataType.BF16_Default, DataType.BF16_Default,
            DataType.BF16_Default, DataType.BF16_Default,
            DataType.BF16_Default, DataType.F32_Default,
            DataType.F32_Default, DataType.F32_Default,
            DataType.F32_Default, DataType.BF16_Default,
            DataType.BF16_Default, DataType.BF16_Default,
            DataType.BF16_Default, DataType.BF16_Default,
            DataType.F32_Default,
        ) \
        .target("Ascend") \
        .get_op_info()

def _infer_shape(*args):
    """Infer output shapes.
    Returns [query_index_shape, key_index_shape, weights_shape, [1]].
    """
    qi_s, ki_s, w_s = args[2], args[3], args[4]
    return [qi_s, ki_s, w_s, [1]]

def _infer_dtype(*args):
    """Infer output dtypes.
    Returns [query_index_dtype, key_index_dtype, weights_dtype, float32].
    """
    qi_t, ki_t, w_t = args[2], args[3], args[4]
    return [qi_t, ki_t, w_t, mstype.float32]

class DenseLightningIndexerGradKLLoss(Cell):
    """aclnnDenseLightningIndexerGradKLLoss wrapped as a MindSpore Cell.

    Computes the gradient of KL-loss for dense lightning indexer attention.
    Supports both BSND (4D) and TND (3D) layouts.

    Args (in construct):
        query (Tensor[fp16/bf16]):           [B,S1,N1,D] or [T,N1,D]
        key (Tensor[fp16/bf16]):             [B,S2,N2,D] or [T,N2,D]
        query_index (Tensor[fp16/bf16]):     [B,S1,Nidx1,D] or [T,Nidx1,D]
        key_index (Tensor[fp16/bf16]):       [B,S2,Nidx2,D] or [T,Nidx2,D]
        weights (Tensor[fp16/bf16]):         [B,S1,Nidx1] or [T,Nidx1]
        softmax_max (Tensor[fp32]):     [B,N2,S1,G] or [N2,T,G]
        softmax_sum (Tensor[fp32]):     [B,N2,S1,G] or [N2,T,G]
        softmax_max_index (Tensor[fp32]): [B,N2idx,S1] or [N2idx,T]
        softmax_sum_index (Tensor[fp32]): [B,N2idx,S1] or [N2idx,T]
        scale_value (float):            Scaling factor, default 1.0
        query_rope (Tensor[fp16/bf16]): Rope for query (required)
        key_rope (Tensor[fp16/bf16]):   Rope for key (required)
        actual_seq_qlen (list[int], optional): Per-batch query lengths (TND)
        actual_seq_klen (list[int], optional): Per-batch key lengths (TND)
        layout (str):                   "BSND" or "TND", default "BSND"
        sparse_mode (int):              Default 3
        pre_tokens (int):               Default INT64_MAX
        next_tokens (int):              Default INT64_MAX

    Returns:
        tuple of 4 Tensors:
            d_query_index: same shape/dtype as query_index
            d_key_index:   same shape/dtype as key_index
            d_weights:     same shape/dtype as weights
            loss:          shape [1], dtype float32
    """
    def __init__(self):
        super(DenseLightningIndexerGradKLLoss, self).__init__()
        reg_info = _build_reg_info()
        self._custom_op = ops.Custom(
            "aclnnDenseLightningIndexerGradKLLoss",
            out_shape=_infer_shape,
            out_dtype=_infer_dtype,
            func_type="aot",
            bprop=None,
            reg_info=reg_info,
        )
        self._custom_op._generate_get_worspace_size_func_by_types(
            _ACLNN_WORKSPACE_SIGNATURE)

    def construct(self, query, key, query_index, key_index, weights,
                  softmax_max, softmax_sum, softmax_max_index, softmax_sum_index,
                  scale_value=1.0,
                  query_rope=None, key_rope=None,
                  actual_seq_qlen=None, actual_seq_klen=None,
                  layout="BSND", sparse_mode=3,
                  pre_tokens=INT64_MAX, next_tokens=INT64_MAX):
        """Forward pass. See class docstring for argument details."""
        if query_rope is None or key_rope is None:
            raise ValueError("query_rope and key_rope are required and cannot be None")
        return self._custom_op(
            query, key, query_index, key_index, weights,
            softmax_max, softmax_sum, softmax_max_index, softmax_sum_index,
            query_rope, key_rope,
            actual_seq_qlen, actual_seq_klen,
            scale_value, layout, sparse_mode,
            pre_tokens, next_tokens)
