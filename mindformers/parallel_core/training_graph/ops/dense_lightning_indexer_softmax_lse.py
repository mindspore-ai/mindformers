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
Custom operator wrapper for aclnnDenseLightningIndexerSoftmaxLse.

Provides a ready-to-use MindSpore Cell for static graph (GRAPH_MODE) execution
via ops.Custom + CustomRegOp. No C++ compilation needed.

Usage:
    import mindspore as ms
    from mindspore import Tensor
    from custom_aclnn_dense_lightning import DenseLightningIndexerSoftmaxLse

    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

    op = DenseLightningIndexerSoftmaxLse()
    softmax_max, softmax_sum = op(
        query_index, key_index, weight,
        actual_seq_qlen=None,    # optional list[int]
        actual_seq_klen=None,    # optional list[int]
        layout="BSND",
        sparse_mode=3,
        pre_tokens=9223372036854775807,
        next_tokens=9223372036854775807,
    )

Requirements:
    - MindSpore >= 2.3.0
    - Ascend 910B with CANN (aclnnDenseLightningIndexerSoftmaxLse symbol available)
    - GRAPH_MODE with jit_level O0
"""
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops import DataType, CustomRegOp
import mindspore.common.dtype as mstype

INT64_MAX = 9223372036854775807

_ACLNN_WORKSPACE_SIGNATURE = (
    "const aclTensor* queryIndex, const aclTensor* keyIndex, "
    "const aclTensor* weight, "
    "const aclIntArray* actualSeqLengthsQueryOptional, "
    "const aclIntArray* actualSeqLengthsKeyOptional, "
    "const char* layoutOptional, int64_t sparseMode, "
    "int64_t preTokens, int64_t nextTokens, "
    "aclTensor* softmaxMaxOut, aclTensor* softmaxSumOut, "
    "uint64_t* workspaceSize, aclOpExecutor** executor"
)


def _build_reg_info():
    """Build CustomRegOp registration info."""
    return CustomRegOp("aclnnDenseLightningIndexerSoftmaxLse") \
        .input(0, "query_index", "required") \
        .input(1, "key_index", "required") \
        .input(2, "weight", "required") \
        .attr("actual_seq_qlen", "optional", "listInt") \
        .attr("actual_seq_klen", "optional", "listInt") \
        .attr("layout", "required", "str") \
        .attr("sparse_mode", "required", "int") \
        .attr("pre_tokens", "required", "int") \
        .attr("next_tokens", "required", "int") \
        .output(0, "softmax_max", "required") \
        .output(1, "softmax_sum", "required") \
        .dtype_format(
            DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
            DataType.F32_Default, DataType.F32_Default,
        ) \
        .dtype_format(
            DataType.BF16_Default, DataType.BF16_Default, DataType.BF16_Default,
            DataType.F32_Default, DataType.F32_Default,
        ) \
        .target("Ascend") \
        .get_op_info()


def _infer_shape(*args):
    """Infer output shapes."""
    qi_shape = args[0]
    ki_shape = args[1]

    # In TND layout: query_index is [T1, Nidx1, D], key_index is [T2, Nidx2, D]
    # output should be: [Nidx2, T1]
    if len(qi_shape) == 3:
        # It's TND
        t1 = qi_shape[0]
        nidx2 = ki_shape[1] if len(ki_shape) > 1 else 1
        out_shape = [nidx2, t1]
    else:
        # It's BSND: query_index is [B, S1, Nidx1, D]
        # output should be: [B, Nidx2, S1]
        b = qi_shape[0]
        s1 = qi_shape[1]
        nidx2 = ki_shape[2] if len(ki_shape) > 2 else 1
        out_shape = [b, nidx2, s1]

    return [out_shape, out_shape]


def _infer_dtype(*args):
    """Infer output dtypes."""
    _ = args  # pylint: disable=unused-argument
    return [mstype.float32, mstype.float32]


class DenseLightningIndexerSoftmaxLse(Cell):
    """aclnnDenseLightningIndexerSoftmaxLse wrapped as a MindSpore Cell.

    Computes the SoftmaxLse (Max and Sum) for dense lightning indexer attention.
    Supports both BSND (4D) and TND (3D) layouts.

    Args (in construct):
        query_index (Tensor[fp16/bf16]):     [B, S1, Nidx1, DIndex] or [T1, Nidx1, DIndex]
        key_index (Tensor[fp16/bf16]):       [B, S2, Nidx2, DIndex] or [T2, Nidx2, DIndex]
        weight (Tensor[fp16/bf16]):          [B, S1, Nidx1] or [T1, Nidx1]
        actual_seq_qlen (list[int], optional): Per-batch query lengths. For TND layout, pass cumulative style.
        actual_seq_klen (list[int], optional): Per-batch key lengths. For TND layout, pass cumulative style.
        layout (str):                        "BSND" or "TND", default "BSND"
        sparse_mode (int):                   Default 3
        pre_tokens (int):                    Default INT64_MAX
        next_tokens (int):                   Default INT64_MAX

    Returns:
        tuple of 2 Tensors:
            softmax_max: shape [B, Nidx2, S1] or [Nidx2, T1], dtype float32
            softmax_sum: shape [B, Nidx2, S1] or [Nidx2, T1], dtype float32
    """
    def __init__(self):
        super(DenseLightningIndexerSoftmaxLse, self).__init__()
        reg_info = _build_reg_info()
        self._custom_op = ops.Custom(
            "aclnnDenseLightningIndexerSoftmaxLse",
            out_shape=_infer_shape,
            out_dtype=_infer_dtype,
            func_type="aot",
            bprop=None,
            reg_info=reg_info,
        )
        self._custom_op._generate_get_worspace_size_func_by_types(
            _ACLNN_WORKSPACE_SIGNATURE)

    def construct(self, query_index, key_index, weight,
                  actual_seq_qlen=None, actual_seq_klen=None,
                  layout="BSND", sparse_mode=3, pre_tokens=INT64_MAX, next_tokens=INT64_MAX):
        """Forward pass. See class docstring for argument details."""
        return self._custom_op(
            query_index, key_index, weight,
            actual_seq_qlen, actual_seq_klen,
            layout, sparse_mode,
            pre_tokens, next_tokens)
