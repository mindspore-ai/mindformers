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
Custom operator wrapper for aclnnSparseLightningIndexerGradKLLoss.

Provides a ready-to-use MindSpore Cell for static graph (GRAPH_MODE) execution
via ops.Custom + CustomRegOp. No C++ compilation needed.

Usage:
    import mindspore as ms
    from mindspore import Tensor
    from sparse_lightning_indexer_grad_kl_loss import SparseLightningIndexerGradKLLoss

    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

    op = SparseLightningIndexerGradKLLoss()
    d_query_index, d_key_index, d_weights, loss = op(
        query, key, query_index, key_index, weights,
        sparse_indices, softmax_max, softmax_sum,
        scale_value=1.0,
        query_rope=query_rope,
        key_rope=key_rope,
        layout="BSND",
        sparse_mode=3,
        pre_tokens=65536,
        next_tokens=65536,
    )

Requirements:
    - MindSpore == master
    - Ascend 910B with CANN (aclnnSparseLightningIndexerGradKLLoss symbol available)
    - GRAPH_MODE with jit_level O0
"""
from mindspore.nn import Cell
from mindspore import ops
import mindspore.common.dtype as mstype

INT64_MAX = 9223372036854775807


def _infer_shape(*args):
    """Infer output shapes.

    Returns [query_index_shape, key_index_shape, weights_shape, [1]].
    Args positional order: query(0), key(1), query_index(2), key_index(3),
    weights(4), sparse_indices(5), softmax_max(6), softmax_sum(7),
    query_rope(8), key_rope(9), then attrs...
    """
    qi_s, ki_s, w_s = args[2], args[3], args[4]
    return [qi_s, ki_s, w_s, [1]]


def _infer_dtype(*args):
    """Infer output dtypes.

    Returns [query_index_dtype, key_index_dtype, weights_dtype, float32].
    """
    qi_t, ki_t, w_t = args[2], args[3], args[4]
    return [qi_t, ki_t, w_t, mstype.float32]


class SparseLightningIndexerGradKLLoss(Cell):
    """aclnnSparseLightningIndexerGradKLLoss wrapped as a MindSpore Cell.

    Computes the gradient of KL-loss for sparse lightning indexer attention.
    Supports both BSND (4D) and TND (3D) layouts.

    Args (in construct):
        query (Tensor[fp16/bf16]):           [B,S1,N1,D] or [T,N1,D]
        key (Tensor[fp16/bf16]):             [B,S2,N2,D] or [T,N2,D]
        query_index (Tensor[fp16/bf16]):     [B,S1,Nidx1,DIndex] or [T,Nidx1,DIndex]
        key_index (Tensor[fp16/bf16]):       [B,S2,Nidx2,DIndex] or [T,Nidx2,DIndex]
        weights (Tensor[fp16/bf16]):         [B,S1,Nidx1] or [T,Nidx1]
        sparse_indices (Tensor[int32]):      [B,S1,Nidx2,topK] or [T,Nidx2,topK]
        softmax_max (Tensor[fp32]):          [B,N2,S1,G] or [N2,T,G]
        softmax_sum (Tensor[fp32]):          [B,N2,S1,G] or [N2,T,G]
        scale_value (float):                 Scaling factor, default 1.0
        query_rope (Tensor[fp16/bf16]):      Rope for query. PTA accepts
                                             None, but this wrapper requires a
                                             real tensor because ops.Custom
                                             cannot pass None tensor inputs.
        key_rope (Tensor[fp16/bf16]):        Rope for key. PTA accepts None,
                                             but this wrapper requires a real
                                             tensor for the same reason.
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
        super().__init__()
        # _generate_get_workspace_size_func
        self._custom_op = ops.Custom(
            "aclnnSparseLightningIndexerGradKLLoss",
            out_shape=_infer_shape,
            out_dtype=_infer_dtype,
            func_type="aot",
            bprop=None,
        ).add_prim_attr("value_depend", [10, 11])

    def construct(self, query, key, query_index, key_index, weights,
                  sparse_indices, softmax_max, softmax_sum,
                  scale_value=1.0,
                  query_rope=None, key_rope=None,
                  actual_seq_qlen=None, actual_seq_klen=None,
                  layout="BSND", sparse_mode=3,
                  pre_tokens=INT64_MAX, next_tokens=INT64_MAX):
        """Forward pass. See class docstring for argument details."""
        if query_rope is None or key_rope is None:
            raise ValueError(
                "query_rope and key_rope are required and cannot be None. "
                "MindSpore ops.Custom does not support None tensor inputs "
                "(PTA uses undefined tensor as fallback, but MindSpore "
                "framework passes nullptr which is rejected). "
                "ACLNN also requires at least one rope to be non-null.")
        return self._custom_op(
            query, key, query_index, key_index, weights,
            sparse_indices, softmax_max, softmax_sum,
            query_rope, key_rope,
            actual_seq_qlen, actual_seq_klen,
            scale_value, layout, sparse_mode,
            pre_tokens, next_tokens, True)
