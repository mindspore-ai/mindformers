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
"""DeepSeek Sparse Attention."""
from dataclasses import dataclass
from typing import Union

import mindspore.common.dtype as mstype
from mindspore import nn, ops
from mindspore.ops import operations as P
from mindspore.ops import auto_generate as aclnn_ops
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.parallel_core.training_graph.transformer.dsa.dsa_indexer_loss import DSAIndexerLoss
from mindformers.parallel_core.training_graph.device_matrix import layout
from mindformers.parallel_core.training_graph.communication import get_dp_cp_id


@dataclass
class DSAttentionSubmodules:
    """
    Configuration class for specifying the submodules of DSAttention.

    Args:
        indexer: DSA Indexer module for computing sparse attention indices.
    """

    indexer: Union[ModuleSpec, type] = None


class DSAttention(nn.Cell):
    """
    This module implements sparse attention mechanism using an DSA Indexer to compute top-k attention indices
    for reducing computational complexity.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L491-L597
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: DSAttentionSubmodules,
        layer_number: int,
        attention_type: str = None,
        attn_mask_type: str = None,
        softmax_scale: float = None,
        cp_comm_type: str = None,
    ):
        super().__init__()
        if attn_mask_type:
            raise NotImplementedError("For FlashAttention, 'attn_mask_type' is not supported for now.")
        if attention_type:
            raise NotImplementedError("For FlashAttention, 'attention_type' is unused for now.")
        if cp_comm_type:
            raise NotImplementedError("For FlashAttention, 'cp_comm_type' is not supported for now.")
        self.nope_dim = config.kv_lora_rank
        self.pe_dim = config.qk_pos_emb_head_dim
        self.input_layout = config.input_layout
        self.is_tnd = config.input_layout == "TND"
        self.layer_number = layer_number
        self.attention_mode = 2 # currently, attention mode only support 2

        _, cp_id = get_dp_cp_id(config)
        self.offset_id = cp_id

        self.indexer = build_module(
            submodules.indexer, config=config
        )
        self.softmax_scale = softmax_scale or config.kv_channels ** -0.5
        self.indexer_loss = DSAIndexerLoss(config, self.softmax_scale)

        # common operators
        self.cast = aclnn_ops.Cast()
        self.transpose_q = aclnn_ops.Transpose()
        self.transpose_kv = aclnn_ops.Transpose()
        self.split_q = aclnn_ops.SplitWithSize()
        self.split_k = aclnn_ops.SplitWithSize()
        self.transpose_attn = aclnn_ops.Transpose()
        # morph operators
        self.clamp = aclnn_ops.ClampScalar()
        self.relu = aclnn_ops.ReLU()
        self.roll = aclnn_ops.Roll(1)
        self.cumsum = aclnn_ops.CumsumExt()
        self.sparse_flash_attention = P.Morph(
            self._sparse_flash_attention_forward,
            self.sfa_infer_shape,
            lambda *args: (args[0], mstype.float32, mstype.float32)
        ).add_prim_attr("self_define_shard", True)

        if _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
            self.shard()

    def sfa_infer_shape(self, *args):
        """sparse flash attention infer shape"""
        if self.is_tnd:
            tq, nq, _ = args[0]
            _, nk, _ = args[1]
            softmax_shape = (nk, tq, nq // nk)
        else:
            b, sq, nq, _ = args[0]
            _, _, nk, _ = args[1]
            softmax_shape = (b, nk, sq, nq // nk)
        return args[0], softmax_shape, softmax_shape

    def _sparse_flash_attention_forward(
        self, q, k, v, topk_indices,
        query_rope=None, key_rope=None,
        actual_seq_qlen=None, actual_seq_kvlen=None
    ):
        """get attention output with fused sfa operator"""
        if self.is_tnd:
            slice_tq = q.shape[0]
            slice_tk = k.shape[0]
            offset_q = slice_tq * self.offset_id
            # process actual_seq_len for fused operator to auto-generate mask in individual shard
            new_actual_seq_qlen = self.cast(self.clamp(actual_seq_qlen - offset_q, 0, slice_tq), mstype.int32)
            new_actual_seq_kvlen = actual_seq_kvlen - self.relu(actual_seq_qlen - offset_q) + new_actual_seq_qlen
            prev_seq_klen = self.roll(new_actual_seq_kvlen)
            prev_seq_klen[0] = 0
            new_actual_seq_kvlen = self.cumsum(self.relu(new_actual_seq_kvlen - prev_seq_klen), 0)
            new_actual_seq_kvlen[-1] = slice_tk
            actual_seq_qlen = new_actual_seq_qlen
            actual_seq_kvlen = new_actual_seq_kvlen

        attention_out, softmax_max, softmax_sum = ops.sparse_flash_attention(
            q, k, v, topk_indices, self.softmax_scale,
            query_rope=query_rope, key_rope=key_rope,
            actual_seq_lengths_query=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_kvlen,
            layout_query=self.input_layout,
            layout_kv=self.input_layout,
            attention_mode=self.attention_mode,
            return_softmax_lse=True
        )
        return attention_out, softmax_max, softmax_sum

    def construct(
            self, query, key, value, topk_indices,
            actual_seq_qlen=None, actual_seq_kvlen=None
    ):
        """Forward pass for Sparse Attention."""
        if not self.is_tnd:
            query = self.transpose_q(query, (1, 0, 2, 3))
            key = self.transpose_kv(key, (1, 0, 2, 3))
            value = self.transpose_kv(value, (1, 0, 2, 3))

        q_nope, q_rope = self.split_q(query, [self.nope_dim, self.pe_dim], -1)
        k_nope, k_rope = self.split_k(key, [self.nope_dim, self.pe_dim], -1)

        # ===================================
        # Run sparse attention kernel
        # ===================================
        attention_output = self.sparse_flash_attention(
            q_nope, k_nope, value, topk_indices,
            q_rope, k_rope, actual_seq_qlen, actual_seq_kvlen
        )[0]

        if not self.is_tnd:
            attention_output = self.transpose_attn(attention_output, (1, 0, 2, 3))

        return attention_output

    def shard(self):
        """Set parallel strategy."""
        self.transpose_q.shard((layout("cp", "dp", "tp", "None"),))
        self.transpose_kv.shard((layout("cp", "dp", "None", "None"),))
        self.transpose_attn.shard((layout("dp", "cp", "tp", "None"),))
        if self.is_tnd:
            q_shard = layout("dp_cp", "tp", "None")
            kv_shard = layout("dp", "None", "None")
            idx_shard = layout("dp_cp", "None", "None")
            sfa_shard = (q_shard, kv_shard, kv_shard, idx_shard, q_shard, kv_shard, layout("dp"), layout("dp"))
            attn_shard = layout("dp_cp", "tp", "None")
            softmax_shard = layout("None", "cp", "None")
        else:
            q_shard = layout("dp", "None", "tp", "None")
            kv_shard = layout("dp", "None", "None", "None")
            idx_shard = layout("dp", "None", "None", "None")
            attn_shard = layout("dp", "None", "tp", "None")
            softmax_shard = layout("dp", "None", "None", "None")
            sfa_shard = (q_shard, kv_shard, kv_shard, idx_shard, q_shard, kv_shard)
        self.split_q.shard((q_shard,))
        self.split_k.shard((kv_shard,))
        self.sparse_flash_attention.shard(sfa_shard, (attn_shard, softmax_shard, softmax_shard))
