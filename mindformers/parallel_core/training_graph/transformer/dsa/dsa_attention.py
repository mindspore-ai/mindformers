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
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import nn, ops, Parameter, Tensor
from mindspore.ops import operations as P
from mindspore.ops import auto_generate as aclnn_ops
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.parallel_core.training_graph.transformer.dsa.dsa_indexer_loss import DSAIndexerLoss
from mindformers.parallel_core.training_graph.device_matrix import layout
from mindformers.parallel_core.training_graph.communication import get_dp_cp_id
from mindformers.parallel_core.training_graph.transformer.mask_generate import CausalEODMaskGenerate
from mindformers.parallel_core.training_graph.transformer.dsa.utils import adjust_bsnd_input, adjust_tnd_input


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
        self.seq_length = config.seq_length
        self.head_num = config.num_attention_heads
        self.layer_number = layer_number
        self.attention_mode = 2 # currently, attention mode only support 2
        self.cp = config.context_parallel_size
        self.sparse_loss = config.dsa_indexer_use_sparse_loss
        self.mask_compression = config.use_attn_mask_compression and not self.is_tnd

        _, cp_id = get_dp_cp_id(config)
        self.offset_id = cp_id

        self.indexer = build_module(
            submodules.indexer, config=config
        )
        self.softmax_scale = softmax_scale or config.kv_channels ** -0.5
        self.indexer_loss = DSAIndexerLoss(config, self.softmax_scale)

        self.track_max_attention_logit = config.track_max_attention_logit

        if self.track_max_attention_logit:
            # Parameter to store the maximum attention logit value per head.
            # Note: This is a local max within each device's partition. Cross-device
            # synchronization (AllReduce-Max across DP/CP dimensions) is performed
            # later in GPTModel.allreduce_max_attention_logit() to obtain the global max.
            self.max_logits_val = Parameter(
                Tensor(np.zeros(self.head_num), dtype=mstype.float32),
                parallel_optimizer=False, requires_grad=False
            )
            self.reduce_max = aclnn_ops.ReduceMax().add_prim_attr("self_define_shard", True)
            self.assign = ops.Assign().add_prim_attr("self_define_shard", True)
            self.maximum = ops.Maximum().add_prim_attr("self_define_shard", True)

        # common operators
        self.cast = aclnn_ops.Cast()
        self.reshape = aclnn_ops.Reshape()
        self.split_q = aclnn_ops.SplitWithSize()
        self.split_k = aclnn_ops.SplitWithSize()
        if self.sparse_loss:
            # Sparse stage use SparseFlashAttention.
            self.sparse_flash_attention = P.Morph(
                self._sparse_flash_attention_forward,
                self.sfa_infer_shape,
                lambda *args: (args[0], mstype.float32, mstype.float32)
            ).add_prim_attr("self_define_shard", True)
        else:
            # Dense stage use common FlashAttention.
            # the softmax_out of FA in TND layout is hard to convert to the correct layout
            # and the convert process notably increase the compile time (about 10x),
            # so fall to use BSND + full eod_mask
            self.dense_flash_attention = FlashAttentionScore(
                head_num=self.head_num,
                scale_value=self.softmax_scale,
                inner_precise=0,
                input_layout="BSND",
                sparse_mode=config.sparse_mode if not self.is_tnd else 0
            )
            self.eod_mask_generator = CausalEODMaskGenerate(config)
            self.softmax_transpose = aclnn_ops.Transpose()
            self.slice =  aclnn_ops.StridedSlice().add_prim_attr("self_define_shard", True)
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
            actual_seq_qlen, actual_seq_kvlen = adjust_tnd_input(self.offset_id, q, k,
                                                                 actual_seq_qlen, actual_seq_kvlen)
        elif self.cp > 1:
            k = adjust_bsnd_input(self.offset_id, q, k)
            v = adjust_bsnd_input(self.offset_id, q, v)
            key_rope = adjust_bsnd_input(self.offset_id, q, key_rope)

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
            self, query, key, value, topk_indices, attention_mask=None,
            actual_seq_qlen=None, actual_seq_kvlen=None
    ):
        """Forward pass for Sparse Attention."""
        if self.sparse_loss:
            q_nope, q_rope = self.split_q(query, [self.nope_dim, self.pe_dim], -1)
            k_nope, k_rope = self.split_k(key, [self.nope_dim, self.pe_dim], -1)
            attention_output, softmax_max, softmax_sum = self.sparse_flash_attention(
                q_nope, k_nope, value, topk_indices,
                q_rope, k_rope, actual_seq_qlen, actual_seq_kvlen
            )
            if self.track_max_attention_logit:
                if self.is_tnd:
                    max_logits = self.reduce_max(softmax_max, (0, 1))
                else:
                    max_logits = self.reduce_max(softmax_max, (0, 1, 2))
                self.assign(self.max_logits_val, self.maximum(self.max_logits_val, max_logits))
        else:
            # During dense warmup stage, common FlashAttention will be applied instead of SparseFlashAttention
            if self.is_tnd:
                t, n, _ = query.shape
                b, s = t // self.seq_length, self.seq_length
                attention_mask = self.eod_mask_generator(self.reshape(actual_seq_qlen, (b, -1)))
                query = self.reshape(query, (b, s, n, -1))
                key = self.reshape(key, (b, s, n, -1))
                value = self.reshape(value, (b, s, n, -1))
                actual_seq_qlen = None
                actual_seq_kvlen = None

            softmax_max, softmax_sum, _, attention_output = self.dense_flash_attention(
                query, key, value,
                None, None, None,
                attn_mask=attention_mask,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen
            )
            softmax_shape = softmax_max.shape[:-1]
            begin = (0,) * len(softmax_max.shape)
            step = (1,) * len(softmax_max.shape)
            softmax_max = self.slice(softmax_max, begin, (*softmax_shape, 1), step)
            softmax_sum = self.slice(softmax_sum, begin, (*softmax_shape, 1), step)
            if self.track_max_attention_logit:
                max_logits = self.reduce_max(softmax_max, (0, 2, 3))
                self.assign(self.max_logits_val, self.maximum(self.max_logits_val, max_logits))
            if self.is_tnd:
                # [b, s, n, d] -> [t, n, d]
                attention_output = self.reshape(attention_output, (t, n, -1))
                # [b, n, s, 1] -> [n, b, s, 1] -> [n, t, 1]
                softmax_max = self.reshape(self.softmax_transpose(softmax_max, (1, 0, 2, 3)), (n, t, 1))
                softmax_sum = self.reshape(self.softmax_transpose(softmax_sum, (1, 0, 2, 3)), (n, t, 1))

        return attention_output, softmax_max, softmax_sum

    def shard(self):
        """Set parallel strategy."""
        # in sparse stage, head_num of kv is 1 so tp shard is not applied.
        kv_head_split_num = "None" if self.sparse_loss else "tp"
        if self.is_tnd:
            q_shard = layout("dp_cp", "tp", "None")
            kv_shard = layout("dp", kv_head_split_num, "None")
            idx_shard = layout("dp_cp", "None", "None")
            attn_shard = layout("dp_cp", "tp", "None")
            softmax_shard = layout("None", "dp_cp", "tp") if self.sparse_loss else layout("dp_cp", "tp", "None")
            sfa_shard = (q_shard, kv_shard, kv_shard, idx_shard, q_shard, kv_shard, layout("dp"), layout("dp"))
        else:
            q_shard = layout("dp", "cp", "tp", "None")
            kv_shard = layout("dp", "None", kv_head_split_num, "None")
            idx_shard = layout("dp", "cp", "None", "None")
            attn_shard = layout("dp", "cp", "tp", "None")
            softmax_shard = layout("dp", "None", "cp", "tp") if self.sparse_loss else layout("dp", "tp", "cp", "None")
            sfa_shard = (q_shard, kv_shard, kv_shard, idx_shard, q_shard, kv_shard)
        self.split_q.shard((q_shard,))
        self.split_k.shard((kv_shard,))
        if self.sparse_loss:
            self.sparse_flash_attention.shard(sfa_shard, (attn_shard, softmax_shard, softmax_shard))
        else:
            fa_shard = (
                layout("dp", "cp", "tp", "None"),
                layout("dp", "None", kv_head_split_num, "None"),
                layout("dp", "None", kv_head_split_num, "None"),
                layout("None", "None") if self.mask_compression else layout("dp", "None", "cp", "None"),
            )
            softmax_shard = layout("dp", "tp", "cp", "None")
            attn_shard = layout("dp", "cp", "tp", "None")
            self.dense_flash_attention.shard(fa_shard, (softmax_shard, softmax_shard, layout("None"), attn_shard))
            self.slice.shard((softmax_shard,), (softmax_shard,))
            self.softmax_transpose.shard((layout("dp", "tp", "cp", "None"),))
        if self.track_max_attention_logit:
            self.assign.shard((layout("tp"), layout("tp")), (layout("tp"),))
            self.maximum.shard((layout("tp"), layout("tp")), (layout("tp"),))
            self.reduce_max.shard((softmax_shard,), (layout("tp"),))
