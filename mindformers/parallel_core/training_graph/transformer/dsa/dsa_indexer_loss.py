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
"""DSA Indexer Loss."""
import mindspore.common.dtype as mstype
from mindspore import nn, ops, Tensor
from mindspore.ops import auto_generate as aclnn_ops
from mindspore.ops import operations as P
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode

from mindformers.tools.utils import get_real_rank
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.parallel_core.training_graph.device_matrix import layout
from mindformers.parallel_core.training_graph.communication import get_dp_cp_id, get_cp_group_name
from mindformers.parallel_core.training_graph.transformer.dsa.utils import adjust_bsnd_input, adjust_tnd_input
from mindformers.parallel_core.training_graph.ops.dense_lightning_indexer_grad_kl_loss import (
    DenseLightningIndexerGradKLLoss
)
from mindformers.parallel_core.training_graph.ops.sparse_lightning_indexer_grad_kl_loss import (
    SparseLightningIndexerGradKLLoss
)


def loss_infer_shape(*args):
    dq_shape = args[2]
    dk_shape = args[3]
    dw_shape = args[4]
    return dq_shape, dk_shape, dw_shape, [1]


def loss_infer_dtype(*args):
    return args[2], args[3], args[4], mstype.float32


class DSAIndexerLoss(nn.Cell):
    """
    Compute KL divergence loss between index_scores and true attention_scores.

    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.

    Reference:
        Section 2.1 of https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

    Args:
        config (TransformerConfig): The configuration for the transformer model.
        softmax_scale: Scale coefficient after q @ k^T.
    """
    def __init__(self, config: MLATransformerConfig, softmax_scale=None):
        super().__init__()
        self.input_layout = config.input_layout
        self.is_tnd = config.input_layout == "TND"
        self.sparse_loss = config.dsa_indexer_use_sparse_loss
        self.nope_dim = config.kv_lora_rank if self.sparse_loss else config.qk_head_dim
        self.pe_dim = config.qk_pos_emb_head_dim
        self.loss_coeff = config.dsa_indexer_loss_coeff
        self.sparse_count = config.dsa_indexer_topk
        self.num_attention_heads = config.num_attention_heads
        self.is_tnd = config.input_layout == "TND"
        self.cp = config.context_parallel_size
        self.tp = config.tensor_model_parallel_size
        softmax_scale = softmax_scale or config.kv_channels ** -0.5

        self.softmax_scale = Tensor([softmax_scale], mstype.float32)

        _, cp_id = get_dp_cp_id(config)
        self.offset_id = cp_id
        if self.cp > 1:
            self.cp_group = get_cp_group_name(
                get_real_rank(),
                config.data_parallel_size,
                config.tensor_model_parallel_size,
                config.context_parallel_size
            )[0]
            self.cp_allreduce = ops.AllReduce("sum", self.cp_group)

        # common operators
        self.cast = aclnn_ops.Cast()
        self.reshape = aclnn_ops.Reshape()
        self.split_q = aclnn_ops.SplitWithSize()
        self.split_k = aclnn_ops.SplitWithSize()
        # morph operators
        self.pad = aclnn_ops.ConstantPadND()
        if self.sparse_loss:
            self.loss_op = SparseLightningIndexerGradKLLoss()
            self.compute_indexer_loss = P.Morph(
                self._sparse_indexer_loss,
                loss_infer_shape,
                loss_infer_dtype
            ).add_prim_attr("self_define_shard", True)
        else:
            self.loss_op = DenseLightningIndexerGradKLLoss()
            self.compute_indexer_loss = P.Morph(
                self._dense_indexer_loss,
                loss_infer_shape,
                loss_infer_dtype
            ).add_prim_attr("self_define_shard", True)
        self.zeros_like = aclnn_ops.ZerosLikeExt()

        if _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
            self.shard()

    def _dense_indexer_loss(
            self, q, k, q_index, k_index, weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index,
            query_rope, key_rope, actual_seq_qlen=None, actual_seq_klen=None
    ):
        """get dense indexer loss and gradient with fused operator"""
        if self.is_tnd:
            loss_scale = q.shape[0]
            actual_seq_qlen, actual_seq_klen = adjust_tnd_input(self.offset_id, q, k, actual_seq_qlen, actual_seq_klen)
        elif self.cp > 1:
            loss_scale = q.shape[0] * q.shape[1]
            origin_seq_len = k.shape[1]
            k = adjust_bsnd_input(self.offset_id, q, k)
            k_index = adjust_bsnd_input(self.offset_id, q, k_index)
            key_rope = adjust_bsnd_input(self.offset_id, q, key_rope)
        else:
            loss_scale = q.shape[0] * q.shape[1]

        d_q, d_k, d_w, loss = self.loss_op(
            q, k, q_index, k_index, weights,
            softmax_max, softmax_sum,
            softmax_max_index, softmax_sum_index,
            scale_value=self.softmax_scale,
            query_rope=query_rope, key_rope=key_rope,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout=self.input_layout
        )
        d_q = d_q / loss_scale / 8 / self.tp / self.cp
        d_k = d_k / loss_scale / 8 / self.tp / self.cp
        d_w = d_w / loss_scale / 8 / self.tp / self.cp
        loss = loss / loss_scale
        if self.cp > 1:
            loss = self.cp_allreduce(loss) / self.cp
            if not self.is_tnd:
                d_k = self.pad(d_k, (0, 0, 0, 0, 0, origin_seq_len - d_k.shape[1], 0, 0))
        return d_q, d_k, d_w, loss

    def _sparse_indexer_loss(
            self, q, k, q_index, k_index, weights, topk_indices, softmax_max, softmax_sum,
            query_rope, key_rope, actual_seq_qlen=None, actual_seq_klen=None
    ):
        """get sparse indexer loss and gradient with fused operator"""
        if self.is_tnd:
            loss_scale = q.shape[0]
            actual_seq_qlen, actual_seq_klen = adjust_tnd_input(self.offset_id, q, k, actual_seq_qlen, actual_seq_klen)
        elif self.cp > 1:
            loss_scale = q.shape[0] * q.shape[1]
            origin_seq_len = k.shape[1]
            k = adjust_bsnd_input(self.offset_id, q, k)
            k_index = adjust_bsnd_input(self.offset_id, q, k_index)
            key_rope = adjust_bsnd_input(self.offset_id, q, key_rope)
        else:
            loss_scale = q.shape[0] * q.shape[1]

        d_q, d_k, d_w, loss = self.loss_op(
            q, k, q_index, k_index, weights, topk_indices,
            softmax_max, softmax_sum,
            scale_value=self.softmax_scale,
            query_rope=query_rope, key_rope=key_rope,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout=self.input_layout
        )
        d_q = d_q / loss_scale / 8 / self.tp / self.cp
        d_k = d_k / loss_scale / 8 / self.tp / self.cp
        d_w = d_w / loss_scale / 8 / self.tp / self.cp
        loss = loss / loss_scale
        if self.cp > 1:
            loss = self.cp_allreduce(loss) / self.cp
            if not self.is_tnd:
                d_k = self.pad(d_k, (0, 0, 0, 0, 0, origin_seq_len - d_k.shape[1], 0, 0))
        return d_q, d_k, d_w, loss

    def construct(
            self, query, key, query_index, key_index, weights,
            topk_indices, softmax_max, softmax_sum,
            softmax_max_index=None, softmax_sum_index=None,
            actual_seq_qlen=None, actual_seq_klen=None
    ):
        """compute indexer loss"""
        q_nope, q_rope = self.split_q(query, [self.nope_dim, self.pe_dim], -1)
        k_nope, k_rope = self.split_k(key, [self.nope_dim, self.pe_dim], -1)
        if self.sparse_loss:
            d_q, d_k, d_w, loss = self.compute_indexer_loss(
                q_nope, k_nope, query_index, key_index, weights, topk_indices,
                softmax_max, softmax_sum, q_rope, k_rope, actual_seq_qlen, actual_seq_klen
            )
        else:
            d_q, d_k, d_w, loss = self.compute_indexer_loss(
                q_nope, k_nope, query_index, key_index, weights,
                softmax_max, softmax_sum, softmax_max_index, softmax_sum_index,
                q_rope, k_rope, actual_seq_qlen, actual_seq_klen
            )

        return d_q, d_k, d_w, loss

    # pylint: disable=W0612, W0613
    def bprop(
            self, query, key, query_index, key_index, weights, topk_indices,
            softmax_max, softmax_sum, softmax_max_index, softmax_sum_index,
            actual_seq_qlen, actual_seq_klen, out, dout
    ):
        """backward."""
        d_loss = dout[3] # add this useless line for pipeline parallel in dense stage
        d_q, d_k = self.zeros_like(query), self.zeros_like(key)
        d_qi = out[0]
        d_ki = out[1]
        d_w = out[2]
        d_topk = self.zeros_like(topk_indices)
        d_softmax1, d_softmax2 = self.zeros_like(softmax_max), self.zeros_like(softmax_sum)
        if not self.sparse_loss:
            d_softmax_i1, d_softmax_i2 = self.zeros_like(softmax_max_index), self.zeros_like(softmax_sum_index)
        else:
            d_softmax_i1, d_softmax_i2 = None, None
        if self.is_tnd:
            d_qlen, d_klen = self.zeros_like(actual_seq_qlen), self.zeros_like(actual_seq_klen)
        else:
            d_qlen, d_klen = None, None
        return d_q, d_k, d_qi, d_ki, d_w, d_topk, d_softmax1, d_softmax2, d_softmax_i1, d_softmax_i2, d_qlen, d_klen

    def shard(self):
        """Set parallel strategy."""
        if self.is_tnd:
            q_shard = layout("dp_cp", "None", "None")
            k_shard = layout("dp", "None", "None")
            w_shard = layout("dp_cp", "None")
            topk_shard = layout("dp_cp", "None", "None")
            if self.sparse_loss:
                softmax_shard = layout("None", "dp_cp", "None")
                loss_shard = (q_shard, k_shard, q_shard, k_shard, w_shard, topk_shard,
                              softmax_shard, softmax_shard, q_shard, k_shard, layout("dp"), layout("dp"))
            else:
                softmax_shard = layout("dp_cp", "None", "None")
                softmax_index_shard = layout("None", "dp_cp")
                loss_shard = (q_shard, k_shard, q_shard, k_shard, w_shard, softmax_shard, softmax_shard,
                              softmax_index_shard, softmax_index_shard, q_shard, k_shard, layout("dp"), layout("dp"))
        else:
            q_shard = layout("dp", "cp", "None", "None")
            k_shard = layout("dp", "None", "None", "None")
            w_shard = layout("dp", "cp", "None")
            topk_shard = layout("dp", "cp", "None", "None")
            softmax_shard = layout("dp", "None", "cp", "None")
            if self.sparse_loss:
                loss_shard = (q_shard, k_shard, q_shard, k_shard, w_shard, topk_shard,
                              softmax_shard, softmax_shard, q_shard, k_shard)
            else:
                softmax_index_shard = layout("dp", "None", "cp")
                loss_shard = (q_shard, k_shard, q_shard, k_shard, w_shard, softmax_shard, softmax_shard,
                              softmax_index_shard, softmax_index_shard, q_shard, k_shard)
        self.split_q.shard((q_shard,))
        self.split_k.shard((k_shard,))

        self.compute_indexer_loss.shard(
            loss_shard, (q_shard, k_shard, w_shard, layout("None"))
        )
