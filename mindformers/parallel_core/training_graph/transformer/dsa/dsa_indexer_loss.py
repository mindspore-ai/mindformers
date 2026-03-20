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
from mindspore import nn, Tensor
from mindspore.ops import auto_generate as aclnn_ops
from mindspore.ops import operations as P
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode

from mindformers.parallel_core.training_graph.transformer.mask_generate import CausalEODMaskGenerate
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.device_matrix import layout
from mindformers.parallel_core.training_graph.transformer.identity_op import IdentityOp


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

    Input:
        index_scores: Scores predicted by indexer [batch, seqlen_q, seqlen_k].
        topk_indices: Top-k indices [batch, seqlen_q, index_topk].
        query: Query tensor [seqlen_q, batch, heads, dim].
        key: Key tensor [seqlen_k, batch, heads, dim].
        softmax_scale: Scale coefficient after q @ k^T.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: bool, whether to use sparse indexer loss. If True, only the topk
            indices will be used to compute the loss.
        pg_collection: Process group collection, must have TP process group.

    Returns:
        index_loss: KL divergence loss (scalar).
    """
    def __init__(self, config: TransformerConfig, softmax_scale=None):
        super().__init__()
        self.sparse_loss = config.dsa_indexer_use_sparse_loss
        self.loss_coeff = config.dsa_indexer_loss_coeff
        self.sparse_count = config.dsa_indexer_topk
        self.num_attention_heads = config.num_attention_heads
        self.is_tnd = config.input_layout == "TND"
        self.tp = config.tensor_model_parallel_size
        softmax_scale = softmax_scale or config.kv_channels ** -0.5

        self.softmax_scale = Tensor([softmax_scale], mstype.float32)
        self.inf = Tensor([-1e5], mstype.float32)
        self.eps = Tensor([1.e-8], mstype.float32)
        self.ignore = Tensor([-1], mstype.int32)

        self.eod_mask_generator = CausalEODMaskGenerate(config)

        self.cast = aclnn_ops.Cast()
        self.reshape = aclnn_ops.Reshape()
        self.transpose_q = aclnn_ops.Transpose()
        self.transpose_k = aclnn_ops.Transpose()
        self.bmm = aclnn_ops.BatchMatMul()
        self.mul_scalar1 = aclnn_ops.Mul()
        self.slice = aclnn_ops.SliceExt().add_prim_attr("self_define_shard", True)
        self.equal = aclnn_ops.Equal()
        self.tile = aclnn_ops.Tile()
        self.gather = aclnn_ops.GatherD().add_prim_attr("self_define_shard", True)
        self.mul_scalar2 = aclnn_ops.Mul()
        self.add = aclnn_ops.AddExt()
        self.softmax1 = aclnn_ops.Softmax(-1)
        self.sum = aclnn_ops.SumExt()
        if self.tp > 1:
            self.presum = P.Morph(
                self._presum_head,
                self.presum_infer_shape,
                lambda *args: args[0]
            ).add_prim_attr("self_define_shard", True)
        else:
            self.presum = IdentityOp()
        self.sum1 = aclnn_ops.SumExt().add_prim_attr("self_define_shard", True)
        self.sum2 = aclnn_ops.SumExt().add_prim_attr("self_define_shard", True)
        self.softmax2 = aclnn_ops.Softmax(-1)
        self.div = aclnn_ops.Div()
        self.add_scalar = aclnn_ops.AddExt()
        self.sub = aclnn_ops.SubExt()
        self.mul = aclnn_ops.Mul()
        self.log = aclnn_ops.Log()
        self.sum3 = aclnn_ops.SumExt().add_prim_attr("self_define_shard", True)
        self.mean = aclnn_ops.MeanExt()
        if _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
            self.shard()

    def presum_infer_shape(self, *args):
        b, _, s1, s2 = args[0]
        return [b, self.tp, s1, s2]

    def _presum_head(self, attention_scores):
        return self.sum(attention_scores, 1, True)

    def compute_indexer_loss(self, index_scores, topk_indices, query, key, mask):
        """Compute indexer loss of BSND layout"""
        # q: [b, n, sq, d], k: [b, n, d, sk]
        # attention_score: q @ k -> [b, n, sq, sk]
        attention_scores = self.mul_scalar1(self.bmm(query, key), self.softmax_scale)

        if self.sparse_loss:
            # when use sparse loss, the input `index_scores` is already in topk format
            # topk_indices: [b, 1, sq, sk] -> [b, n, sq, sk]
            # attention_score: [b, n, sq, sk] -> [b, n, sq, topk] -> [b, sq, topk]
            if self.is_tnd:
                mask = self.equal(topk_indices, self.ignore)
            else:
                mask = self.slice(mask, 3, 0, self.sparse_count, 1)
            topk_indices = self.tile(topk_indices, (1, self.num_attention_heads, 1, 1))
            attention_scores = self.gather(attention_scores, -1, topk_indices)

        # add mask to attention score and do softmax and L1-normalize
        mask = self.cast(mask, mstype.float32)
        mask = self.mul_scalar2(mask, self.inf)
        attention_scores = self.add(attention_scores, mask)
        attention_scores = self.softmax1(attention_scores)
        attention_scores = self.presum(attention_scores)
        attention_scores = self.sum1(attention_scores, dim=1)
        attention_scores = self.div(attention_scores, self.sum2(attention_scores, -1, keepdim=True) + self.eps)

        # index_scores is already masked. Just do softmax.
        index_scores = self.cast(index_scores, mstype.float32)
        index_scores = self.softmax2(index_scores)

        # compute KL divergence loss,  [b, sq, sk] -> [b, sq] -> [1]
        kl_per_element = self.mul(attention_scores,
                                  self.sub(self.log(self.add_scalar(attention_scores, self.eps)),
                                           self.log(self.add_scalar(index_scores, self.eps))))
        # Each token has same weight in the loss.
        kl_div = self.mean(self.sum3(kl_per_element, -1))

        # Scale by coefficient.
        indexer_loss = kl_div * self.loss_coeff
        return indexer_loss

    def construct(self, index_scores, topk_indices, query, key, mask, actual_seq_len=None):
        """compute indexer loss"""
        # convert tnd&sbnd layout to bnsd
        if self.is_tnd:
            bsz, seqlen, _, _ = query.shape
            query = self.cast(self.transpose_q(query, (0, 2, 1, 3)), mstype.float32)
            key = self.cast(self.transpose_k(key, (0, 2, 3, 1)), mstype.float32)
        else:
            seqlen, bsz, _, _ = query.shape
            query = self.cast(self.transpose_q(query, (1, 2, 0, 3)), mstype.float32)
            key = self.cast(self.transpose_k(key, (1, 2, 3, 0)), mstype.float32)
        index_scores = self.reshape(index_scores, (bsz, seqlen, -1))
        topk_indices = self.reshape(topk_indices, (bsz, 1, seqlen, -1))
        if self.is_tnd and not self.sparse_loss:
            bsz = query.shape[0]
            actual_seq_len = self.reshape(actual_seq_len, (bsz, -1))
            mask = self.eod_mask_generator(actual_seq_len)
        return self.compute_indexer_loss(index_scores, topk_indices, query, key, mask)

    def shard(self):
        """Set parallel strategy."""
        if self.is_tnd:
            q_input_shard = layout("dp", "cp", "tp", "None")
            k_input_shard = layout("dp", "cp", "None", "None")
        else:
            q_input_shard = layout("cp", "dp", "tp", "None")
            k_input_shard = layout("cp", "dp", "None", "None")
        q_shard = layout("dp", "tp", "cp", "None")
        k_shard = layout("dp", "None", "None", "None")
        scalar_shard = layout("None")
        attn_shard = q_shard
        mask_shard = layout("dp", "None", "cp", "None")
        topk_shard = layout("dp", "None", "cp", "None")
        pooled_attn_shard = layout("dp", "cp", "None")
        self.transpose_q.shard((q_input_shard,))
        self.transpose_k.shard((k_input_shard,))
        self.bmm.shard((q_shard, k_shard))
        self.mul_scalar1.shard((attn_shard, scalar_shard))
        self.mul_scalar2.shard((mask_shard, scalar_shard))
        self.tile.shard((topk_shard,), (attn_shard,))
        self.gather.shard((attn_shard, layout(), attn_shard), (attn_shard,))
        self.equal.shard((mask_shard, scalar_shard))
        self.slice.shard((mask_shard,), (mask_shard,))
        self.add.shard((attn_shard, mask_shard))
        self.add_scalar.shard((pooled_attn_shard, scalar_shard))
        self.softmax1.shard((attn_shard,))
        self.presum.shard((attn_shard,), (attn_shard,))
        self.sum1.shard((layout("dp", "None", "cp", "None"),), (pooled_attn_shard,))
        self.softmax2.shard((pooled_attn_shard,))
        self.sum2.shard((pooled_attn_shard,), (pooled_attn_shard,))
        self.div.shard((pooled_attn_shard, pooled_attn_shard))
        self.log.shard((pooled_attn_shard,))
        self.sub.shard((pooled_attn_shard, pooled_attn_shard))
        self.mul.shard((pooled_attn_shard, pooled_attn_shard))
        self.sum3.shard((pooled_attn_shard,), (layout("dp", "cp"),))
