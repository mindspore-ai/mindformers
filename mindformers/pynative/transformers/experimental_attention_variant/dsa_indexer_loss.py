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
"""DSA Indexer Loss for pynative mode."""
# MindSpore ``Cell``/``_Function`` subclasses intentionally use operator-specific signatures.
# pylint: disable=arguments-differ,abstract-method
import mindspore as ms
from mindspore import Tensor, nn, ops, mint
from mindspore.common._grad_function import _Function

from hyper_parallel.custom_ops.experimental import (
    npu_dense_lightning_indexer_grad_kl_loss,
    npu_sparse_lightning_indexer_grad_kl_loss,
)
from mindformers.parallel_core.transformer_config import MLATransformerConfig


class _DSAIndexerGradFunction(_Function):
    """
    Custom autograd function for DSA Indexer Loss gradient routing.

    Routes pre-computed gradients from the fused loss operator to indexer inputs
    while blocking gradient flow to the main model.
    """

    @staticmethod
    def forward(
            ctx, q, k, q_index, k_index, weights,
            topk_indices, softmax_max, softmax_sum,
            softmax_max_index, softmax_sum_index,
            softmax_scale, q_rope, k_rope,
            actual_seq_qlen, actual_seq_klen,
            input_layout, sparse_loss
    ):
        """Forward pass: save pre-computed gradients and return loss."""
        if sparse_loss:
            d_query_index, d_key_index, d_weights, loss = npu_sparse_lightning_indexer_grad_kl_loss(
                q, k, q_index, k_index, weights, topk_indices,
                softmax_max, softmax_sum,
                softmax_scale,
                query_rope=q_rope,
                key_rope=k_rope,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_klen=actual_seq_klen,
                layout=input_layout,
            )
        else:
            d_query_index, d_key_index, d_weights, loss = npu_dense_lightning_indexer_grad_kl_loss(
                q, k, q_index, k_index, weights,
                softmax_max, softmax_sum,
                softmax_max_index, softmax_sum_index,
                softmax_scale,
                query_rope=q_rope,
                key_rope=k_rope,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_klen=actual_seq_klen,
                layout=input_layout,
            )
        ctx.d_query_index = d_query_index
        ctx.d_key_index = d_key_index
        ctx.d_weights = d_weights
        return loss[0]

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward pass: route pre-computed gradients to indexer inputs."""
        res_list = [None] * 12
        grad_scale = grad_output[0]
        d_query_index = ctx.d_query_index
        d_key_index = ctx.d_key_index
        d_weights = ctx.d_weights
        # ``grad_scale`` is a device scalar (normally 1 / token_count). Do not
        # convert it to a Python bool: that synchronizes the device and creates
        # another output allocation at the peak of sparse-DSA backward. Scale
        # the precomputed gradients in place to avoid three large temporaries.
        d_query_index.mul_(grad_scale)
        d_key_index.mul_(grad_scale)
        d_weights.mul_(grad_scale)
        return None, None, d_query_index, d_key_index, d_weights, *res_list


class DSAIndexerLossCompute(nn.Cell):
    """Hookable DSA indexer-loss kernel boundary.

    The parent ``DSAIndexerLoss`` keeps local-only operations such as
    ``stop_gradient`` and q/k splitting outside this boundary.  HP hooks are
    attached here, so q/k/indexer tensors receive TP/CP placements before the
    fused dense/sparse indexer-loss operator is invoked.

    This submodule MUST exist as a named attribute (``compute_indexer_loss``)
    on ``DSAIndexerLoss`` so that the TP/CP parallelize plan can attach hooks
    to it via ``prepare_module_input``.
    """

    def __init__(self, config: MLATransformerConfig, softmax_scale: float):
        super().__init__()
        self.input_layout = config.input_layout
        self.is_tnd = config.input_layout == "TND"
        self.sparse_loss = config.dsa_indexer_use_sparse_loss
        self.softmax_scale = float(softmax_scale)

    def construct(
            self, query, key, query_index, key_index, weights,
            topk_indices, softmax_max, softmax_sum,
            query_rope, key_rope,
            softmax_max_index=None, softmax_sum_index=None,
            actual_seq_qlen=None, actual_seq_klen=None
    ):
        """Compute pre-routed indexer gradients and loss."""
        loss = _DSAIndexerGradFunction.apply(query, key, query_index, key_index, weights,
            topk_indices, softmax_max, softmax_sum,
            softmax_max_index, softmax_sum_index,
            self.softmax_scale, query_rope, key_rope,
            actual_seq_qlen, actual_seq_klen, self.input_layout, self.sparse_loss)
        return loss


class DSAIndexerLoss(nn.Cell):
    """
    Compute KL divergence loss between indexer scores and true attention scores.

    This loss trains the indexer to predict which tokens are important by matching
    the distribution of true attention scores. Adapted for pynative mode.

    Args:
        config (MLATransformerConfig): The configuration for the transformer model.
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
        softmax_scale = softmax_scale or config.kv_channels ** -0.5
        self.softmax_scale = float(softmax_scale)
        self.split = mint.split
        self.compute_indexer_loss = DSAIndexerLossCompute(config, self.softmax_scale)

    def construct(
            self, query, key, query_index, key_index, weights,
            topk_indices, softmax_max, softmax_sum,
            softmax_max_index=None, softmax_sum_index=None,
            actual_seq_qlen=None, actual_seq_klen=None
    ):
        """
        Compute indexer loss.

        Returns:
            loss: Scalar KL divergence loss.
        """
        q_nope, q_rope = self.split(query, [self.nope_dim, self.pe_dim], dim=-1)
        k_nope, k_rope = self.split(key, [self.nope_dim, self.pe_dim], dim=-1)
        query_index = ops.cast(query_index, ms.bfloat16)
        key_index = ops.cast(key_index, ms.bfloat16)
        indexer_loss = self.compute_indexer_loss(q_nope, k_nope, query_index, key_index, weights,
            topk_indices, softmax_max, softmax_sum, q_rope, k_rope,
            softmax_max_index, softmax_sum_index,
            actual_seq_qlen, actual_seq_klen)
        if self.is_tnd:
            loss_scale = query.shape[0]
        else:
            loss_scale = query.shape[0] * query.shape[1]
        return indexer_loss / loss_scale
