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
"""DeepSeek Sparse Attention for pynative mode."""
# MindSpore ``Cell.construct`` intentionally exposes operator-specific signatures.
# pylint: disable=arguments-differ
from dataclasses import dataclass
from typing import Union

from mindspore import nn, ops, mint
from mindspore import dtype as mstype
from mindspore.ops import auto_generate as aclnn_ops
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from hyper_parallel import DTensor
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.pynative.layers.identity_op import IdentityOp
from mindformers.pynative.transformers.experimental_attention_variant.dsa_indexer_loss import DSAIndexerLoss


@dataclass
class DSAttentionSubmodules:
    """
    Configuration class for specifying the submodules of DSAttention.

        indexer: DSA Indexer module for computing sparse attention indices.
    """

    indexer: Union[ModuleSpec, type] = None


class DSASparseFlashAttention(nn.Cell):
    """Hookable boundary for the DSA sparse flash attention op."""

    def __init__(self, input_layout: str, softmax_scale: float, attention_mode: int):
        super().__init__()
        self.input_layout = input_layout
        self.softmax_scale = softmax_scale
        self.attention_mode = attention_mode

    def construct(
            self, query, key, value, topk_indices, query_rope=None, key_rope=None,
            actual_seq_qlen=None, actual_seq_kvlen=None
    ):
        return ops.sparse_flash_attention(
            query, key, value, topk_indices, self.softmax_scale,
            query_rope=query_rope,
            key_rope=key_rope,
            actual_seq_lengths_query=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_kvlen,
            layout_query=self.input_layout,
            layout_kv=self.input_layout,
            attention_mode=self.attention_mode,
            return_softmax_lse=True
        )

class DSADenseFlashAttention(nn.Cell):
    """Hookable boundary for the DSA dense flash attention op."""

    def __init__(self, input_layout: str, head_num: int, softmax_scale: float, sparse_mode: int):
        super().__init__()
        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            scale_value=softmax_scale,
            pre_tokens=2147483647,
            next_tokens=0,
            inner_precise=0,
            input_layout=input_layout,
            sparse_mode=sparse_mode,
        )

    def construct(
            self, query, key, value, attention_mask=None,
            actual_seq_qlen=None, actual_seq_kvlen=None
    ):
        """Run dense FlashAttention and return its output and LSE statistics."""
        softmax_max, softmax_sum, _, attention_output = self.flash_attention(
            query, key, value,
            None, None, None,
            attn_mask=attention_mask,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
        )
        return attention_output, softmax_max[..., :1], softmax_sum[..., :1]


class DSASoftmaxConverter(nn.Cell):
    """Convert dense FlashAttention TND LSE stats to dense indexer-loss layout."""

    def __init__(self, input_layout: str):
        super().__init__()
        self.is_tnd = input_layout == "TND"
        self.cp_rank = 0
        self.cast = aclnn_ops.Cast()
        self.reshape = aclnn_ops.Reshape()
        self.clamp = aclnn_ops.ClampScalar()
        self.roll = aclnn_ops.Roll(1)
        self.repeat_scalar = aclnn_ops.RepeatInterleaveInt()
        self.arange = aclnn_ops.Arange()
        self.mod = aclnn_ops.FmodScalar()
        self.repeat_interleave = aclnn_ops.RepeatInterleaveTensor()
        self.equal = aclnn_ops.Equal()
        self.select = aclnn_ops.MaskedSelect()
        self.stack = aclnn_ops.StackExt()
        self.transpose = aclnn_ops.Transpose()

    def set_cp_rank(self, cp_rank: int):
        """Set current rank's index inside the context-parallel mesh."""
        self.cp_rank = int(cp_rank)

    def construct(self, softmax_max, softmax_sum, actual_seq_len=None):
        """Return stats as ``[N, T, 1]`` for dense TND indexer loss."""
        if not self.is_tnd:
            return softmax_max, softmax_sum
        if actual_seq_len is None:
            return self.transpose(softmax_max, (1, 0, 2)), self.transpose(softmax_sum, (1, 0, 2))

        t, n, _ = softmax_max.shape
        softmax_max = self.reshape(softmax_max, (-1,))
        softmax_sum = self.reshape(softmax_sum, (-1,))
        offset_q = t * self.cp_rank
        actual_seq_qlen = self.cast(self.clamp(actual_seq_len - offset_q, 0, t), mstype.int32)
        prev_seq_qlen = self.roll(actual_seq_qlen)
        prev_seq_qlen[0] = 0
        interleave_seq_qlen = actual_seq_qlen - prev_seq_qlen
        interleave_seq_qlen = self.repeat_scalar(interleave_seq_qlen, n)
        base = self.mod(self.arange(0, interleave_seq_qlen.shape[0], 1), n)
        interleave_seq_qlen = self.repeat_interleave(base, interleave_seq_qlen)

        softmax_maxs = []
        softmax_sums = []
        for i in range(n):
            head_mask = self.equal(interleave_seq_qlen, i)
            softmax_maxs.append(self.select(softmax_max, head_mask))
            softmax_sums.append(self.select(softmax_sum, head_mask))
        softmax_max = self.reshape(self.stack(softmax_maxs), (n, t, 1))
        softmax_sum = self.reshape(self.stack(softmax_sums), (n, t, 1))
        return softmax_max, softmax_sum


def _to_local_tensor(value):
    """Return a local tensor when HP hooks keep a boundary output as DTensor."""
    return value.to_local() if isinstance(value, DTensor) else value


class DSAttention(nn.Cell):
    """
    Sparse attention mechanism using DSA Indexer for pynative mode.

    This module implements sparse attention with top-k token selection
    for reduced computational complexity. Single card only, no parallel sharding.

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
        self.head_num = config.num_attention_heads
        self.layer_number = layer_number
        self.attention_mode = 2
        self.sparse_loss = config.dsa_indexer_use_sparse_loss

        self.indexer = build_module(
            submodules.indexer, config=config
        )
        self.softmax_scale = softmax_scale or config.kv_channels ** -0.5
        self.indexer_loss = DSAIndexerLoss(config, self.softmax_scale)
        self.sparse_key_handoff = IdentityOp()
        self.sparse_value_handoff = IdentityOp()
        self.sparse_key_rope_handoff = IdentityOp()

        if self.sparse_loss:
            self.sparse_flash_attention = DSASparseFlashAttention(
                input_layout=self.input_layout,
                softmax_scale=self.softmax_scale,
                attention_mode=self.attention_mode,
            )
        else:
            # Dense stage: use standard FlashAttention
            self.dense_flash_attention = DSADenseFlashAttention(
                head_num=self.head_num,
                softmax_scale=self.softmax_scale,
                input_layout=self.input_layout,
                sparse_mode=config.sparse_mode,
            )
            self.softmax_converter = DSASoftmaxConverter(self.input_layout)

    def construct(
            self, query, key, value, topk_indices, attention_mask=None,
            actual_seq_qlen=None, actual_seq_kvlen=None
    ):
        """
        Forward pass for Sparse Attention.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            topk_indices: Top-k indices from indexer.
            attention_mask: Attention mask (dense stage only).
            actual_seq_qlen: Actual query sequence lengths (TND mode).
            actual_seq_kvlen: Actual key sequence lengths (TND mode).

        Returns:
            attention_output: Attention output tensor.
            softmax_max: Softmax max statistics.
            softmax_sum: Softmax sum statistics.
        """
        if self.sparse_loss:
            # Sparse stage: split Q/K into nope and rope components
            q_nope, q_rope = mint.split(query, [self.nope_dim, self.pe_dim], dim=-1)
            k_nope, k_rope = mint.split(key, [self.nope_dim, self.pe_dim], dim=-1)
            k_nope = self.sparse_key_handoff(k_nope)
            value = self.sparse_value_handoff(value)
            k_rope = self.sparse_key_rope_handoff(k_rope)
            attention_output, softmax_max, softmax_sum = self.sparse_flash_attention(
                q_nope, k_nope, value, topk_indices, q_rope, k_rope,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
            )
            attention_output = _to_local_tensor(attention_output)
            softmax_max = _to_local_tensor(softmax_max)
            softmax_sum = _to_local_tensor(softmax_sum)
        else:
            attention_output, softmax_max, softmax_sum = self.dense_flash_attention(
                query, key, value,
                attention_mask=attention_mask,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
            )
            if self.is_tnd:
                softmax_max, softmax_sum = self.softmax_converter(softmax_max, softmax_sum, actual_seq_qlen)
        return attention_output, softmax_max, softmax_sum
