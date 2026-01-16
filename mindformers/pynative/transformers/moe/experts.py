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
"""GroupedMLP for MoE"""
__all__ = ["GroupedMLP"]

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.common.parameter import Parameter
from mindspore.ops.auto_generate import GroupedMatmul
from mindformers.tools.logger import logger
from mindformers.pynative.layers.activation import get_activation
from mindformers.parallel_core.transformer_config import TransformerConfig


class GroupedMLP(nn.Cell):
    """
    An efficient implementation of the Experts layer using GroupedGEMM.

    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.

    Args:
        config (TransformerConfig): Configuration object for the GroupedMLP module.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.num_local_experts = self.config.num_moe_experts
        self.top_k = self.config.moe_router_topk

        if self.config.moe_apply_probs_on_input:
            if self.config.moe_router_topk == 1:
                raise ValueError("`moe_apply_probs_on_input` only works with `moe_router_topk`=1.")

        if self.config.add_bias_linear:
            logger.error("bias not supported in GroupedMLP yet, please set: "
                "model_config: \n"
                "    add_bias_linear: False \n"
                "in yaml configuration.")
            raise NotImplementedError(
                "bias not supported in GroupedMLP yet, please set: "
                "model_config: \n"
                "    add_bias_linear: False \n"
                "in yaml configuration.")

        self.moe_ffn_hidden_size = self.config.moe_ffn_hidden_size
        self.hidden_size = self.config.hidden_size
        self.activation_type = self.config.hidden_act
        self.activation_func = get_activation(self.activation_type)
        self.gated_linear_unit = self.config.gated_linear_unit
        if self.gated_linear_unit:
            self.moe_ffn_hidden_size *= 2
            self.mul = mint.mul
        self.moe_token_dispatcher_type = config.moe_token_dispatcher_type
        self.init_method = config.init_method

        # parameters
        self.weight1 = Parameter(
            self.init_method([self.num_local_experts * self.hidden_size, self.moe_ffn_hidden_size]),
            name='w1')
        self.weight2 = Parameter(
            self.init_method([self.num_local_experts * self.config.moe_ffn_hidden_size, self.hidden_size]),
            name='w2')

        self.cast = ops.cast
        self.reshape = mint.reshape
        self.chunk = mint.chunk
        self.mul = mint.mul
        self.unsqueeze = mint.unsqueeze
        self.ones_like = mint.ones_like
        self.histc = mint.histc
        self.argsort = mint.argsort
        self.zeros = mint.zeros
        self.sum = mint.sum
        self.cumsum = mint.cumsum
        self.bmm = mint.bmm

    def permute(self, tokens, top_scores, selected_experts_indices):
        """
        Reorders token indices to match the order of experts for MoE routing.
        """
        _, _, dim = tokens.shape
        tokens = self.reshape(tokens, (-1, dim))
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = self.histc(
            selected_experts_indices,
            bins=self.num_local_experts,
            min=0,
            max=self.num_local_experts,
        )
        num_tokens_per_expert = self.cast(num_tokens_per_expert, selected_experts_indices.dtype)

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = self.argsort(self.reshape(selected_experts_indices, (-1,)), stable=True)

        top_scores_experts_sorted = self.reshape(top_scores, (-1,))[token_indices_experts_sorted]

        # shape (bs*slen*top_k, dim)
        routed_input = tokens[token_indices_experts_sorted // self.top_k]

        return num_tokens_per_expert, token_indices_experts_sorted, top_scores_experts_sorted, routed_input

    def unpermute(self, routed_output, token_indices_experts_sorted, shape):
        """
        Restores the original token order from expert-sorted outputs for MoE routing.
        """
        (bs, slen, dim) = shape
        # Unsort routed outputs
        routed_output_unsorted = self.zeros(
            (bs * slen * self.top_k, dim),
            dtype=routed_output.dtype,
        )
        routed_output_unsorted[token_indices_experts_sorted] = routed_output
        routed_output_unsorted = self.reshape(routed_output_unsorted, (-1, self.top_k, dim))
        out_experts = routed_output_unsorted.sum(dim=1)
        return out_experts

    def construct(self, tokens, probs, topk_indices):
        """Construct function of GroupedMLP."""
        tokens_per_expert, token_indices_experts_sorted, permuted_probs, permuted_local_hidden_states = self.permute(
            tokens, probs, topk_indices)

        permuted_local_hidden_states = self.reshape(permuted_local_hidden_states, (-1, self.hidden_size))
        if self.config.moe_apply_probs_on_input:
            permuted_local_hidden_states = self.mul(self.unsqueeze(permuted_probs, -1), permuted_local_hidden_states)
            permuted_probs = self.ones_like(permuted_probs)

        experts_output = self.experts_forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)

        output = self.unpermute(experts_output, token_indices_experts_sorted, tokens.shape)
        return output

    def experts_forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        """Forward step of GroupedMLP."""
        original_dtype = permuted_local_hidden_states.dtype
        w1 = self.cast(self.weight1, original_dtype)
        w2 = self.cast(self.weight2, original_dtype)
        w1 = self.reshape(w1, (-1, self.hidden_size, self.moe_ffn_hidden_size))
        w2 = self.reshape(w2, (-1, self.config.moe_ffn_hidden_size, self.hidden_size))
        tokens_per_expert = self.cumsum(tokens_per_expert, dim=0, dtype=ms.int64)
        fc1_output = GroupedMatmul(split_item=3, group_type=0)(
            [permuted_local_hidden_states], [w1], None, None, None, None, None, tokens_per_expert)[0]

        if self.gated_linear_unit:
            if self.activation_type == 'fusedswiglu':
                intermediate_parallel = self.activation_func(fc1_output, -1).reshape((-1, w2.shape[1]))
            else:
                x0, x1 = self.chunk(fc1_output, 2, -1)
                act_out = self.activation_func(x0)
                intermediate_parallel = self.mul(act_out, x1)
        else:
            intermediate_parallel = self.activation_func(fc1_output)

        permuted_probs = self.cast(permuted_probs, intermediate_parallel.dtype)
        intermediate_parallel = self.mul(intermediate_parallel, self.unsqueeze(permuted_probs, -1))
        fc2_output = GroupedMatmul(split_item=3, group_type=0)(
            [intermediate_parallel], [w2], None, None, None, None, None, tokens_per_expert)[0]
        return fc2_output
