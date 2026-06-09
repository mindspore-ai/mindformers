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
import numpy as np
from mindspore import mint, nn, ops
from mindspore.common.parameter import Parameter
from mindspore.ops.auto_generate import GroupedMatmul

from hyper_parallel import DTensor

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
        # SwiGLU clamp: get_activation returns ClampedSwiGlu when clamp_value is
        # set (SwiGLU + MoE only per config validation). ClampedSwiGlu shares
        # ops.swiglu's first-half/second-half chunk semantics, so the call site
        # self.activation_func(fc1_output, -1) is unchanged.
        clamp_value = self.config.activation_func_clamp_value
        self.activation_func = get_activation(self.activation_type, clamp_value=clamp_value)
        self.gated_linear_unit = self.config.gated_linear_unit
        if self.gated_linear_unit:
            self.moe_ffn_hidden_size *= 2
            self.mul = mint.mul
        self.moe_token_dispatcher_type = config.moe_token_dispatcher_type
        self.init_method = config.init_method
        self.compute_dtype = config.compute_dtype
        self.moe_permute_fusion = config.moe_permute_fusion

        # parameters
        self.weight1 = Parameter(
            mint.empty([self.num_local_experts, self.hidden_size, self.moe_ffn_hidden_size],
                       dtype=self.config.params_dtype),
            name='w1')
        self.weight2 = Parameter(
            mint.empty([self.num_local_experts, self.config.moe_ffn_hidden_size, self.hidden_size],
                       dtype=self.config.params_dtype),
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

    def permute(self, tokens, top_scores, selected_experts_indices, num_tokens_per_expert):
        """
        Reorders token indices to match the order of experts for MoE routing.

        Returns:
            tuple: (num_tokens_per_expert, sort_index, permuted_probs, routed_input)
                - sort_index: in fusion mode this is the ``unsort_map`` produced by
                  ``ops.moe_token_permute`` (used directly by ``ops.moe_token_unpermute``);
                  in non-fusion mode it is the argsort index used to scatter outputs back.
                - permuted_probs: in fusion mode it is in original ``(N, K)`` order unless
                  ``moe_apply_probs_on_input`` is enabled, in which case it is gathered into
                  expert-sorted order so it can be broadcast onto ``routed_input``. In
                  non-fusion mode it is always expert-sorted ``(N*K,)``.
        """
        _, _, dim = tokens.shape
        tokens = self.reshape(tokens, (-1, dim))

        num_tokens_per_expert = self.cast(num_tokens_per_expert, selected_experts_indices.dtype)
        num_tokens_per_expert = self.cumsum(num_tokens_per_expert, dim=0, dtype=ms.int64)

        if self.moe_permute_fusion:
            routing_map = self.reshape(selected_experts_indices, (-1, self.top_k))
            routed_input, unsort_map = ops.moe_token_permute(
                tokens, routing_map.astype(ms.int32))
            routed_input = self.reshape(routed_input, (-1, self.hidden_size))
            if self.config.moe_apply_probs_on_input:
                # need probs in expert-sorted order to broadcast onto routed_input
                sort_map = self.argsort(self.cast(unsort_map, ms.float32))
                permuted_probs = self.reshape(top_scores, (-1,))[sort_map]
            else:
                # leave probs unpermuted; ops.moe_token_unpermute will consume (N, K)
                permuted_probs = top_scores
            return num_tokens_per_expert, unsort_map, permuted_probs, routed_input

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        selected_experts_indices = self.cast(selected_experts_indices, ms.float32)
        token_indices_experts_sorted = self.argsort(self.reshape(selected_experts_indices, (-1,)), stable=True)

        top_scores_experts_sorted = self.reshape(top_scores, (-1,))[token_indices_experts_sorted]

        # shape (bs*slen*top_k, dim)
        routed_input = tokens[mint.floor_divide(token_indices_experts_sorted, self.top_k)]
        routed_input = self.reshape(routed_input, (-1, self.hidden_size))
        return num_tokens_per_expert, token_indices_experts_sorted, top_scores_experts_sorted, routed_input

    def unpermute(self, routed_output, token_indices_experts_sorted, probs, shape):
        """
        Restores the original token order from expert-sorted outputs for MoE routing.
        """
        (bs, slen, dim) = shape

        if self.moe_permute_fusion:
            # token_indices_experts_sorted is the unsort_map from ops.moe_token_permute.
            # ops.moe_token_unpermute with probs scatters back and reduces over top_k in one op.
            unsort_map = self.reshape(token_indices_experts_sorted, (-1,))
            probs = self.reshape(probs, (-1, self.top_k))
            probs = self.cast(probs, routed_output.dtype)
            out_experts = ops.moe_token_unpermute(routed_output, unsort_map, probs)
            return out_experts

        # Unsort routed outputs
        routed_output_unsorted = self.zeros(
            (bs * slen * self.top_k, dim),
            dtype=routed_output.dtype,
        )

        probs = self.cast(probs, routed_output.dtype)
        routed_output = self.mul(routed_output, self.unsqueeze(probs, -1))
        routed_output_unsorted[token_indices_experts_sorted] = routed_output
        routed_output_unsorted = self.reshape(routed_output_unsorted, (-1, self.top_k, dim))
        out_experts = routed_output_unsorted.sum(dim=1)
        return out_experts

    def construct(self, tokens, probs, topk_indices, num_tokens_per_expert):
        """Construct function of GroupedMLP."""
        # Func permute and unpermute is required when using a single card or ep is not enabled.
        # When ep parallelism is enabled, ExpertParallel inference will take over the dispatch operation.
        need_dispatch = not isinstance(self.weight1, DTensor) or "ep" not in self.weight1.device_mesh.mesh_dim_names

        if need_dispatch:
            tokens_per_expert, token_indices_experts_sorted, permuted_probs, permuted_local_hidden_states = (
                self.permute(tokens, probs, topk_indices, num_tokens_per_expert))
        else:
            tokens_per_expert, permuted_probs, permuted_local_hidden_states = num_tokens_per_expert, probs, tokens
            token_indices_experts_sorted = None

        if self.config.moe_apply_probs_on_input:
            permuted_local_hidden_states = self.mul(self.unsqueeze(permuted_probs, -1), permuted_local_hidden_states)
            permuted_probs = self.ones_like(permuted_probs)

        experts_output = self.experts_forward(permuted_local_hidden_states, tokens_per_expert)

        if need_dispatch:
            experts_output = self.unpermute(experts_output, token_indices_experts_sorted, permuted_probs, tokens.shape)

        return experts_output

    def experts_forward(self, permuted_local_hidden_states, tokens_per_expert):
        """Forward step of GroupedMLP."""
        original_dtype = permuted_local_hidden_states.dtype
        permuted_local_hidden_states = self.cast(permuted_local_hidden_states, self.compute_dtype)
        w1 = self.cast(self.weight1, self.compute_dtype)
        w2 = self.cast(self.weight2, self.compute_dtype)
        w1 = w1.to_local() if isinstance(w1, DTensor) else w1
        w2 = w2.to_local() if isinstance(w2, DTensor) else w2

        fc1_output = GroupedMatmul(split_item=3, group_type=0)(
            [permuted_local_hidden_states], [w1], None, None, None, None,
            None, tokens_per_expert)[0]

        if self.gated_linear_unit:
            if self.activation_type == 'fusedswiglu':
                intermediate_parallel = self.activation_func(fc1_output, -1).reshape((-1, w2.shape[1]))
            else:
                x0, x1 = self.chunk(fc1_output, 2, -1)
                act_out = self.activation_func(x0)
                intermediate_parallel = self.mul(act_out, x1)
        else:
            intermediate_parallel = self.activation_func(fc1_output)

        fc2_output = GroupedMatmul(split_item=3, group_type=0)(
            [intermediate_parallel], [w2], None, None, None, None,
            None, tokens_per_expert)[0]
        fc2_output = self.cast(fc2_output, original_dtype)
        return fc2_output

    def reset_parameter(self):
        """Reset expert weights for delayed initialization."""
        self.weight1.normal_(mean=0.0, std=0.01)
        self.weight2.normal_(mean=0.0, std=0.01)
