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
"""MoE Layer implementation."""
from mindspore import nn, Tensor, mint, ops
import mindspore as ms
from mindspore.common.parameter import Parameter

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.layers.linear import Linear
from mindformers.pynative.transformers.mlp import MLPSubmodules
from .router import TopKRouter
from .experts import GroupedMLP
from .shared_experts import SharedExpertMLP


class MoELayer(nn.Cell):
    """
    MoE Layer that combines Router, Grouped Experts, and Shared Experts.

    Args:
        config (TransformerConfig): Configuration for the transformer model.
        layer_number (int): The layer index in the transformer stack. Used to
            track per-layer aux loss in MoETracker. Default: 0.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = 0):
        super().__init__()
        self.config = config
        self.num_experts = config.num_moe_experts
        self.top_k = config.moe_router_topk
        self.layer_number = layer_number

        # Router
        self.router = TopKRouter(config, layer_number=layer_number)

        # Experts
        self.experts = GroupedMLP(config)

        # Shared Experts
        self.shared_experts = None
        if config.shared_expert_num > 0:
            # Construct default submodules for SharedExpertMLP
            submodules = MLPSubmodules(
                linear_fc1=Linear,
                linear_fc2=Linear
            )
            self.shared_experts = SharedExpertMLP(config, submodules)

        # Determine when to apply scores: before or after experts
        self.score_before_experts = config.moe_apply_probs_on_input

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.enable_expert_bias = config.moe_router_enable_expert_bias
        if self.enable_expert_bias:
            self.expert_bias = Parameter(
                mint.zeros(self.num_experts, dtype=ms.float32),
                name="expert_bias",
                requires_grad=False,
            )
            self.load_balance_coeff = config.moe_router_bias_update_rate
        else:
            self.expert_bias = None
            self.load_balance_coeff = 0.0

        # Buffers for load balancing
        self.tokens_per_expert = Parameter(
            mint.zeros(self.num_experts, dtype=ms.float32),
            name="tokens_per_expert",
            requires_grad=False
        )

        # Mint operators
        self.reshape = mint.reshape
        self.zeros = mint.zeros
        self.sum = mint.sum
        self.add = mint.add
        self.mul = mint.mul
        self.bmm = mint.bmm
        self.unsqueeze = mint.unsqueeze
        self.cast = ops.cast
        self.ones_like = mint.ones_like


    def construct(self, hidden_states: Tensor):
        """
        Forward pass for MoELayer.
        Args:
            hidden_states (Tensor): Input tensor of shape (bs, seq_length, dim)
        """
        bs, seq_length, dim = hidden_states.shape
        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            hidden_states, self.expert_bias
        )

        self.tokens_per_expert.add_(num_tokens_per_expert)

        routed_output = self.experts(
            hidden_states, top_scores, selected_experts_indices, num_tokens_per_expert
        )

        shared_output = None
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)

        out_experts = self.reshape(routed_output, (bs, seq_length, dim))

        if shared_output is not None:
            final_out = self.add(shared_output, out_experts)
        else:
            final_out = out_experts

        return final_out
