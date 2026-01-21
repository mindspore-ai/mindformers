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
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_moe_experts
        self.top_k = config.moe_router_topk

        # Router
        self.router = TopKRouter(config)

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

        self.load_balance_coeff = config.moe_aux_loss_coeff

        # Buffers for load balancing
        self.tokens_per_expert = Parameter(
            mint.zeros(self.num_experts, dtype=ms.float32),
            name="tokens_per_expert",
            requires_grad=False
        )

        self.expert_bias = None

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
        x_flat = self.reshape(hidden_states, (-1, dim))

        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(x_flat, self.expert_bias)

        self.tokens_per_expert.add_(num_tokens_per_expert)

        routed_output = self.experts(hidden_states, top_scores, selected_experts_indices)

        shared_output = None
        if self.shared_experts is not None:
            shared_output, _ = self.shared_experts(hidden_states)

        out_experts = self.reshape(routed_output, (bs, seq_length, dim))

        if shared_output is not None:
            final_out = self.add(shared_output, out_experts)
        else:
            final_out = out_experts

        return final_out
