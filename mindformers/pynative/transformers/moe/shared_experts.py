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
"""Transformer SharedExpertMLP"""
__all__ = [
    "SharedExpertMLP"
]

from copy import deepcopy

from mindspore import ops, Tensor, mint
from mindspore.nn.layer import Dense
from mindformers.pynative.transformers.mlp import MLP, MLPSubmodules
from mindformers.parallel_core.transformer_config import TransformerConfig


class SharedExpertMLP(MLP):
    """
    Implementation of a shared expert feedforward block that inherits from MLP.

    This module extends the standard MLP to support shared expert logic, typically used in MoE settings.

    Args:
        config (TransformerConfig): Configuration for the transformer model.
        submodules (MLPSubmodules): The submodules used to construct the MLP, such as activation and linear layers.

    Inputs:
        - **hidden_states** (Tensor) - Input tensor of shape :math:`(S, B, H)`, where
          :math:`S` is sequence length, :math:`B` is batch size, and :math:`H` is hidden size.

    Outputs:
        - **output** (Tensor) - Output tensor of shape :math:`(S, B, H)`.
        - **output_bias** (Tensor) - Bias tensor of shape :math:`(S, B, H)` (if applicable).

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, config: TransformerConfig, submodules: MLPSubmodules):
        config = deepcopy(config)
        config.ffn_hidden_size = config.moe_shared_expert_intermediate_size
        super().__init__(config, submodules)
        self.cast = ops.cast
        self.router_dense_type = config.moe_router_dtype
        self.use_shared_expert_gate = config.use_shared_expert_gating
        self.compute_dtype = config.compute_dtype
        if self.use_shared_expert_gate:
            self.shared_experts_gate = Dense(in_channels=config.hidden_size,
                                             out_channels=1,
                                             has_bias=False,
                                             dtype=self.router_dense_type)
            self.sigmoid = mint.nn.functional.sigmoid
            self.mul_shared_gate = mint.mul

    def construct(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """ Construct function of shared_expert_mlp block. """
        shared_experts_output = super().construct(hidden_states)
        if self.use_shared_expert_gate:
            gate = self.sigmoid(self.shared_experts_gate(self.cast(hidden_states, self.router_dense_type)))
            shared_experts_output = self.mul_shared_gate(shared_experts_output, self.cast(gate, self.compute_dtype))
        return shared_experts_output
