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
"""LoRA adapter for PyNative routed experts."""
__all__ = ["GroupedMLPWithLoRA"]

import mindspore as ms
from mindspore import mint
from mindspore.common.parameter import Parameter
from mindspore.ops.auto_generate import GroupedMatmul

from mindformers.pynative.layers.dropout import Dropout
from mindformers.pynative.layers.identity_op import IdentityOp
from mindformers.pynative.dtensor_compat import to_local
from mindformers.pynative.pet.utils import init_lora_parameters
from mindformers.pynative.transformers.moe.experts import GroupedMLP


class GroupedMLPWithLoRA(GroupedMLP):
    """GroupedMLP with independent split-LoRA adapters for every local expert."""

    def __init__(self, config, lora_rank=8, lora_alpha=16, lora_dropout=0.0,
                 lora_a_std=0.01, lora_b_std=0.0):
        super().__init__(config)
        if lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, but got {lora_rank}.")
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = float(lora_alpha) / float(lora_rank)
        self.lora_a_std = lora_a_std
        self.lora_b_std = lora_b_std
        self.lora_dropout = Dropout(lora_dropout) if lora_dropout > 0.0 else IdentityOp()

        self.weight1_lora_a = Parameter(
            mint.empty((self.num_local_experts, self.hidden_size, lora_rank), dtype=self.compute_dtype),
            name="weight1_lora_a")
        self.weight1_lora_b = Parameter(
            mint.empty((self.num_local_experts, lora_rank, self.moe_ffn_hidden_size), dtype=self.compute_dtype),
            name="weight1_lora_b")
        self.weight2_lora_a = Parameter(
            mint.empty((self.num_local_experts, self.config.moe_ffn_hidden_size, lora_rank),
                       dtype=self.compute_dtype),
            name="weight2_lora_a")
        self.weight2_lora_b = Parameter(
            mint.empty((self.num_local_experts, lora_rank, self.hidden_size), dtype=self.compute_dtype),
            name="weight2_lora_b")

    @staticmethod
    def _gmm(x, weight, group_list):
        return GroupedMatmul(split_item=3, group_type=0)(
            [x], [weight], None, None, None, None, None, group_list)[0]

    def _lora_delta(self, x, lora_a, lora_b, tokens_per_expert):
        x = self.cast(self.lora_dropout(x), self.compute_dtype)
        hidden = self._gmm(x, to_local(lora_a), tokens_per_expert)
        return self._gmm(hidden, to_local(lora_b), tokens_per_expert) * self.scaling

    def experts_forward(self, permuted_local_hidden_states, tokens_per_expert):
        """Apply base and adapter grouped GEMMs with the identical cumulative group list."""
        w1 = to_local(self.weight1)
        w2 = to_local(self.weight2)
        fc1_output = self._gmm(permuted_local_hidden_states, w1, tokens_per_expert)
        fc1_output = fc1_output + self._lora_delta(
            permuted_local_hidden_states, self.weight1_lora_a,
            self.weight1_lora_b, tokens_per_expert)

        if self.gated_linear_unit:
            if self.activation_type == "fusedswiglu":
                intermediate = self.activation_func(fc1_output, -1).reshape((-1, w2.shape[1]))
            else:
                gate, up = self.chunk(fc1_output, 2, -1)
                intermediate = self.mul(self.activation_func(gate), up)
        else:
            intermediate = self.activation_func(fc1_output)

        fc2_output = self._gmm(intermediate, w2, tokens_per_expert)
        return fc2_output + self._lora_delta(
            intermediate, self.weight2_lora_a, self.weight2_lora_b, tokens_per_expert)

    def reset_parameter(self):
        """Reset base and adapter parameters during delayed initialization."""
        super().reset_parameter()
        init_lora_parameters(
            (self.weight1_lora_a, self.weight2_lora_a),
            (self.weight1_lora_b, self.weight2_lora_b),
            self.lora_a_std,
            self.lora_b_std,
        )

    @classmethod
    def from_base(cls, base, **kwargs):
        """Build on meta and preserve the original expert Parameters and names."""
        with ms.DeviceCtx("meta"):
            obj = cls(base.config, **kwargs)
        obj.weight1 = base.weight1
        obj.weight2 = base.weight2
        return obj
