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
"""Glm4Moe config converter."""
from typing import Dict, Any

from mindformers.models.glm4.config_converter_glm4 import Glm4ConfigConverter
from mindformers.parallel_core.config_converter import ConversionContext


class Glm4MoeConfigConverter(Glm4ConfigConverter):
    """
    Glm4MoeConfigConverter
    """
    CONFIG_MAPPING = {
        # MoE
        "num_experts_per_tok": "moe_router_topk",
        "moe_intermediate_size": "moe_ffn_hidden_size",
        "routed_scaling_factor": "moe_router_topk_scaling_factor",
        "topk_group": "moe_router_group_topk",
        "n_group": "moe_router_num_groups",
        "n_routed_experts": "num_moe_experts",
        "n_shared_experts": "shared_expert_num",

        "use_qk_norm": "qk_layernorm",
    }

    @classmethod
    def _pre_process(cls, model_config: Dict[str, Any], ctx: ConversionContext) -> None:
        super()._pre_process(model_config, ctx)
