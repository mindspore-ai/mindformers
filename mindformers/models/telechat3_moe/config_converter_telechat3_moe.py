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
"""Telechat3Moe config converter."""
from typing import Dict, Any

from mindformers.models.telechat3.config_converter_telechat3 import Telechat3ConfigConverter
from mindformers.parallel_core.config_converter import ConversionContext
from mindformers.parallel_core.transformer_config_utils import get_drop_policy


class Telechat3MoeConfigConverter(Telechat3ConfigConverter):
    """
    Telechat3MoeConfigConverter
    """
    CONFIG_MAPPING = {
        # Model Architecture
        "num_nextn_predict_layers": "mtp_num_layers",
        "mtp_loss_factor": "mtp_loss_scaling_factor",
        # MoE
        "n_group": "moe_router_num_groups",
        "n_routed_experts": "num_moe_experts",
        "topk_group": "moe_router_group_topk",
        "router_dense_type": "moe_router_dtype",
        "n_shared_experts": "shared_expert_num",
        "num_experts_per_tok": "moe_router_topk",
        "moe_intermediate_size": "moe_ffn_hidden_size",
        "routed_scaling_factor": "moe_router_topk_scaling_factor",
        "moe_token_drop_policy": ("moe_token_drop_policy", get_drop_policy),
    }

    @classmethod
    def _pre_process(cls, model_config: Dict[str, Any], ctx: ConversionContext) -> None:
        super()._pre_process(model_config, ctx)
