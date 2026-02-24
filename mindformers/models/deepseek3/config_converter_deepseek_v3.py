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
"""DeepseekV3 config converter."""
from typing import Dict, Any

from mindformers.parallel_core.config_converter import ConfigConverter, ConversionContext
from mindformers.parallel_core.transformer_config_utils import is_float_32, get_drop_policy


class DeepseekV3ConfigConverter(ConfigConverter):
    """
    DeepseekV3ConfigConverter
    """
    CONFIG_MAPPING = {
        # Mixed-Precision
        "softmax_compute_dtype": ("attention_softmax_in_fp32", is_float_32),

        # Model Architecture
        "attention_bias": "add_qkv_bias",
        "num_hidden_layers": "num_layers",
        "rms_norm_eps": "layernorm_epsilon",
        "intermediate_size": "ffn_hidden_size",
        "num_key_value_heads": "num_query_groups",
        "num_nextn_predict_layers": "mtp_num_layers",
        "mtp_loss_factor": "mtp_loss_scaling_factor",
        # Flash Attention
        "initializer_range": "init_method_std",

        # MoE
        "n_group": "moe_router_num_groups",
        "n_routed_experts": "num_moe_experts",
        "topk_group": "moe_router_group_topk",
        "n_shared_experts": "shared_expert_num",
        "num_experts_per_tok": "moe_router_topk",
        "moe_intermediate_size": "moe_ffn_hidden_size",
        "routed_scaling_factor": "moe_router_topk_scaling_factor",
        "moe_token_drop_policy": ("moe_token_drop_policy", get_drop_policy),
        # MLA
        "rope_theta": "rotary_base",
        "qk_nope_head_dim": "qk_head_dim",
        "qk_rope_head_dim": "qk_pos_emb_head_dim",
    }

    @classmethod
    def _pre_process(cls, model_config: Dict[str, Any], ctx: ConversionContext) -> None:
        super()._pre_process(model_config, ctx)
