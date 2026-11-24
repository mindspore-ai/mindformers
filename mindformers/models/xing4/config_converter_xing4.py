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
"""Xing4 config converter."""
from mindformers.parallel_core.config_converter import ConfigConverter, ConversionContext
from mindformers.parallel_core.transformer_config_utils import is_float_32, get_drop_policy


class Xing4ConfigConverter(ConfigConverter):
    """
    Xing4ConfigConverter
    """
    CONFIG_MAPPING = {
        # Mixed-Precision
        "softmax_compute_dtype": ("attention_softmax_in_fp32", is_float_32),

        # Model Architecture
        "attention_bias": "add_qkv_bias",
        "num_hidden_layers": "num_layers",
        "intermediate_size": "ffn_hidden_size",
        "rms_norm_eps": "layernorm_epsilon",
        "num_key_value_heads": "num_query_groups",
        "num_nextn_predict_layers": "mtp_num_layers",
        "mtp_loss_factor": "mtp_loss_scaling_factor",
        # Flash Attention
        "initializer_range": "init_method_std",

        # MoE
        "n_group": "moe_router_num_groups",
        "topk_group": "moe_router_group_topk",
        "n_routed_experts": "num_moe_experts",
        "n_shared_experts": "shared_expert_num",
        "num_experts_per_tok": "moe_router_topk",
        "moe_intermediate_size": "moe_ffn_hidden_size",
        "routed_scaling_factor": "moe_router_topk_scaling_factor",
        "moe_token_drop_policy": ("moe_token_drop_policy", get_drop_policy),

        # MLA
        "rope_theta": "rotary_base",
        "qk_nope_head_dim": "qk_head_dim",
        "qk_rope_head_dim": "qk_pos_emb_head_dim",

        # DSA
        "index_n_heads": "dsa_indexer_n_heads",
        "index_head_dim": "dsa_indexer_head_dim",
        "index_topk": "dsa_indexer_topk",

        # MHC
        "hc_eps": "mhc_layernorm_epsilon",
        "hc_mult": "num_residual_streams",
        "hc_sinkhorn_iters": "mhc_sinkhorn_iterations",
    }

    @classmethod
    def _pre_process(cls, model_config: dict, ctx: ConversionContext):
        """Combine HF clamp bounds into the framework's optional pair."""
        super()._pre_process(model_config, ctx)
        if not {"mhc_h_res_clamp_min", "mhc_h_res_clamp_max"}.intersection(model_config):
            return
        lower = model_config.pop("mhc_h_res_clamp_min", None)
        upper = model_config.pop("mhc_h_res_clamp_max", None)
        if (lower is None) != (upper is None):
            raise ValueError("Both mHC clamp bounds must be provided, or both must be None.")
        if lower is not None and lower > upper:
            raise ValueError("mHC clamp minimum must not exceed its maximum.")
        ctx.result["hc_sinkhorn_clamp"] = None if lower is None else (lower, upper)
