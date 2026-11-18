# Copyright 2026 bzantium and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on the TeleChatV4 implementations from the TeleChat AI team. (https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B)
# ============================================================================
# """TeleChatV4 Config"""
#
# Modification points:
# 1. Change `PretrainedConfig` to MindSpore Transformers;
# 2. Delete useless code for logging;
# 3. Add the `__all__` information of the Config class;
# 4. Add `MindFormerRegister` decorator to adapt to training/inference process of MindSpore Transformers;
# 5. Add `register_mf_model_parameter` decorator to pass other required parameters except HuggingFace parameters;
# 6. Add `ignore_and_delete_parameter` decorator to shield unnecessary configuration information.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TeleChatV4 HuggingFace Model Configs."""

__all__ = ['TeleChatV4Config']

from mindformers.models.telechat4_moe.config_converter_telechat_v4 import TeleChatV4ConfigConverter
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.model_config_utils import (
    register_mf_model_parameter,
    ignore_and_delete_parameter,
    NotSupportedInfo
)

TELECHAT_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


@MindFormerRegister.register(
    MindFormerModuleType.CONFIG, legacy=False, search_names=['telechat_v4', 'telechat_mtp_v4', 'telechat4_moe']
)
class TeleChatV4Config(PretrainedConfig):
    r"""Configuration for the TeleChat4 29B MoE model (PyNative only).

    Defaults describe a 40-layer MLA decoder with a hidden size of 3584,
    32 attention heads, 64 routed experts (top-4), one shared expert and
    one MTP layer. The first two decoder layers use dense FFNs of width
    9216; MoE experts use width 1024. Attention uses Q/KV LoRA ranks
    768/512, QK non-RoPE/RoPE dimensions 128/64 and value dimension 128.

    The default attention is ordinary MLA. DSA and compressed attention
    remain opt-in through ``experimental_attention_variant``. SwiGLU is
    unclamped by default (``swiglu_limit=None``). Parameters named
    ``hc_*`` configure mHC. Defaults follow the Hugging Face
    ``XingChen-AGI/Xing4.0-29B-A4B`` configuration
    with DSA disabled; hash routing is also disabled.

    Args:
        intermediate_size (int): Dense FFN width. Default: 9216.
        qk_nope_head_dim (int): Non-RoPE QK head dimension. Default: 128.
        topk_group (int): Router group top-k. Default: 1 (select the sole group).
        **kwargs: MindFormers model parameters and ``PretrainedConfig`` options.

    Explicit arguments describe the public HF architecture. Optional backend
    settings use native MindFormers names through ``kwargs``.
    """

    model_type = "telechat_v4"
    keys_to_ignore_at_inference = ["past_key_values"]

    @register_mf_model_parameter(
        seq_length=262144,
        hidden_dropout=0.0,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_fusion=True,
        add_bias_linear=False,
        gated_linear_unit=True,
        params_dtype="bf16",
        embedding_params_dtype="bf16",
        qk_layernorm=True,
        mtp_loss_scaling_factor=0.1,
        mtp_enable_hyper_connections=False,
        hc_sinkhorn_clamp=(-30, 30),
        mhc_compute_dtype="bf16",
        enable_hc_head=False,
        mhc_megatron_compatibility=True,
        moe_grouped_gemm=True,
    )
    @ignore_and_delete_parameter(extra_ignore_param=[
        ('scoring_func', NotSupportedInfo.useless),
        ('expert_dtype', NotSupportedInfo.useless),
    ])
    def __init__(
            self,
            vocab_size=131072,
            hidden_size=3584,
            intermediate_size=9216,
            moe_intermediate_size=1024,
            num_hidden_layers=40,
            num_nextn_predict_layers=1,
            num_attention_heads=32,
            num_key_value_heads=32,
            n_shared_experts=1,
            n_routed_experts=64,
            routed_scaling_factor=2.0,
            topk_method='noaux_tc',
            num_experts_per_tok=4,
            norm_topk_prob=True,
            q_lora_rank=768,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            v_head_dim=128,
            qk_rope_head_dim=64,
            hc_eps=1e-06,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            hidden_act="silu",
            max_position_embeddings=262144,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            enable_hyper_connections=True,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=2,
            **kwargs,
    ):
        """TeleChat V4 Config"""
        # Accept the native MindFormers spellings without leaving two conflicting
        # sources for one converted field. YAMLs commonly use these names.
        intermediate_size = kwargs.pop("ffn_hidden_size", intermediate_size)
        hc_eps = kwargs.pop("mhc_layernorm_epsilon", hc_eps)
        hc_mult = kwargs.pop("num_residual_streams", hc_mult)
        hc_sinkhorn_iters = kwargs.pop("mhc_sinkhorn_iterations", hc_sinkhorn_iters)

        self.vocab_size = vocab_size
        self.actual_vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.first_k_dense_replace = first_k_dense_replace

        # MLA parameters
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim

        self.hc_eps = hc_eps
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.enable_hyper_connections = enable_hyper_connections

        # Standard parameters
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        # A fresh HF-style YaRN dictionary retains the 4K original context;
        # TransformerConfig converts this to its native base-context convention.
        self.rope_scaling = rope_scaling if rope_scaling is not None else {
            "type": "yarn", "factor": 64.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32, "beta_slow": 1,
            "mscale": 1.0, "mscale_all_dim": 1.0,
        }
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def convert_to_transformer_config(self, is_mla_model: bool = True):
        """
        Convert TeleChatV4Config to TransformerConfig.
        Args:
            is_mla_model (bool, optional): V4 uses MLATransformerConfig.
        Returns:
            TransformerConfig: The converted transformer configuration.
        """
        return TeleChatV4ConfigConverter.convert(self.to_dict(), is_mla_model=is_mla_model)
