# Copyright 2026 bzantium and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on the DeepSeekV4 implementations from the DeepSeek AI team. (https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
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
"""DeepSeekV4 HuggingFace Model Configs."""

__all__ = ['DeepseekV4Config']

from mindformers.models.deepseek4.config_converter_deepseek_v4 import DeepseekV4ConfigConverter
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.model_config_utils import (
    register_mf_model_parameter,
    ignore_and_delete_parameter,
    NotSupportedInfo
)

DEEPSEEK_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


@MindFormerRegister.register(MindFormerModuleType.CONFIG, legacy=False, search_names=['deepseek_v4', 'deepseek_mtp_v4'])
class DeepseekV4Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV4Model`].
    It is used to instantiate an DeepSeek-V4 model according to the specified arguments,
    defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the DeepSeek-V4.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 129280):
            Vocabulary size of the Deep model.
        hidden_size (`int`, *optional*, defaults to 7168):
            Dimension of the hidden representations.
        moe_intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 61):
            Number of hidden layers in the Transformer decoder.
        num_nextn_predict_layers (`int`, *optional*, defaults to 1):
            Number of nextn predict layers.
        num_attention_heads (`int`, *optional*, defaults to 128):
            Number of attention heads for each attention layer.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        n_routed_experts (`int`, *optional*, defaults to 384):
            Number of routed experts.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor for routed experts.
        topk_method (`str`, *optional*, defaults to `noaux_tc`):
            Topk method used in routed gate.
        num_experts_per_tok (`int`, *optional*, defaults to 6):
            Number of selected experts per token.
        norm_topk_prob (`bool`, *optional*, defaults to True):
            Whether to normalize the weights of the routed experts.
        scoring_func (`str`, *optional*, defaults to 'sqrtsoftplus'):
            Method of computing expert weights.
        num_key_value_heads (`int`, *optional*, defaults to 1):
            Number of key_value heads for Grouped Query Attention.
        head_dim (`int`, *optional*, defaults to 512):
            Dimension of each attention head.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Rank of the Q LoRA projections.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of the RoPE head for QK projections.
        o_lora_rank (`int`, *optional*, defaults to 1024):
            Rank of the O LoRA projections.
        o_groups (`int`, *optional*, defaults to 16):
            Number of groups for O projections.
        index_n_heads (`int`, *optional*, defaults to 64):
            Number of index heads for hash attention.
        index_head_dim (`int`, *optional*, defaults to 128):
            Dimension of index heads.
        index_topk (`int`, *optional*, defaults to 1024):
            Number of topk indices for hash attention.
        num_hash_layers (`int`, *optional*, defaults to 3):
            Number of hash layers.
        compress_ratios (`list`, *optional*):
            Compression ratios for each layer.
        compress_rope_theta (`int`, *optional*, defaults to 160000):
            Theta for compressed RoPE.
        sliding_window (`int`, *optional*, defaults to 128):
            Sliding window size for attention.
        swiglu_limit (`float`, *optional*, defaults to 10.0):
            Limit for SwiGLU activation.
        expert_dtype (`str`, *optional*, defaults to "fp4"):
            Data type for experts.
        hc_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for hyper-connection.
        hc_mult (`int`, *optional*, defaults to 4):
            Multiplier for hyper-connection.
        hc_sinkhorn_iters (`int`, *optional*, defaults to 20):
            Number of Sinkhorn iterations for hyper-connection.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function.
        max_position_embeddings (`int`, *optional*, defaults to 1048576):
            The maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation for initializing weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use cache.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for attention probabilities.
    """

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]

    @register_mf_model_parameter(
        seq_length=4096,
        compute_dtype='bf16',
        layernorm_compute_dtype="fp32",
        rotary_dtype="fp32",
        hidden_dropout=0.0,
        use_flash_attention=True,
        moe_router_score_function="sqrtsoftplus",
        moe_router_enable_expert_bias=True,
        moe_router_fusion=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        gated_linear_unit=True,
        multi_latent_attention=True,
        is_dynamic=False,
    )
    @ignore_and_delete_parameter(extra_ignore_param=[
        ('scoring_func', NotSupportedInfo.useless),
        ('expert_dtype', NotSupportedInfo.useless),
    ])
    def __init__(
            self,
            vocab_size=129280,
            hidden_size=7168,
            moe_intermediate_size=3072,
            num_hidden_layers=61,
            num_nextn_predict_layers=1,
            num_attention_heads=128,
            num_key_value_heads=1,
            n_shared_experts=1,
            n_routed_experts=384,
            routed_scaling_factor=2.5,
            topk_method='noaux_tc',
            num_experts_per_tok=6,
            norm_topk_prob=True,
            scoring_func='sqrtsoftplus',
            head_dim=512,
            q_lora_rank=1536,
            qk_rope_head_dim=64,
            o_lora_rank=1024,
            o_groups=16,
            index_n_heads=64,
            index_head_dim=128,
            index_topk=1024,
            num_hash_layers=3,
            compress_ratios=None,
            compress_rope_theta=160000,
            sliding_window=128,
            swiglu_limit=10.0,
            expert_dtype="fp4",
            hc_eps=1e-06,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            hidden_act="silu",
            max_position_embeddings=1048576,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=0,
            eos_token_id=1,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            experimental_attention_variant="dsv4_hybrid",
            enable_hyper_connections=True,
            enable_hc_head=False,
            n_group=0,
            **kwargs,
    ):
        """Deepseek V4 Config"""
        self.vocab_size = vocab_size
        self.actual_vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
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
        self.scoring_func = scoring_func
        self.n_group = n_group

        # V4 specific attention parameters
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim

        # Hash attention parameters
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.num_hash_layers = num_hash_layers

        # Compression parameters
        self.compress_ratios = compress_ratios or [128] * 61 + [0]
        self.compress_rope_theta = compress_rope_theta

        # Other V4 specific parameters
        self.sliding_window = sliding_window
        self.swiglu_limit = swiglu_limit
        self.expert_dtype = expert_dtype
        self.hc_eps = hc_eps
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.enable_hyper_connections = enable_hyper_connections
        self.enable_hc_head = enable_hc_head

        # Standard parameters
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.experimental_attention_variant = experimental_attention_variant

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def convert_to_transformer_config(self, is_mla_model: bool = True):
        """
        Convert DeepseekV4Config to TransformerConfig.
        Args:
            is_mla_model (bool, optional): V4 uses MLATransformerConfig.
        Returns:
            TransformerConfig: The converted transformer configuration.
        """
        return DeepseekV4ConfigConverter.convert(self.to_dict(), is_mla_model=is_mla_model)
