# coding=utf-8
# Copyright 2025 bzantium and the HuggingFace Inc. team. All rights reserved.
#
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
"""TeleChat3 Hugging Face Model Configs."""

__all__ = ['TeleChat3Config']

from mindformers.models.telechat3.config_converter_telechat3 import Telechat3ConfigConverter
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.model_config_utils import (
    register_mf_model_parameter,
    ignore_and_delete_parameter,
    NotSupportedInfo
)


@MindFormerRegister.register(MindFormerModuleType.CONFIG, legacy=False, search_names=["telechat3"])
class TeleChat3Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`TeleChat3Model`].
    It is used to instantiate an TeleChat3 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the TeleChat3.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 131072):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`TeleChat3Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by mean-pooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        head_dim: Projection weights dimension in multi-head attention.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers using full attention. The first `max_window_layers`
            layers will use full attention, while any
            additional layer afterward will use SWA (Sliding Window Attention).
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_nextn_predict_layers (`int`, *optional*, defaults to 0):
            Number of nextn predict layers in the TeleChat3 Model.
    """
    model_type = "telechat3"
    keys_to_ignore_at_inference = ["past_key_values"]

    @register_mf_model_parameter(
        seq_length=4096,
        pad_token_id=151643,
        block_size=32,
        num_blocks=1024,
        normalization="RMSNorm",
        add_bias_linear=False,
        gated_linear_unit=True,
        use_contiguous_weight_layout_attention=False,
        coeff=0.007,
        is_dynamic=False,
    )
    @ignore_and_delete_parameter(extra_ignore_param=[
        ('max_window_layers', NotSupportedInfo.useless),
        ('sliding_window', NotSupportedInfo.useless),
        ('use_sliding_window', NotSupportedInfo.useless),
        ('layer_types', NotSupportedInfo.useless),
        ('logn', NotSupportedInfo.useless),
        ('training_seqlen', NotSupportedInfo.useless),
        ('embed_layernorm', NotSupportedInfo.useless),
        ('unk_token_id', NotSupportedInfo.useless),
        ('flash_attn', NotSupportedInfo.useless),
        ('base_seqlen', NotSupportedInfo.useless),
        ('mlp_bias', NotSupportedInfo.useless),
        ('pretraining_tp', NotSupportedInfo.useless),
        ('share_attention', NotSupportedInfo.useless),
        ('share_ffn', NotSupportedInfo.useless),
    ])
    def __init__(
            self,
            vocab_size=131072,
            hidden_size=4096,
            intermediate_size=22016,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            head_dim=128,
            hidden_act="silu",
            max_position_embeddings=32768,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            use_sliding_window=False,
            sliding_window=4096,
            max_window_layers=28,
            layer_types=None,
            attention_dropout=0.0,
            num_nextn_predict_layers=0,
            **kwargs):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.head_dim = head_dim
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        self.layer_types = layer_types
        if self.layer_types is not None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def convert_to_transformer_config(self, is_mla_model: bool = False):
        """
        Convert Telechat3Config to TransformerConfig.
        Args:
            is_mla_model (bool, optional): Whether converting to MLATransformerConfig. Defaults to False.
        Returns:
            TransformerConfig: The converted transformer configuration.
        """
        return Telechat3ConfigConverter.convert(self.to_dict(), is_mla_model=is_mla_model)
