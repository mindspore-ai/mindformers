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
"""Shared HF-style configuration and independent defaults for Xing4 models."""
from copy import deepcopy

from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.model_config_utils import (
    NotSupportedInfo, ignore_and_delete_parameter, register_mf_model_parameter,
)
from mindformers.modules.transformer.transformer import TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from .config_converter_xing4 import Xing4ConfigConverter

__all__ = ["Xing4Config", "Xing4V4Config", "Xing4V41Config"]


class Xing4Config(PretrainedConfig):
    """Common configuration interface; instantiate one of the two model subclasses.

    HF architecture fields retain their original names. Framework options can be
    supplied through kwargs. Explicit values override the selected model defaults.
    """

    model_type = "xing4"
    keys_to_ignore_at_inference = ["past_key_values"]
    _MODEL_DEFAULTS = {}
    _HF_DEFAULTS = {
        "vocab_size": 131072,
        "hidden_size": None,
        "intermediate_size": None,
        "moe_intermediate_size": None,
        "num_hidden_layers": None,
        "num_attention_heads": None,
        "num_key_value_heads": None,
        "n_routed_experts": None,
        "n_shared_experts": 1,
        "num_experts_per_tok": None,
        "routed_scaling_factor": None,
        "first_k_dense_replace": None,
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "n_group": None,
        "topk_group": None,
        "q_lora_rank": None,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "index_n_heads": None,
        "index_head_dim": 128,
        "index_topk": 2048,
        "num_nextn_predict_layers": 1,
        "rms_norm_eps": None,
        "initializer_range": None,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "max_position_embeddings": 32768,
        "rope_theta": 100000.0,
        "rope_scaling": {
            "type": "yarn", "factor": 8.0, "original_max_position_embeddings": 32768,
            "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0,
        },
        "hc_eps": 1e-6,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "mhc_h_res_clamp_min": None,
        "mhc_h_res_clamp_max": None,
    }
    _NATIVE_ALIASES = {
        "ffn_hidden_size": "intermediate_size",
        "mhc_layernorm_epsilon": "hc_eps",
        "num_residual_streams": "hc_mult",
        "mhc_sinkhorn_iterations": "hc_sinkhorn_iters",
        "dsa_indexer_n_heads": "index_n_heads",
        "dsa_indexer_head_dim": "index_head_dim",
        "dsa_indexer_topk": "index_topk",
    }

    @register_mf_model_parameter(
        seq_length=4096,
        params_dtype="bf16",
        embedding_params_dtype="bf16",
        qk_layernorm=True,
        hidden_dropout=0.0,
        add_bias_linear=False,
        gated_linear_unit=True,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_fusion=True,
        moe_grouped_gemm=True,
        experimental_attention_variant="dsa",
        dsa_indexer_use_sparse_loss=True,
        mtp_loss_scaling_factor=0.1,
        mtp_enable_dsa=False,
        mtp_enable_hyper_connections=False,
        enable_hc_head=False,
    )
    @ignore_and_delete_parameter(extra_ignore_param=[
        ("scoring_func", NotSupportedInfo.useless),
        ("expert_dtype", NotSupportedInfo.useless),
        ("ep_size", NotSupportedInfo.useless),
        ("dtype", NotSupportedInfo.useless),
    ])
    def __init__(self, **kwargs):
        if not self._MODEL_DEFAULTS:
            raise TypeError("Choose Xing4V4Config (29B) or Xing4V41Config (285B).")
        values = deepcopy(self._HF_DEFAULTS)
        values.update(deepcopy(self._MODEL_DEFAULTS))
        values.update(kwargs)
        for native, hf_name in self._NATIVE_ALIASES.items():
            if native in kwargs:
                values[hf_name] = values.pop(native)
        if "hc_sinkhorn_clamp" in values:
            clamp = values.pop("hc_sinkhorn_clamp")
            if clamp is None:
                bounds = (None, None)
            elif isinstance(clamp, (list, tuple)):
                if len(clamp) != 2:
                    raise ValueError(f"hc_sinkhorn_clamp sequence must have 2 elements, got {clamp}.")
                try:
                    bounds = tuple(float(value) for value in clamp)
                except (TypeError, ValueError) as error:
                    raise ValueError("hc_sinkhorn_clamp bounds must be numeric.") from error
            else:
                try:
                    magnitude = abs(float(clamp))
                except (TypeError, ValueError) as error:
                    raise ValueError("hc_sinkhorn_clamp must be a scalar, a two-element sequence or None.") from error
                bounds = (-magnitude, magnitude)
            values["mhc_h_res_clamp_min"], values["mhc_h_res_clamp_max"] = bounds
        for name in self._HF_DEFAULTS:
            setattr(self, name, values.pop(name))
        values.pop("model_type", None)
        if isinstance(values.get("parallel_config"), dict):
            values["parallel_config"] = TransformerOpParallelConfig(**values["parallel_config"])
        self.actual_vocab_size = self.vocab_size
        super().__init__(**values)
        indexer_types = getattr(self, "dsa_indexer_types", None)
        has_shared_indexer = indexer_types is not None and any(str(kind).lower() != "full" for kind in indexer_types)
        if getattr(self, "dsa_indexer_topk_freq", 1) != 1 or has_shared_indexer:
            raise ValueError("Xing4 supports independent per-layer DSA indexers; index sharing is unsupported.")

    @classmethod
    def _config_class(cls, config_dict):
        """Resolve model identity without inferring it from parameter dimensions."""
        model_type = config_dict.get("model_type")
        aliases = {"xing4_0": Xing4V4Config, "deepseek_v32": Xing4V41Config}
        if model_type in aliases:
            return aliases[model_type]
        for config_class in (Xing4V4Config, Xing4V41Config):
            if model_type == config_class.model_type or config_dict.get("type") == config_class.__name__:
                return config_class
        raise ValueError(f"Unknown Xing4 model_type {model_type!r}; expected xing4_v4/xing4_0 "
                         "(29B) or xing4_v41/deepseek_v32 (285B).")

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        if cls is Xing4Config:
            return cls._config_class(config_dict).from_dict(config_dict, **kwargs)
        config_dict = dict(config_dict)
        if config_dict.get("model_type") == "xing4_0" and not any(
                config_dict.get(name) is not None for name in ("index_n_heads", "dsa_indexer_n_heads")):
            config_dict.setdefault("experimental_attention_variant", None)
        if "vocab_size" in kwargs and "actual_vocab_size" not in kwargs:
            config_dict.pop("actual_vocab_size", None)
        constructor_keys = (cls._HF_DEFAULTS.keys() | cls._MODEL_DEFAULTS.keys() |
                            cls._NATIVE_ALIASES.keys() | {"parallel_config", "hc_sinkhorn_clamp"})
        for name in constructor_keys.intersection(kwargs):
            config_dict[name] = kwargs.pop(name)
        return super().from_dict(config_dict, **kwargs)

    @classmethod
    def from_json_file(cls, json_file):
        return cls.from_dict(cls._dict_from_json_file(json_file))

    @classmethod
    def from_pretrained(cls, yaml_name_or_path, **kwargs):
        if cls is Xing4Config:
            config_dict, kwargs = cls.get_config_dict(yaml_name_or_path, **kwargs)
            return cls.from_dict(config_dict, **kwargs)
        return super().from_pretrained(yaml_name_or_path, **kwargs)

    def convert_to_transformer_config(self, is_mla_model=True):
        """Convert either Xing4 model to the common MLA transformer configuration."""
        return Xing4ConfigConverter.convert(self.to_dict(), is_mla_model=is_mla_model)


@MindFormerRegister.register(MindFormerModuleType.CONFIG, legacy=False, search_names=["xing4_v4"])
class Xing4V4Config(Xing4Config):
    """Xing4 29B: mHC, independent DSA indexers and one plain MTP layer."""

    model_type = "xing4_v4"
    _MODEL_DEFAULTS = {
        "hidden_size": 3584, "intermediate_size": 9216, "moe_intermediate_size": 1024,
        "num_hidden_layers": 40, "num_attention_heads": 32, "num_key_value_heads": 32,
        "n_routed_experts": 64, "num_experts_per_tok": 4, "routed_scaling_factor": 2.0,
        "first_k_dense_replace": 2, "n_group": 1, "topk_group": 1,
        "q_lora_rank": 768, "index_n_heads": 16,
        "rms_norm_eps": 1e-6, "initializer_range": 0.02,
        "enable_hyper_connections": True, "mhc_compute_dtype": "bf16",
        "mhc_megatron_compatibility": True,
        "mhc_h_res_clamp_min": -30, "mhc_h_res_clamp_max": 30,
        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": None,
    }


@MindFormerRegister.register(MindFormerModuleType.CONFIG, legacy=False, search_names=["xing4_v41"])
class Xing4V41Config(Xing4Config):
    """Xing4.1-Preview 285B configuration with independent DSA indexers."""

    model_type = "xing4_v41"
    _MODEL_DEFAULTS = {
        "hidden_size": 4096, "intermediate_size": 18432, "moe_intermediate_size": 2048,
        "num_hidden_layers": 45, "num_attention_heads": 128, "num_key_value_heads": 128,
        "n_routed_experts": 256, "num_experts_per_tok": 8, "routed_scaling_factor": 2.5,
        "first_k_dense_replace": 3, "n_group": 1, "topk_group": 1,
        "q_lora_rank": 1536, "index_n_heads": 64,
        "rms_norm_eps": 1e-5, "initializer_range": 0.0048,
        "enable_hyper_connections": False,
        "num_nextn_predict_layers": 0, "max_position_embeddings": 1048576,
        "rope_scaling": {
            "type": "yarn", "factor": 32.0, "original_max_position_embeddings": 32768,
            "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0,
        },
        "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
    }
