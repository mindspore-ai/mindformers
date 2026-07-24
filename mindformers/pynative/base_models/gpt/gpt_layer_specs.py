# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Modification points:
# 1. Replace all interfaces with MindSpore TransFormers'.
# 2. Modify some input parameters for MindSpore TransFormers.
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
"""GPT LayerSpec."""
from typing import Optional, Union

from mindformers.pynative.layers.linear import Linear
from mindformers.pynative.transformers.attention import SelfAttentionSubmodules, SelfAttention
from mindformers.pynative.layers.flash_attention import FlashAttention
from mindformers.pynative.layers.identity_op import IdentityOp
from mindformers.pynative.transformers.mlp import MLP, MLPSubmodules
from mindformers.pynative.layers.layer_norm import get_norm_cls
from mindformers.pynative.transformers.transformer_block import TransformerBlockSubmodules
from mindformers.pynative.transformers.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
    TransformerLayerSubmodules,
)
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.pynative.base_models.gpt.moe_module_specs import get_moe_module_spec
from mindformers.pynative.base_models.gpt.experimental_attention_variant_module_specs import (
    get_dsv4_hybrid_module_spec,
)
from mindformers.pynative.transformers.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules
)
from mindformers.pynative.transformers.multi_token_prediction import (
    MultiTokenPredictionBlockSubmodules,
    get_mtp_layer_spec,
)
from mindformers.pynative.transformers.hyper_connection import HyperConnectionHead
from mindformers.pynative.transformers.experimental_attention_variant.dsa import (
    DSAttention,
    DSAttentionSubmodules,
)
from mindformers.pynative.transformers.experimental_attention_variant.dsa_indexer import (
    DSAIndexer,
    DSAIndexerSubmodules,
)


def get_attention_module_spec(
        sparse_attention: Optional[bool] = False,
        fused_norm: Optional[bool] = True,
) -> ModuleSpec:
    """Helper function to get module spec for Core Attention.

    Args:
        sparse_attention: If True, use DSAttention with DSAIndexer for sparse attention.
        fused_norm: Whether to use fused normalization.

    Returns:
        ModuleSpec for core attention (DSAttention or FlashAttention).
    """
    if sparse_attention:
        return ModuleSpec(
            module=DSAttention,
            submodules=DSAttentionSubmodules(
                indexer=ModuleSpec(
                    module=DSAIndexer,
                    submodules=DSAIndexerSubmodules(
                        linear_wq_b=Linear,
                        linear_wk=Linear,
                        k_norm=get_norm_cls("LayerNorm", fused_norm),
                        linear_weights_proj=Linear,
                    ),
                )
            ),
        )
    return FlashAttention

def get_mlp_module_spec(
        num_experts: Optional[int] = None,
        moe_grouped_gemm: Optional[bool] = True,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    mlp = MLP
    if not num_experts:
        return ModuleSpec(
            module=mlp,
            submodules=MLPSubmodules(
                linear_fc1=Linear,
                linear_fc2=Linear,
            ),
        )

    return get_moe_module_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )

def get_gpt_layer_local_spec(
        num_experts: Optional[int] = None,
        moe_grouped_gemm: Optional[bool] = False,
        qk_layernorm: Optional[bool] = False,
        multi_latent_attention: Optional[bool] = False,
        enable_hyper_connections: Optional[bool] = False,
        sparse_attention: Optional[bool] = False,
        fused_norm: Optional[bool] = True,
        normalization: Optional[str] = "RMSNorm",
        is_dsv4_hybrid: Optional[bool] = False
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        multi_latent_attention (bool, optional): To use MultiLatentAttention. Defaults to False.
        enable_hyper_connections (bool, optional): Whether to build HyperConnectionTransformerLayer.
            Defaults to False.
        fused_norm (bool): Whether to use fused-normalization. Defaults to True.
        normalization (str): The type of the norm. Defaults to RMSNorm.
    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """

    mlp = get_mlp_module_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )
    layer_cls = HyperConnectionTransformerLayer if enable_hyper_connections else TransformerLayer

    # DSv4 hybrid attention selects its self-attention spec directly here, rather than rebuilding a base layer
    # and swapping the self_attention slot afterwards.
    if is_dsv4_hybrid:
        return ModuleSpec(
            module=layer_cls,
            submodules=TransformerLayerSubmodules(
                input_layernorm=get_norm_cls(normalization, fused_norm),
                self_attention=get_dsv4_hybrid_module_spec(qk_layernorm, fused_norm, normalization),
                pre_mlp_layernorm=get_norm_cls(normalization, fused_norm),
                mlp=mlp,
            ),
        )

    if multi_latent_attention:
        core_attention = get_attention_module_spec(sparse_attention, fused_norm)
        self_attention = ModuleSpec(
            module=MLASelfAttention,
            submodules=MLASelfAttentionSubmodules(
                linear_qkv=Linear,
                linear_qb=Linear,
                linear_kvb=Linear,
                core_attention=core_attention,
                linear_proj=Linear,
                q_layernorm=get_norm_cls(normalization, fused_norm) if qk_layernorm else IdentityOp,
                k_layernorm=get_norm_cls(normalization, fused_norm) if qk_layernorm else IdentityOp,
            ),
        )
        return ModuleSpec(
            module=layer_cls,
            submodules=TransformerLayerSubmodules(
                input_layernorm=get_norm_cls(normalization, fused_norm),
                self_attention=self_attention,
                pre_mlp_layernorm=get_norm_cls(normalization, fused_norm),
                mlp=mlp,
            ),
        )

    return ModuleSpec(
        module=layer_cls,
        submodules=TransformerLayerSubmodules(
            input_layernorm=get_norm_cls(normalization, fused_norm),
            self_attention=ModuleSpec(
                module=SelfAttention,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=Linear,
                    core_attention=FlashAttention,
                    linear_proj=Linear,
                    q_layernorm=get_norm_cls(normalization, fused_norm) if qk_layernorm else IdentityOp,
                    k_layernorm=get_norm_cls(normalization, fused_norm) if qk_layernorm else IdentityOp,
                ),
            ),
            pre_mlp_layernorm=get_norm_cls(normalization, fused_norm),
            mlp=mlp
        )
    )


def get_gpt_decoder_block_spec(config: TransformerConfig) -> TransformerBlockSubmodules:
    """GPT block spec.

    Args:
        config (TransformerConfig): Transformer configuration.
    """

    # Layer specs.
    sparse_attention = getattr(config, "experimental_attention_variant", None) == "dsa"
    dense_layer_spec = get_gpt_layer_local_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        enable_hyper_connections=config.enable_hyper_connections,
        sparse_attention=sparse_attention,
        fused_norm=config.fused_norm,
        is_dsv4_hybrid=config.experimental_attention_variant == "dsv4_hybrid",
    )

    moe_layer_spec = get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        enable_hyper_connections=config.enable_hyper_connections,
        sparse_attention=sparse_attention,
        fused_norm=config.fused_norm,
        is_dsv4_hybrid=config.experimental_attention_variant == "dsv4_hybrid",
    )

    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
    if config.first_k_dense_replace:
        moe_layer_pattern = [0] * config.first_k_dense_replace + \
                            [1] * (config.num_layers - config.first_k_dense_replace)
    elif isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_layers)]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq
        if len(moe_layer_pattern) != config.num_layers:
            raise ValueError(f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
                             f"expected {config.num_layers}, "
                             f"current moe layer pattern: {config.moe_layer_freq}")
    else:
        raise ValueError(
            f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}"
        )

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_layer_spec)
        elif moe_layer_pattern[layer_number] == 0:
            layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    # Block spec.
    block_spec = TransformerBlockSubmodules(
        layer_specs=layer_specs,
        layer_norm=get_norm_cls(config.normalization, config.fused_norm),
        hc_head=HyperConnectionHead if config.enable_hc_head else None,
    )

    return block_spec


def get_gpt_mtp_block_spec(
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        normalization: Optional[str] = "RMSNorm",
) -> MultiTokenPredictionBlockSubmodules:
    """GPT Multi-Token Prediction (MTP) block spec."""
    num_layers_to_build = config.mtp_num_layers if config.mtp_num_layers else 0
    if num_layers_to_build == 0:
        return None

    if isinstance(spec, TransformerBlockSubmodules):
        # get the spec for the last layer of decoder block
        transformer_layer_spec = spec.layer_specs[-1]
        hc_head = spec.hc_head
    elif isinstance(spec, ModuleSpec) and spec.module == TransformerLayer:
        transformer_layer_spec = spec
        hc_head = None
    else:
        raise ValueError(f"Invalid spec: {spec}")

    mtp_layer_spec = get_mtp_layer_spec(
        transformer_layer_spec=transformer_layer_spec,
        normalization=normalization,
        fused_norm=config.fused_norm,
        hc_head=hc_head,
    )
    mtp_num_layers = config.mtp_num_layers if config.mtp_num_layers else 0
    mtp_layer_specs = [mtp_layer_spec] * mtp_num_layers

    return MultiTokenPredictionBlockSubmodules(layer_specs=mtp_layer_specs)
