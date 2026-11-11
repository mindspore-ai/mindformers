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
from dataclasses import replace
from typing import Optional, Union

from mindformers.pynative.layers.linear import Linear
from mindformers.pynative.transformers.attention import SelfAttentionSubmodules, SelfAttention
from mindformers.pynative.layers.flash_attention import FlashAttention
from mindformers.pynative.layers.identity_op import IdentityOp
from mindformers.pynative.transformers.mlp import MLP, MLPSubmodules
from mindformers.pynative.layers.layer_norm import get_norm_cls
from mindformers.pynative.transformers.transformer_block import TransformerBlockSubmodules
from mindformers.pynative.transformers.transformer_layer import (
    BaseTransformerLayer,
    HyperConnectionTransformerLayer,
    TransformerLayer,
    TransformerLayerSubmodules,
)
from mindformers.pynative.transformers.experimental_attention_variant.dsa_transformer_layer import (
    DSAHyperConnectionTransformerLayer,
    DSATransformerLayer,
)
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.tools import logger
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.pynative.base_models.gpt.moe_module_specs import get_moe_module_spec
from mindformers.pynative.base_models.gpt.experimental_attention_variant_module_specs import (
    get_attention_variant_module_spec,
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


def get_self_attention_module_spec(
        qk_layernorm: bool = False,
        multi_latent_attention: bool = False,
        fused_norm: bool = True,
        normalization: str = "RMSNorm",
) -> ModuleSpec:
    """Build the ordinary self-attention spec, i.e. the one no attention variant replaced.

    Also used to walk a layer back from an attention variant to plain attention, which is
    how the MTP layers drop DSA while the decoder keeps it (see get_gpt_mtp_block_spec).

    Args:
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        multi_latent_attention (bool, optional): To use MultiLatentAttention. Defaults to False.
        fused_norm (bool): Whether to use fused-normalization. Defaults to True.
        normalization (str): The type of the norm. Defaults to RMSNorm.

    Returns:
        ModuleSpec: ``MLASelfAttention`` when ``multi_latent_attention`` else ``SelfAttention``.
    """
    q_layernorm = get_norm_cls(normalization, fused_norm) if qk_layernorm else IdentityOp
    k_layernorm = get_norm_cls(normalization, fused_norm) if qk_layernorm else IdentityOp
    if multi_latent_attention:
        return ModuleSpec(
            module=MLASelfAttention,
            submodules=MLASelfAttentionSubmodules(
                linear_qkv=Linear,
                linear_qb=Linear,
                linear_kvb=Linear,
                core_attention=FlashAttention,
                linear_proj=Linear,
                q_layernorm=q_layernorm,
                k_layernorm=k_layernorm,
            ),
        )
    return ModuleSpec(
        module=SelfAttention,
        submodules=SelfAttentionSubmodules(
            linear_qkv=Linear,
            core_attention=FlashAttention,
            linear_proj=Linear,
            q_layernorm=q_layernorm,
            k_layernorm=k_layernorm,
        ),
    )


def get_gpt_layer_local_spec(
        num_experts: Optional[int] = None,
        moe_grouped_gemm: Optional[bool] = False,
        qk_layernorm: Optional[bool] = False,
        multi_latent_attention: Optional[bool] = False,
        enable_hyper_connections: Optional[bool] = False,
        attention_variant: Optional[str] = None,
        fused_norm: Optional[bool] = True,
        normalization: Optional[str] = "RMSNorm",
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        multi_latent_attention (bool, optional): To use MultiLatentAttention. Defaults to False.
        enable_hyper_connections (bool, optional): Whether to build HyperConnectionTransformerLayer.
            Defaults to False.
        attention_variant (str, optional): Registered experimental attention variant.
            The legacy value ``"mla"`` selects the regular MLA implementation.
        fused_norm (bool): Whether to use fused-normalization. Defaults to True.
        normalization (str): The type of the norm. Defaults to RMSNorm.
    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """

    mlp = get_mlp_module_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )
    if attention_variant == "dsa":
        layer_cls = (
            DSAHyperConnectionTransformerLayer
            if enable_hyper_connections
            else DSATransformerLayer
        )
    else:
        layer_cls = HyperConnectionTransformerLayer if enable_hyper_connections else TransformerLayer

    if attention_variant == "mla":
        if not multi_latent_attention:
            raise ValueError(
                "When attention_variant is 'mla', multi_latent_attention must be True."
            )
        attention_variant = None

    if attention_variant:
        self_attention = get_attention_variant_module_spec(
            attention_variant,
            qk_layernorm,
            fused_norm,
            normalization,
        )
    else:
        self_attention = get_self_attention_module_spec(
            qk_layernorm=qk_layernorm,
            multi_latent_attention=multi_latent_attention,
            fused_norm=fused_norm,
            normalization=normalization,
        )

    return ModuleSpec(
        module=layer_cls,
        submodules=TransformerLayerSubmodules(
            input_layernorm=get_norm_cls(normalization, fused_norm),
            self_attention=self_attention,
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
    dense_layer_spec = get_gpt_layer_local_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        enable_hyper_connections=config.enable_hyper_connections,
        attention_variant=config.experimental_attention_variant,
        fused_norm=config.fused_norm,
    )

    moe_layer_spec = get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        enable_hyper_connections=config.enable_hyper_connections,
        attention_variant=config.experimental_attention_variant,
        fused_norm=config.fused_norm,
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


# mHC layer class -> its plain counterpart, for mtp_enable_hyper_connections=False.
_HC_TO_PLAIN_LAYER_CLS = {
    HyperConnectionTransformerLayer: TransformerLayer,
    DSAHyperConnectionTransformerLayer: DSATransformerLayer,
}

# DSA layer class -> its plain counterpart, for mtp_enable_dsa=False. mHC is orthogonal, so
# each flavour maps to the one that keeps its own mHC-ness; dropping that too is the other
# table's job, applied afterwards.
_DSA_TO_PLAIN_LAYER_CLS = {
    DSATransformerLayer: TransformerLayer,
    DSAHyperConnectionTransformerLayer: HyperConnectionTransformerLayer,
}


def _requested_mtp_downgrades(config):
    """Which MTP-only switches ask the MTP layers to be plainer than the decoder's.

    One list for both spec shapes, so the branch that can honour them and the branch that
    cannot agree on what was asked for.
    """
    downgrades = []
    if config.experimental_attention_variant == 'dsa' and not config.mtp_enable_dsa:
        downgrades.append("mtp_enable_dsa")
    if config.enable_hyper_connections and not config.mtp_enable_hyper_connections:
        downgrades.append("mtp_enable_hyper_connections")
    return downgrades


def _plain_layer_cls(table, layer_cls, switch_name):
    """The plain counterpart of ``layer_cls``, or a ValueError naming the switch.

    A custom decoder layer class is not in either table; without this it would surface as a
    bare ``KeyError: <class ...>`` from a line that says nothing about which switch asked.
    """
    try:
        return table[layer_cls]
    except KeyError:
        raise ValueError(
            f"{switch_name}=False needs a plain counterpart for the decoder's layer class "
            f"{layer_cls.__name__}, and none is registered. Add one to the table in "
            f"gpt_layer_specs.py, or leave the switch unset so the MTP layers follow the "
            f"decoder."
        ) from None


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
        downgrades = _requested_mtp_downgrades(config)
        if "mtp_enable_dsa" in downgrades:
            # The decoder keeps DSA but the MTP layers must not, which is what graph mode does
            # unconditionally. Two things carry DSA: the layer class (it owns the warm-up
            # boundary) and the self-attention spec (DSASelfAttention, whose core_attention
            # holds the indexer). Swap both back to their plain counterparts so no indexer
            # parameter is built under ``mtp.``; the MLP and the norms are left untouched.
            transformer_layer_spec = replace(
                transformer_layer_spec,
                module=_plain_layer_cls(
                    _DSA_TO_PLAIN_LAYER_CLS, transformer_layer_spec.module, "mtp_enable_dsa"),
                submodules=replace(
                    transformer_layer_spec.submodules,
                    self_attention=get_self_attention_module_spec(
                        qk_layernorm=config.qk_layernorm,
                        multi_latent_attention=config.multi_latent_attention,
                        fused_norm=config.fused_norm,
                        normalization=normalization,
                    ),
                ),
            )
        if "mtp_enable_hyper_connections" in downgrades:
            # The decoder keeps mHC but the MTP layers must not. The two layer flavours differ
            # only in the layer class -- ``attn_hc``/``ffn_hc`` are built inside
            # HyperConnectionTransformerLayer itself, not declared in the spec -- so swapping
            # the class is enough to leave no mHC parameters under ``mtp.``. The collapse head
            # goes with it: the decoder collapses its own residual streams before the hidden
            # states reach the MTP block, which therefore only ever sees (s, b, h).
            transformer_layer_spec = replace(
                transformer_layer_spec,
                module=_plain_layer_cls(
                    _HC_TO_PLAIN_LAYER_CLS, transformer_layer_spec.module,
                    "mtp_enable_hyper_connections"),
            )
            hc_head = None
    elif (isinstance(spec, ModuleSpec)
          and isinstance(spec.module, type)
          and issubclass(spec.module, BaseTransformerLayer)):
        transformer_layer_spec = spec
        hc_head = None
        ignored = _requested_mtp_downgrades(config)
        if ignored:
            # This shape is a single layer spec handed in directly, with no decoder block to
            # walk a layer back from -- the caller already chose the MTP layer class. Honouring
            # the switches here would mean second-guessing that choice, so they are ignored;
            # say so, because a config that sets them would otherwise look like it took effect.
            logger.warning(
                "%s set but ignored: the MTP block was given an explicit layer spec (%s), which "
                "fixes the MTP layer class regardless of the decoder's.",
                " and ".join(ignored), spec.module.__name__,
            )
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
