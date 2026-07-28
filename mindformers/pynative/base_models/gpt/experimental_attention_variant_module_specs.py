# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright 2026 Huawei Technologies Co., Ltd
#
# This file is derived from Megatron-LM and adapted for MindSpore.
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
"""ModuleSpec registry for experimental attention variants."""
from typing import Callable, Dict

from mindformers.pynative.layers.identity_op import IdentityOp
from mindformers.pynative.layers.layer_norm import get_norm_cls
from mindformers.pynative.layers.linear import Linear
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.pynative.transformers.multi_latent_attention import MLASelfAttentionSubmodules
from mindformers.pynative.transformers.experimental_attention_variant.dsa import (
    DSAttention,
    DSAttentionSubmodules,
)
from mindformers.pynative.transformers.experimental_attention_variant.dsa_attention import (
    DSASelfAttention,
)
from mindformers.pynative.transformers.experimental_attention_variant.dsa_indexer import (
    DSAIndexer,
    DSAIndexerSubmodules,
)
from mindformers.pynative.transformers.experimental_attention_variant.compressor import (
    Compressor,
    CompressorSubmodules
)
from mindformers.pynative.transformers.experimental_attention_variant.csa import (
    CompressedSparseAttention,
    CompressedSparseAttentionSubmodules
)
from mindformers.pynative.transformers.experimental_attention_variant.indexer import (
    CSAIndexer,
    CSAIndexerSubmodules,
)
from mindformers.pynative.transformers.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttention,
    DSv4HybridSelfAttentionSubmodules,
)


AttentionVariantSpecBuilder = Callable[[bool, bool, str], ModuleSpec]
_ATTENTION_VARIANT_SPEC_BUILDERS: Dict[str, AttentionVariantSpecBuilder] = {}


def register_attention_variant(name: str):
    """Register a ModuleSpec builder for one experimental attention variant."""
    if not name:
        raise ValueError("Attention variant name must not be empty.")

    def decorator(builder: AttentionVariantSpecBuilder):
        if name in _ATTENTION_VARIANT_SPEC_BUILDERS:
            raise ValueError(f"Attention variant {name!r} is already registered.")
        _ATTENTION_VARIANT_SPEC_BUILDERS[name] = builder
        return builder

    return decorator


def get_attention_variant_module_spec(
        name: str,
        qk_layernorm: bool = False,
        fused_norm: bool = True,
        normalization: str = "RMSNorm",
) -> ModuleSpec:
    """Build the top-level ModuleSpec registered for ``name``."""
    try:
        builder = _ATTENTION_VARIANT_SPEC_BUILDERS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_ATTENTION_VARIANT_SPEC_BUILDERS))
        raise ValueError(
            f"Unsupported experimental attention variant {name!r}. "
            f"Registered variants: {supported or 'none'}."
        ) from exc
    return builder(qk_layernorm, fused_norm, normalization)


@register_attention_variant("dsa")
def get_dsa_module_spec(
        qk_layernorm: bool = False,
        fused_norm: bool = True,
        normalization: str = "RMSNorm",
) -> ModuleSpec:
    """Build the complete DSA self-attention ModuleSpec."""
    norm_cls = get_norm_cls(normalization, fused_norm)
    return ModuleSpec(
        module=DSASelfAttention,
        submodules=MLASelfAttentionSubmodules(
            linear_qkv=Linear,
            linear_qb=Linear,
            linear_kvb=Linear,
            core_attention=ModuleSpec(
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
                    ),
                ),
            ),
            linear_proj=Linear,
            q_layernorm=norm_cls if qk_layernorm else IdentityOp,
            k_layernorm=norm_cls if qk_layernorm else IdentityOp,
        ),
    )


@register_attention_variant("dsv4_hybrid")
def get_dsv4_hybrid_module_spec(
        qk_layernorm: bool = False,
        fused_norm: bool = True,
        normalization: str = "RMSNorm",
) -> ModuleSpec:
    """Build the DSv4 hybrid attention ModuleSpec for one transformer layer (RFC §3.4.2.8).

    Returned spec tree:

      ::

        DSv4HybridSelfAttention
          ├─ q_layernorm: RMSNorm/LayerNorm or IdentityOp (per qk_layernorm)
          ├─ kv_layernorm: same
          ├─ linear_q_down_proj: Linear
          ├─ linear_q_up_proj: Linear
          ├─ linear_kv_proj: Linear
          ├─ linear_proj: Linear
          └─ core_attention: CompressedSparseAttention
                ├─ compressor: Compressor (rotate=False, ratio per layer)
                │     ├─ linear_wkv: Linear
                │     ├─ linear_wgate: Linear
                │     └─ norm: RMSNorm
                └─ indexer: CSAIndexer
                      ├─ linear_wq_b: Linear
                      ├─ linear_weights_proj: Linear
                      └─ compressor: Compressor (rotate=True)
                            ├─ linear_wkv: Linear
                            ├─ linear_wgate: Linear
                            └─ norm: RMSNorm

    Args:
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fused_norm (bool): Whether to use fused-normalization. Defaults to True.
        normalization (str): The type of the norm. Defaults to RMSNorm.

    Returns:
        ModuleSpec with ``module=DSv4HybridSelfAttention``.
    """
    norm_cls = get_norm_cls(normalization, fused_norm)
    q_norm = norm_cls if qk_layernorm else IdentityOp
    kv_norm = norm_cls if qk_layernorm else IdentityOp

    compressor_spec = ModuleSpec(
        module=Compressor,
        submodules=CompressorSubmodules(
            linear_wkv=Linear,
            linear_wgate=Linear,
            norm=get_norm_cls("RMSNorm", fused_norm),
        ),
    )

    # The inner indexer compressor is structurally identical to the outer compressor,
    # so reuse the same ``compressor_spec`` here instead of rebuilding it.
    indexer_spec = ModuleSpec(
        module=CSAIndexer,
        submodules=CSAIndexerSubmodules(
            linear_wq_b=Linear,
            linear_weights_proj=Linear,
            compressor=compressor_spec,
        ),
    )

    core_attention_spec = ModuleSpec(
        module=CompressedSparseAttention,
        submodules=CompressedSparseAttentionSubmodules(
            compressor=compressor_spec,
            indexer=indexer_spec,
        ),
    )

    return ModuleSpec(
        module=DSv4HybridSelfAttention,
        submodules=DSv4HybridSelfAttentionSubmodules(
            q_layernorm=q_norm,
            kv_layernorm=kv_norm,
            linear_q_down_proj=Linear,
            linear_q_up_proj=Linear,
            linear_kv_proj=Linear,
            core_attention=core_attention_spec,
            linear_proj=Linear,
        ),
    )
