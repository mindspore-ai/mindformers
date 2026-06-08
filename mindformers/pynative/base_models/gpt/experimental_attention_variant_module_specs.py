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
"""DSv4 hybrid attention ModuleSpec builders."""
from mindformers.pynative.layers.identity_op import IdentityOp
from mindformers.pynative.layers.layer_norm import get_norm_cls
from mindformers.pynative.layers.linear import Linear
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
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
