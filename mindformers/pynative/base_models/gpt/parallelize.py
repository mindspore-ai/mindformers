# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Modification points:
# 1. Replace all interfaces with MindSpore Transformers'.
# 2. Add some input parameters for MindSpore Transformers.
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

"""GPT model parallelization for MindFormers PyNative MCore architecture.

Applies parallelism strategies in TorchTitan order:
  1. Tensor Parallelism (TP)   - if enabled (NotImplemented)
  2. Context Parallelism (CP)  - if enabled (NotImplemented)
  3. Activation Checkpointing  - if enabled (NotImplemented)
  4. FSDP/HSDP                 - if enabled

This module is the single source of truth for GPT-family model parallelization.
"""

import os
from typing import Any, List

import mindspore as ms
from mindspore import nn, Parameter, Tensor
from mindspore.ops.communication import set_comm_ops_inplace
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore.mint.distributed import get_rank, all_gather, new_group

from hyper_parallel import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.fully_shard.api import (
    fully_shard,
    MixedPrecisionPolicy,
    HSDPModule,
)
from hyper_parallel.core.fully_shard.utils import OffloadPolicy, CPUOffloadPolicy

from mindformers.pynative.distributed.fsdp import (
    get_fsdp_reshard_after_forward_policy,
    disable_fsdp_gradient_division,
)
from mindformers.pynative.distributed.style import (
    ParallelStyle,
    ColwiseParallel,
    AllGather,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
    ShardTensor,
)
from mindformers.pynative.distributed.mc2_style import (
    MC2ColwiseParallel,
    MC2RowwiseParallel,
)
from mindformers.pynative.distributed.context_parallel import (
    apply_context_parallel_model_io,
    build_context_parallel_attention_style,
)
from mindformers.pynative.distributed.csa_context_parallel import DSv4HybridAttentionContextParallel
from mindformers.pynative.layers.layer_norm import FusedLayerNorm, FusedRMSNorm
from mindformers.pynative.distributed.tensor_parallel import NoParallel
from mindformers.pynative.distributed.expert_parallel import ExpertParallel, DeredundancyExpertParallel
from mindformers.pynative.distributed.ep_overlap import OverlapExpertParallel
from mindformers.pynative.distributed.pipeline_parallel import PpLayerSetting, StageModelBuilder, _create_schedule, _infer_schedule_type
from mindformers.pynative.pet.lora_adapter import build_lora_model
from mindformers.pynative.distributed.parallelize import parallelize_module
from mindformers.pynative.distributed.activation_checkpoint import apply_ac
from mindformers.pynative.distributed.utils import get_loss_sense
from mindformers.pynative.layers.flash_attention import FlashAttention
from mindformers.pynative.base_models.gpt.gpt_model import GPTModel
from mindformers.pynative.transformers.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttention,
)
from mindformers.pynative.transformers.hyper_connection import FusedHyperConnectionModule
from mindformers.pynative.transformers.multi_latent_attention import MLASelfAttention
from mindformers.pynative.transformers.mlp import MLP
from mindformers.pynative.transformers.moe.moe_layer import MoELayer
from mindformers.pynative.transformers.moe.moe_utils import set_moe_aux_loss_group_info
from mindformers.pynative.transformers.experimental_attention_variant.csa import CompressedSparseAttention
from mindformers.tools.logger import logger

__all__ = ["parallelize_gptmodel"]

_DTYPE_MAP = {"float16": ms.float16, "float32": ms.float32, "bfloat16": ms.bfloat16}


# ---------------------------------------------------------------------------
# GPT-specific helpers
# ---------------------------------------------------------------------------


def _unwrap_gptmodel(model: nn.Cell) -> nn.Cell:
    """Unwrap GPTModel from standard wrappers.

    Uses MindSpore's cells_and_names() to locate GPTModel within the model
    hierarchy. This avoids hard-coding wrapper attribute names.
    """
    if isinstance(model, GPTModel):
        return model

    gpt_models = [
        m for _, m in model.cells_and_names()
        if isinstance(m, GPTModel)
    ]

    if not gpt_models:
        raise ValueError(
            f"No GPTModel found in {type(model).__name__}. "
            f"apply_fsdp_to_gptmodel only supports GPT-family models."
        )

    return gpt_models[0]


def _setup_mtp_embedding_grad_sync(model_parts, parallel_dims):
    """Tag the MTP-shared input embedding for cross-PP-stage gradient sync.

    Under PP + MTP, the input embedding is replicated on stage 0 (main forward)
    and the last stage (MTP block reuses it for shifted tokens). Unlike the
    single-card baseline where one tensor receives ``main_grad + mtp_grad``,
    each PP copy only sees its own partial gradient, so without sync the two
    copies drift apart and the MTP loss diverges.

    This builds a process group over the embedding-owning PP ranks (per
    dp/tp/cp/ep coordinate) and tags the local ``word_embeddings.weight`` with
    ``_embedding_grad_sync_group`` / ``_pp_replica_count``. The grad-norm helper
    (`_calculate_global_grad_norm` -> `_get_grad_factor`) then all-reduces the
    tagged gradient and counts the replicated embedding exactly once, with no
    trainer changes.

    ``model_parts`` is the full list of virtual-pipeline chunks owned by this
    rank. Under interleaving the main and MTP embeddings live in different
    chunks on different PP stages, so ownership must be aggregated across ALL
    chunks: per-chunk detection would make each ``all_gather`` see at most one
    owner, ``embed_ranks`` would never reach length 2, and the sync group would
    never form -- silently reintroducing the MTP drift described above.
    """
    if parallel_dims is None or not parallel_dims.pp_enabled:
        return

    if not isinstance(model_parts, (list, tuple)):
        model_parts = [model_parts]
    if not model_parts:
        return

    cfg = _unwrap_gptmodel(model_parts[0]).get_gpt_transformer_config()
    if not getattr(cfg, "mtp_num_layers", 0):
        return

    # Locate this rank's input-embedding weight(s) across ALL chunks it owns
    # (present on stage 0 and the MTP stage). With interleaving the main and MTP
    # embeddings sit in separate chunks, so we must scan every part -- not just
    # one -- to detect that this rank owns an embedding at all.
    embed_weights = [
        param
        for part in model_parts
        for name, param in part.parameters_and_names()
        if name.endswith("embedding.word_embeddings.weight")
    ]

    pp_mesh = parallel_dims.get_mesh("pp")
    pp_ranks = pp_mesh.get_rank_list_along_axis("pp")  # global ranks on this PP line
    local_rank = get_rank()
    owns_embedding = 1 if embed_weights else 0

    # Exchange the ownership flag over the PP group so every rank on the line
    # derives the SAME ``embed_ranks`` (required for a consistent ``new_group``).
    flag = Tensor([owns_embedding], dtype=ms.int32)
    gathered = [Tensor([0], dtype=ms.int32) for _ in pp_ranks]
    all_gather(gathered, flag, group=pp_mesh.get_group())
    embed_ranks = [
        int(r) for r, g in zip(pp_ranks, gathered) if int(g.asnumpy()[0]) == 1
    ]

    if len(embed_ranks) <= 1:
        # Single embedding owner on this line (MTP stage == stage 0, or MTP
        # disabled): nothing to sync.
        return

    # Only member ranks create/join the communicator; middle PP stages must NOT
    # call ``new_group`` with these ranks or HCCL raises "group doesn't contain
    # the global rank".
    if local_rank not in embed_ranks:
        return

    group = new_group(ranks=embed_ranks)
    for weight in embed_weights:
        weight._embedding_grad_sync_group = group
        weight._embedding_grad_sync_size = len(embed_ranks)
    logger.info(
        "[MTP-EmbedSync] rank %d tagged %d embedding weight(s) for grad-sync group %s",
        local_rank, len(embed_weights), embed_ranks,
    )


def is_norm_module(module):
    """Return True when module is a norm or an activation wrapper around a norm."""
    if isinstance(module, (FusedLayerNorm, FusedRMSNorm)):
        return True
    unwrapped = getattr(module, "unwrap_cell", None)
    if unwrapped is not None:
        return isinstance(unwrapped, (FusedLayerNorm, FusedRMSNorm))
    return False


def _layer_inner_hsdp_modules(layer: nn.Cell) -> List[nn.Cell]:
    """Return all FSDP-wrapped sub-modules nested inside a layer (excluding the layer itself)."""
    if not layer:
        return []
    return [m for _, m in layer.cells_and_names() if isinstance(m, HSDPModule) and m is not layer]


def _setup_gpt_prefetch(
        embedding: nn.Cell,
        transformer_layers: List[nn.Cell],
        tail_modules: List[nn.Cell],
) -> None:
    """Set up forward and backward prefetch chains for GPT FSDP modules.

    Each layer additionally prefetches its own inner fully_shard-wrapped
    sub-modules (norms, q/k norms inside self_attention, attn_hc/ffn_hc, MoE
    experts) so their all-gathers can overlap with the layer's forward/backward.
    """
    if not transformer_layers:
        return

    if tail_modules is None:
        logger.warning(
            "No final_layernorm or output_layer found for prefetch setup. "
            "Prefetch chain will be incomplete."
        )
    if embedding is None:
        logger.warning("No embedding found for prefetch setup.")

    # --- Forward prefetch: embedding -> layer[0] -> ... -> [final_layernorm, output_layer] ---
    if embedding is not None and isinstance(embedding, HSDPModule):
        targets = _layer_inner_hsdp_modules(transformer_layers[0])
        targets.append(transformer_layers[0])
        embedding.set_modules_to_forward_prefetch(targets)

    next_layers = transformer_layers[1:] + [None]
    for layer, next_layer in zip(transformer_layers, next_layers):
        if not isinstance(layer, HSDPModule):
            continue
        targets = _layer_inner_hsdp_modules(next_layer)
        if next_layer is not None:
            targets.append(next_layer)
        elif tail_modules is not None:
            targets.extend(tail_modules)
        if targets:
            layer.set_modules_to_forward_prefetch(targets)

    # --- Backward prefetch: output_layer -> layer[N-1] -> ... -> embedding ---
    reversed_layers = list(reversed(transformer_layers))
    prev_layers = reversed_layers[1:] + [None]

    if tail_modules and isinstance(tail_modules[-1], HSDPModule):
        tail_modules[-1].set_modules_to_backward_prefetch([reversed_layers[0]])

    for layer, prev_layer in zip(reversed_layers, prev_layers):
        if not isinstance(layer, HSDPModule):
            continue
        targets = _layer_inner_hsdp_modules(prev_layer)
        if prev_layer is not None:
            targets.append(prev_layer)
        elif embedding is not None and isinstance(embedding, HSDPModule):
            targets.append(embedding)
        if targets:
            layer.set_modules_to_backward_prefetch(targets)


def _distribute_param(module, param_name, device_mesh, placements):
    """
    Distribute a parameter of a module across a device mesh.

    Args:
        module: The module containing the parameter.
        param_name (str): Name of the parameter to distribute.
        device_mesh: The target device mesh for distribution.
        placements: Placement strategy for the distributed tensor.

    Raises:
        ValueError: If the module does not have the specified parameter.
    """
    # Ensure the parameter exists in the module
    if not hasattr(module, param_name):
        raise ValueError(f"Parameter '{param_name}' not found in module.")

    # Retrieve the original parameter
    param = getattr(module, param_name)

    # Replace it with a distributed version wrapped as a Parameter
    setattr(
        module,
        param_name,
        Parameter(
            distribute_tensor(
                param,
                device_mesh=device_mesh,
                placements=placements
            ),
            name=param.name,
            requires_grad=param.requires_grad
        )
    )


def _collect_layer_norms(layer):
    """Collect independent norm submodules from a transformer layer for separate FSDP wrapping."""
    norms = {}
    for attr_name in ("input_layernorm", "pre_mlp_layernorm", "pre_cross_attn_layernorm"):
        norm = getattr(layer, attr_name, None)
        if is_norm_module(norm):
            norms[attr_name] = norm
    for attr_name in ("q_layernorm", "k_layernorm"):
        norm = getattr(layer.self_attention, attr_name, None)
        if is_norm_module(norm):
            norms[attr_name] = norm
    return norms


def _collect_module_replicate_params(module, shard_size):
    """Collect parameters from a module that cannot be evenly FSDP-sharded on dim 0.

    This is the generic version of :func:`_collect_layer_replicate_params` for modules
    that are not standard transformer decoder layers (e.g. embedding, output_layer).
    It checks only the shape divisibility criterion; it does NOT inspect
    layer-specific sub-module attributes.
    """
    replicate_params = []
    for param_name, param in module.parameters_and_names():
        if isinstance(param, DTensor):
            shape = param.local_shape
        else:
            shape = param.shape
        if shape[0] % shard_size != 0:
            replicate_params.append(param)
            logger.warning("The shape[0]=%s of parameter %s is not divisible by "
                           "data_parallel_shard or dense_fsdp_shard_size which is %s, "
                           "then this parameter will not be applied fsdp.", shape[0], param_name, shard_size)
    return replicate_params


def _collect_layer_replicate_params(layer, shard_size):
    """Collect non-module parameters that must be replicated (not sharded) under FSDP."""
    replicate_params = []
    if hasattr(layer.self_attention.core_attention, "max_logits_val"):
        replicate_params.append(layer.self_attention.core_attention.max_logits_val)

    # DSv4 q-head RMS gamma is a fp32 non-trainable buffer. Replicate it so it is
    # kept out of the layer's (bf16) sharded flat param-buffer, which is allocated
    # with a single dtype and would otherwise mis-cast this fp32 tensor.
    if hasattr(layer.self_attention, "q_rms_gamma"):
        replicate_params.append(layer.self_attention.q_rms_gamma)

    if hasattr(layer.mlp, "tokens_per_expert"):
        replicate_params.append(layer.mlp.tokens_per_expert)

    if getattr(layer.mlp, "enable_expert_bias", False):
        replicate_params.append(layer.mlp.expert_bias)

    if hasattr(layer.mlp, "router"):
        if hasattr(layer.mlp.router, "tid2eid") and layer.mlp.router.tid2eid is not None:
            replicate_params.append(layer.mlp.router.tid2eid)

    # Also apply the generic shape[0] divisibility check from _collect_module_replicate_params.
    replicate_params.extend(_collect_module_replicate_params(layer, shard_size))

    return replicate_params


def _direct_parameters(module):
    """Return parameters owned directly by ``module``, excluding child cells."""
    return [param for _, param in module.parameters_and_names(expand=False)]


def _wrap_fp32_linear(linear, fsdp_config, reshard_after_forward, shard_size):
    """FSDP-wrap one forced-FP32 Linear independently from its BF16 owner.

    mHC projection weights are sharded on their input dimension. Any direct
    parameter that cannot be evenly sharded there (including a future bias) is
    kept replicated while still participating in gradient synchronization.
    """
    weight = getattr(linear, "weight", None)
    if weight is None:
        return

    replicate_params = []
    for param in _direct_parameters(linear):
        if param is not weight:
            replicate_params.append(param)
            continue
        shape = param.local_shape if isinstance(param, DTensor) else param.shape
        if shape[1] % shard_size != 0:
            replicate_params.append(param)

    with ms.DeviceCtx("meta"):
        fully_shard(
            linear,
            **fsdp_config,
            shard_placement_fn=lambda _: Shard(1),
            reshard_after_forward=reshard_after_forward,
            replicate_params=replicate_params,
        )


def _wrap_fp32_linear_owner(
        owner, linear_attr, fsdp_config, reshard_after_forward, shard_size, *, wrap_linear=True):
    """Wrap a forced-FP32 Linear and its directly-owned FP32 parameters.

    The Linear is wrapped first so its weight forms an independent sharded
    FSDP unit. The callable owner is then wrapped for the remaining scalar and
    vector parameters, which stay replicated. Fused mHC bypasses
    ``Linear.construct`` and therefore sets ``wrap_linear=False`` so the owner
    hook manages the projection weight as well.
    """
    linear = getattr(owner, linear_attr, None)
    if linear is None:
        return

    direct_params = _direct_parameters(owner)
    if wrap_linear:
        _wrap_fp32_linear(linear, fsdp_config, reshard_after_forward, shard_size)

    def shard_plan(param):
        if param is getattr(linear, "weight", None) and len(param.shape) > 1:
            return Shard(1)
        return Shard(0)

    with ms.DeviceCtx("meta"):
        fully_shard(
            owner,
            **fsdp_config,
            shard_placement_fn=shard_plan,
            reshard_after_forward=reshard_after_forward,
            replicate_params=direct_params,
        )


def _wrap_hc_modules(host, fsdp_config, reshard_after_forward, shard_size):
    """Independently FSDP-wrap a host's mHC sub-modules (attn_hc/ffn_hc).

    Each FP32 projection gets a dedicated FSDP group, while the owner's scalar
    and vector parameters stay replicated. Shared by the main decoder layers
    and the MTP transformer layers so both get the same treatment.
    """
    for hc_attr in ("attn_hc", "ffn_hc"):
        hc_module = getattr(host, hc_attr, None)
        if hc_module is None:
            continue
        _wrap_fp32_linear_owner(
            hc_module,
            "mapping_proj",
            fsdp_config,
            reshard_after_forward,
            shard_size,
            wrap_linear=not isinstance(hc_module, FusedHyperConnectionModule),
        )


def _wrap_hc_head(host, fsdp_config, reshard_after_forward, shard_size):
    """Independently wrap the head FP32 Linear and replicated gate parameters."""
    hc_head = getattr(host, "hc_head", None)
    if hc_head is None:
        return None
    _wrap_fp32_linear_owner(
        hc_head, "hc_fn", fsdp_config, reshard_after_forward, shard_size
    )
    return hc_head


def _wrap_dsv4_attention_modules(self_attention, fsdp_config, reshard_after_forward, replicate_params):
    """Independently FSDP-wrap the fp32 tensors of a DSv4 hybrid attention.

    DSv4 attention mixes bf16 projection weights with fp32 tensors that ``fully_shard``
    cannot place in the same HSDP param-group (it asserts a single original dtype per
    group and allocates one flat buffer per group):

    * the fp32 RMSNorms ``kv_layernorm`` and the compressor ``norm`` cells nested under
      ``core_attention`` / ``core_attention.indexer`` (``q_layernorm`` is already wrapped
      by the generic norm pass in :func:`_collect_layer_norms`, so it is skipped here);
    * the fp32 trainable per-head ``attn_sink`` Parameter on ``core_attention``.

    Strategy: wrap each fp32 norm cell (reached via ``construct``, so its FSDP all-gather
    hook fires normally) into its own group, then wrap ``core_attention`` with every
    parameter *except* ``attn_sink`` in ``ignored_params``. That leaves ``core_attention``'s
    own group holding only the fp32 ``attn_sink``, while the bf16 ``compressor`` /
    ``indexer`` weights fall through to the enclosing transformer-layer wrap. Those bf16
    weights MUST stay with the layer wrap: the indexer is invoked through a custom method
    (``indexer.forward_before_topk``) rather than ``construct``, so a dedicated FSDP wrap
    around it would never fire its pre-forward all-gather and its weights would still be
    sharded DTensors at matmul time. The layer wrap all-gathers them before the layer
    forward, which is when the indexer actually runs. No-op for the non-DSv4 path.
    """
    if not isinstance(self_attention, DSv4HybridSelfAttention):
        return

    q_layernorm = getattr(self_attention, "q_layernorm", None)
    # 1. fp32 norm cells anywhere under the attention (skip the already-wrapped q_layernorm).
    #    These are reached via ``construct`` so their FSDP all-gather hook fires normally.
    norm_modules = [
        module for _, module in self_attention.cells_and_names()
        if module is not q_layernorm and is_norm_module(module)
    ]
    for norm in norm_modules:
        with ms.DeviceCtx("meta"):
            fully_shard(norm, **fsdp_config, replicate_params=replicate_params)

    # 2. Isolate the fp32 ``attn_sink`` into ``core_attention``'s own group; everything else
    #    under core_attention (bf16 compressor / indexer weights) is ignored here so it falls
    #    through to the transformer-layer wrap that all-gathers it before the layer forward.
    core_attention = getattr(self_attention, "core_attention", None)
    attn_sink = getattr(core_attention, "attn_sink", None) if core_attention is not None else None
    if core_attention is None or attn_sink is None:
        return
    ignored = [param for _, param in core_attention.parameters_and_names() if param is not attn_sink]
    with ms.DeviceCtx("meta"):
        fully_shard(core_attention, **fsdp_config, reshard_after_forward=reshard_after_forward,
                    ignored_params=ignored, replicate_params=replicate_params)


# LoRA adapter layouts keyed by the target module's base TP role. Only the non-default
# placement is listed: distribute_module replicates every other (trainable) param, so a
# colwise base needs only lora_b sharded on the out-dim, a rowwise base only lora_a on the
# in-dim, and a NoParallel base only its frozen weight pinned to Replicate (else, being
# frozen and absent from any plan, it would stay a plain Tensor). Centralising this here
# keeps the generic ParallelStyle LoRA-agnostic.
_LORA_ROLE_LAYOUTS = {
    "self_attention.linear_qkv": {"weight": (Replicate(),)},
    "self_attention.linear_kvb": {"lora_b": (Shard(0),)},
    "self_attention.linear_qb": {"lora_b": (Shard(0),)},
    "self_attention.linear_proj": {"lora_a": (Shard(1),)},
    "mlp.linear_fc1": {"lora_b": (Shard(0),)},
    "mlp.linear_fc2": {"lora_a": (Shard(1),)},
    "mlp.shared_experts.linear_fc1": {"lora_b": (Shard(0),)},
    "mlp.shared_experts.linear_fc2": {"lora_a": (Shard(1),)},
}


def _attach_lora_layouts(transformer_layer, layer_plan):
    """Attach LoRA adapter layouts to each target module's ParallelStyle (model-side).

    Sets a generic ``extra_param_layouts`` on the style so the style itself stays
    LoRA-agnostic; the style merges it into its sharding plan only for params the module
    actually has. No-op for a module without an adapter (non-LoRA run).
    """
    for name, extra in _LORA_ROLE_LAYOUTS.items():
        style = layer_plan.get(name)
        sub = transformer_layer
        for part in name.split("."):
            sub = getattr(sub, part, None)
        if style is not None and getattr(sub, "lora_a", None) is not None:
            style.extra_param_layouts = extra


def _get_dsv4_tp_shard_plan(transformer_layer, shard_plans, distribute_param_plan):
    """get dsv4 tp plan"""
    # Fused CSA kernel has CANN shape constraints and no TP benefit from head-sharding
    # non-fused small-op path does benefit.
    sp_layout, norm_plan, _, _ = shard_plans
    rope_freqs_layout = Replicate() if transformer_layer.self_attention.config.apply_rope_fusion else None
    core_attn = transformer_layer.self_attention.core_attention
    csa_q_layout = Replicate() # currently csa doesn't apply tp
    attn_sink_plan = Replicate()
    dsv4_core_attn_plan = PrepareModuleInputOutput(
        # q, key, x, q_compressed
        input_layouts=(Replicate(), Replicate(), Replicate(), Replicate()),
        desired_input_layouts=(csa_q_layout, Replicate(), Replicate(), Replicate()),
        output_layouts=(csa_q_layout,),
        desired_output_layouts=(Replicate(),),
        use_local_input=True,
        use_local_output=False,
    )
    unfused_indexer_loss_plan = PrepareModuleInputOutput(
        input_layouts=(Replicate(),),
        desired_input_layouts=(Replicate(),),
        output_layouts=(Replicate(),),
        desired_output_layouts=(Replicate(),),
        use_local_output=False,
    )
    layer_plan = {
        "input_layernorm": norm_plan,
        # Input: SP Shard(0) → Replicate (attention internals)
        # Output: Replicate → SP Shard(0) (so output_cell h_post/x dims match)
        "self_attention": PrepareModuleInputOutput(
            input_layouts=(sp_layout,),
            desired_input_layouts=(Replicate(),),
            output_layouts=(Replicate(),),
            desired_output_layouts=(sp_layout,),
            use_local_output=False,
        ),
        # --- Q path ---
        "self_attention.linear_q_down_proj": NoParallel(use_local_output=False),
        "self_attention.q_layernorm": NoParallel(use_local_output=False),
        "self_attention.linear_q_up_proj": ColwiseParallel(
            output_layouts=Replicate(), use_local_output=False,
        ),
        # --- KV path (single shared head) ---
        "self_attention.linear_kv_proj": NoParallel(use_local_output=False),
        "self_attention.kv_layernorm": NoParallel(use_local_output=False),
        # --- Core attention ---
        "self_attention.core_attention": dsv4_core_attn_plan,
        # --- Output projection ---
        "self_attention.linear_proj": NoParallel(use_local_output=False),
        "pre_mlp_layernorm": norm_plan,
        "self_attention.apply_rotary_emb": PrepareModuleInputOutput(
            input_layouts=(Replicate(), rope_freqs_layout, None),
            desired_input_layouts=(Replicate(), rope_freqs_layout, None),
            output_layouts=(Replicate(),),
            desired_output_layouts=(Replicate(),),
            use_local_output=False,
        ),
    }
    if getattr(core_attn, "unfused_indexer_loss", None) is not None:
        layer_plan["self_attention.core_attention.unfused_indexer_loss.post_head_sum"] = unfused_indexer_loss_plan
    distribute_param_plan.append([transformer_layer.self_attention.core_attention, "attn_sink", (attn_sink_plan,)])
    distribute_param_plan.append([transformer_layer.self_attention, "linear_o_group_proj", (Replicate(),)])
    distribute_param_plan.append([transformer_layer.self_attention, "q_rms_gamma", (Replicate(),)])
    return layer_plan, distribute_param_plan


def _configure_local_fa(core_attention, world):
    """Rebuild FlashAttentionScore with the local head count (n/t) for the TP flow.

    The DTensor flow fed FA a head-sharded DTensor whose logical head dim is the
    global ``num_attention_heads``; the local flow feeds a plain tensor holding only
    this rank's ``n/t`` heads, so ``head_num`` must be the local count.
    """
    if not isinstance(core_attention, FlashAttention) or world == 1:
        return
    if core_attention.head_num % world != 0:
        raise ValueError(
            f"num_attention_heads ({core_attention.head_num}) must be divisible by the "
            f"TP degree ({world}) for the local FlashAttention flow.")
    core_attention.head_num = core_attention.head_num // world
    core_attention.flash_attention = FlashAttentionScore(
        head_num=core_attention.head_num,
        scale_value=core_attention.scalar_value,
        pre_tokens=core_attention.pre_tokens,
        next_tokens=core_attention.next_tokens,
        inner_precise=core_attention.inner_precise,
        input_layout=core_attention.input_layout,
        sparse_mode=core_attention.sparse_mode,
    )


def _mla_attention_layer_plan(self_attn, tp_mesh, enable_mc2=False):
    """Run MLA compression on the sequence shard and gather only projected Q/K/V."""
    _configure_local_fa(self_attn.core_attention, tp_mesh.size())
    colwise = MC2ColwiseParallel if enable_mc2 else ColwiseParallel
    rowwise = MC2RowwiseParallel() if enable_mc2 else RowwiseParallel()
    plan = {
        "self_attention.linear_qkv": SequenceParallel(sequence_dim=0),
        "self_attention.k_layernorm": SequenceParallel(sequence_dim=0),
        "self_attention.linear_kvb": colwise(gather_input=True),
        "self_attention.linear_proj": rowwise,
        "self_attention.apply_rotary_emb_k": PrepareModuleInput(
            input_transforms=(AllGather(0), None),
        ),
    }
    if self_attn.config.q_lora_rank is not None:
        plan["self_attention.linear_qb"] = colwise(gather_input=True)
        plan["self_attention.q_layernorm"] = SequenceParallel(sequence_dim=0)
    return plan


def _gqa_attention_layer_plan(self_attn, tp_mesh, enable_mc2=False):
    """Layer-plan entries for standard GQA / MHA self-attention (Qwen3, GLM4, TeleChat3).

    The fused QKV projection (``linear_qkv``) is colwise: its ``Shard(0)`` split follows
    the group-interleaved weight layout ``[g0_q..q k v | g1_q..q k v | ...]``, so each TP
    rank owns whole query-groups (requires ``tp`` to divide ``num_query_groups``). Under
    SP its pre-hook all-gathers the sequence-sharded input before the local matmul. The
    o-projection (``linear_proj``) is rowwise (reduce-scatter back to the shard). The
    optional per-head q/k RMSNorm (Qwen3) run local on the head-sharded activations and
    replicate their ``head_dim`` weight, whose gradient reduces over TP like the other
    norms (SequenceParallel).
    """
    _configure_local_fa(self_attn.core_attention, tp_mesh.size())
    colwise = MC2ColwiseParallel() if enable_mc2 else ColwiseParallel(gather_input=True)
    rowwise = MC2RowwiseParallel() if enable_mc2 else RowwiseParallel()
    plan = {
        "self_attention.linear_qkv": colwise,
        "self_attention.linear_proj": rowwise,
    }
    if getattr(self_attn, "q_layernorm", None) is not None:
        plan["self_attention.q_layernorm"] = SequenceParallel(sequence_dim=0)
    if getattr(self_attn, "k_layernorm", None) is not None:
        plan["self_attention.k_layernorm"] = SequenceParallel(sequence_dim=0)
    return plan


class _DSv4QUpParallel(ParallelStyle):
    """Column-shard Q-up while reducing its replicated input gradient."""

    def _apply(self, module, device_mesh):
        # Shard -> AllGather is an identity in forward.  Its backward is
        # AllGather -> ReduceScatter, which sums the Q-up column-shard partials.
        module = PrepareModuleInput(input_transforms=ShardTensor(0))._apply(
            module, device_mesh
        )
        return ColwiseParallel(gather_input=True, gather_output=True)._apply(
            module, device_mesh
        )


def _dsv4_attention_layer_plan(self_attn, tp_mesh):
    """Build the Local-TP plan for DSV4 hybrid fused attention.

    Fused CSA's indexer KL target aggregates attention probabilities over all
    query heads before normalisation, so head-local CSA calls are not
    mathematically decomposable.  Keep CSA/indexer/output projection replicated
    and use TP only for Q-up, gathering its head shards before CSA.  The whole
    attention receives a gathered sequence and returns the local sequence
    shard, preserving sequence parallelism without changing the fused loss.
    """
    world = tp_mesh.size()
    if self_attn.query_projection_size % world != 0:
        raise ValueError(
            f"query_projection_size ({self_attn.query_projection_size}) must be "
            f"divisible by TP degree ({world}) for DSV4 Q-up projection."
        )

    plan = {
        "self_attention": PrepareModuleInputOutput(
            # Output-shard backward reconstructs the same complete CSA output
            # gradient on every TP rank.  Therefore input backward must select
            # the local sequence gradient, not reduce identical copies again.
            input_transforms=AllGather(0, reduce_grad=False),
            output_transforms=ShardTensor(0),
        ),
        "self_attention.linear_q_up_proj": _DSv4QUpParallel(),
    }

    core_attention = self_attn.core_attention
    indexer = getattr(core_attention, "indexer", None)
    if indexer is not None:
        plan["self_attention.core_attention.indexer"] = SequenceParallel(
            sequence_dim=0
        )

    # Full CSA runs on every TP rank, so all of its parameters except Q-up and
    # the indexer hold identical already-global gradients.  Keep the count on
    # the cell for propagation after FSDP replaces the Parameter objects.
    setattr(self_attn, "_tp_full_attention_replica_count", world)

    distribute_param_plan = []
    return plan, distribute_param_plan


def _tag_dsv4_tp_replicated_grad_norm_params(model):
    """Tag replicated DSV4 attention parameters for global grad-norm de-duplication."""
    for _, cell in model.cells_and_names():
        replica_count = getattr(cell, "_tp_full_attention_replica_count", 1)
        if replica_count <= 1:
            continue
        for param_name, param in cell.parameters_and_names():
            if param_name.startswith("linear_q_up_proj."):
                continue
            if param_name.startswith("core_attention.indexer."):
                continue
            setattr(param, "_grad_norm_replica_count", replica_count)


def _dense_mlp_layer_plan(enable_mc2=False):
    """Return the complete local TP plan for a dense MLP."""
    colwise = MC2ColwiseParallel() if enable_mc2 else ColwiseParallel(gather_input=True)
    rowwise = MC2RowwiseParallel() if enable_mc2 else RowwiseParallel()
    return {
        "mlp.linear_fc1": colwise,
        "mlp.linear_fc2": rowwise,
    }


def _apply_layers_tp(
    transformer_layer,
    tp_mesh,
    enable_ep,
    enable_mc2=False,
):
    """Apply local (to_local) tensor + sequence parallelism to one GPT transformer layer.

    The whole layer runs on plain, sequence-sharded local tensors; the tensor-parallel
    collectives are hand-written in the style forward hooks (colwise pre-hook all-gather,
    rowwise post-hook reduce-scatter) plus the MLA ``k_pe`` all-gather inside the module.
    Weights stay sharded DTensors (FSDP / checkpoint unchanged), localised at compute.
    """
    world = tp_mesh.size()
    self_attn = transformer_layer.self_attention
    is_dsv4 = isinstance(self_attn, DSv4HybridSelfAttention)
    is_mla = isinstance(self_attn, MLASelfAttention)
    if is_mla and self_attn.config.q_lora_rank is None:
        # Without q compression, q comes straight out of linear_qkv (sequence-sharded, not
        # head-sharded) with no colwise up-projection to all-gather it, so SP-native MLA
        # cannot form the full-sequence query.
        raise NotImplementedError("Local sequence-parallel MLA requires q_lora_rank (q compression).")
    if not is_mla and not is_dsv4:
        # Standard GQA/MHA: the fused QKV is colwise-sharded along the group-interleaved
        # output dim, so each rank must own whole query-groups.
        num_kv_groups = self_attn.kv_num_heads
        if num_kv_groups % world != 0:
            raise ValueError(
                f"num_query_groups ({num_kv_groups}) must be divisible by the TP degree "
                f"({world}) for the local GQA flow (fused-QKV colwise split)."
            )

    # Block-level norms run local on the sequence shard (Replicate weights, no collective);
    # the attention-internal projections/norms are dispatched by attention type (MLA vs GQA).
    layer_plan = {
        "input_layernorm": SequenceParallel(sequence_dim=0),
        "pre_mlp_layernorm": SequenceParallel(sequence_dim=0),
    }
    if getattr(transformer_layer, "attn_hc", None) is not None:
        layer_plan.update({
            "attn_hc": SequenceParallel(sequence_dim=0),
            "ffn_hc": SequenceParallel(sequence_dim=0),
        })
    distribute_param_plan = []
    if is_dsv4:
        dsv4_plan, dsv4_params = _dsv4_attention_layer_plan(self_attn, tp_mesh)
        layer_plan.update(dsv4_plan)
        distribute_param_plan.extend(dsv4_params)
    elif is_mla:
        layer_plan.update(_mla_attention_layer_plan(self_attn, tp_mesh, enable_mc2))
    else:
        layer_plan.update(_gqa_attention_layer_plan(self_attn, tp_mesh, enable_mc2))

    # Dense MLP.
    if isinstance(transformer_layer.mlp, MLP):
        layer_plan.update(_dense_mlp_layer_plan(enable_mc2))

    # MoE: the router runs on local tokens (gate weight to_local'd in its construct).
    if isinstance(transformer_layer.mlp, MoELayer):
        router = transformer_layer.mlp.router
        # Hidden states are sequence-sharded under Local TP.  Hash routing must
        # consume the matching token-id shard instead of the full input_ids,
        # otherwise its [tokens, top_k] expert map no longer matches the local
        # router logits.
        layer_plan["mlp"] = PrepareModuleInput(
            input_kwarg_transforms={"input_ids": ShardTensor(1)}
        )
        if getattr(router, "aux_loss_type", None) == "global_aux_loss":
            raise NotImplementedError("Local TP flow does not support global_aux_loss routing yet.")
        # Match the baseline parameter placements even for non-trainable MoE state.
        distribute_param_plan.append([router, "weight", (Replicate(),)])
        distribute_param_plan.append([transformer_layer.mlp, "tokens_per_expert", (Replicate(),)])
        if getattr(transformer_layer.mlp, "enable_expert_bias", False):
            distribute_param_plan.append([transformer_layer.mlp, "expert_bias", (Replicate(),)])
        if not enable_ep:
            distribute_param_plan.extend([
                [transformer_layer.mlp.experts, "weight1", (Replicate(),)],
                [transformer_layer.mlp.experts, "weight2", (Replicate(),)],
            ])

    # Routed and shared experts both keep the SP token shard. Shared-expert
    # parameters are replicated over TP, so its fc1/fc2 and optional gate compute
    # locally without sequence collectives.
    shared_experts = getattr(transformer_layer.mlp, "shared_experts", None)
    if shared_experts is not None:
        layer_plan["mlp.shared_experts"] = SequenceParallel(sequence_dim=0)

    core_attention = self_attn.core_attention
    if hasattr(core_attention, "max_logits_val"):
        distribute_param_plan.append([core_attention, "max_logits_val", (Shard(0),)])

    for sub_module, param_name, sub_plan in distribute_param_plan:
        _distribute_param(sub_module, param_name=param_name, device_mesh=tp_mesh, placements=sub_plan)

    _attach_lora_layouts(transformer_layer, layer_plan)
    parallelize_module(module=transformer_layer, device_mesh=tp_mesh, parallelize_plan=layer_plan)

def apply_non_moe_tp(
        model: nn.Cell,
        tp_mesh: DeviceMesh,
        enable_ep: bool = False,
        enable_mc2: bool = False,
):
    """Apply local (to_local) tensor + sequence parallelism to the whole GPTModel forward.

    Every forward module runs on plain, sequence-sharded local tensors; the tensor-parallel
    collectives are hand-written in forward hooks. Weights stay sharded DTensors (FSDP /
    checkpoint unchanged) and are localised at compute -- DTensor never appears in the forward.
    """
    world = tp_mesh.size()
    rank = tp_mesh.get_local_rank()
    group = tp_mesh.get_group()

    # 1. Embedding: vocab-parallel local gather + reduce onto the sequence shard.
    if hasattr(model.model, "embedding"):
        parallelize_module(
            model.model.embedding,
            tp_mesh,
            {
                "word_embeddings": RowwiseParallel()
            },
        )

    # 2. Final norm: local RMSNorm on the sequence shard (Replicate weight, no collective).
    if getattr(model.model.decoder, "final_layernorm", None):
        SequenceParallel(sequence_dim=0)._apply(model.model.decoder.final_layernorm, tp_mesh)

    # 3. Output layer (LM head): colwise; the output stays vocab-sharded local for the loss.
    if hasattr(model.model, "output_layer"):
        ColwiseParallel(gather_input=True)._apply(model.model.output_layer, tp_mesh)

    # 4. Loss: vocab-parallel CE on the local vocab slice (pass the TP metadata).
    loss = getattr(model.model, "loss", None)
    if loss is not None and world > 1:
        loss.enable_vocab_parallel(group, rank, world)

    # 5. Decoder layers.
    for transformer_layer in model.model.decoder.layers:
        _apply_layers_tp(
            transformer_layer, tp_mesh, enable_ep, enable_mc2=enable_mc2)

    # 6. MTP outer modules and inner transformer layers. Match graph mode for eh_proj:
    # keep its parameters replicated and compute directly on the local sequence shard.
    # The inner MLA+MoE layer uses the same local plan as the decoder layers.
    mtp = getattr(model.model, "mtp", None)
    if mtp is not None:
        for layer in mtp.layers:
            parallelize_module(
                layer,
                tp_mesh,
                {
                    "enorm": SequenceParallel(sequence_dim=0),
                    "hnorm": SequenceParallel(sequence_dim=0),
                    "eh_proj": SequenceParallel(sequence_dim=0),
                    "final_layernorm": SequenceParallel(sequence_dim=0),
                },
            )
            _apply_layers_tp(
                layer.transformer_layer, tp_mesh, enable_ep, enable_mc2=enable_mc2)

    logger.info(
        "Applied local (to_local) Tensor + Sequence Parallelism to the model "
        "(MC2 fusion %s).", "ENABLED" if enable_mc2 else "disabled")


def apply_moe_ep_tp(
        model: nn.Cell,
        tp_mesh: DeviceMesh = None,
        ep_mesh: DeviceMesh = None,
        moe_token_dispatcher_type: str = "all_to_all",
        npu_nums_per_device: int = 8,
        expert_async_d2h: bool = False,
):
    """
    Apply Expert Parallelism (EP) and Tensor Parallelism (TP) to MoE layers in model.

 	Args:
        model (nn.Cell): The model to be parallelized.
        tp_mesh (DeviceMesh): The device mesh for tensor parallelism.
        ep_mesh (DeviceMesh): The device mesh for expert parallelism.
        moe_token_dispatcher_type (str): The type of token dispatcher for MoE.
        npu_nums_per_device (int): The number of NPUs per device.

 	MoE layer structure:
        - mlp.experts.weight1: Expert FFN first layer weights
        - mlp.experts.weight2: Expert FFN second layer weights
        - mlp.router.weight: Router/gate weights
        - mlp.shared_experts.linear_fc1.weight: Shared expert FFN first layer
        - mlp.shared_experts.linear_fc2.weight: Shared expert FFN second layer
        - mlp.tokens_per_expert: Token distribution tracking

    Returns:
        nn.Cell: The parallelized model.
    """

    if tp_mesh is None and ep_mesh is None:
        raise ValueError("At least one of ep_mesh or tp_mesh must be provided.")

    transformer_layers = list(model.model.decoder.layers)
    if getattr(model.model, "mtp", None) is not None:
        transformer_layers.extend(
            mtp_layer.transformer_layer for mtp_layer in model.model.mtp.layers
        )

    for transformer_block in transformer_layers:
        if not hasattr(transformer_block.mlp, 'experts'):
            continue

        # ============ Apply EP to Experts ============
        # Determine which mesh and plan to use for experts
        experts_mesh = ep_mesh
        if ep_mesh is None:
            # No EP, use TP for experts (TensorParallel on expert weights)
            experts_mesh = tp_mesh

        set_comm_ops_inplace(False)
        # Apply TP to experts: shard along the hidden dimension
        if moe_token_dispatcher_type == "alltoall":
            experts_plan = ExpertParallel(model.model.config.moe_permute_fusion,
                                          async_d2h=expert_async_d2h)
        elif moe_token_dispatcher_type == "alltoall_deredundancy":
            experts_plan = DeredundancyExpertParallel(npu_nums_per_device)
        else:
            raise ValueError(
                f"Unsupported moe_token_dispatcher_type: '{moe_token_dispatcher_type}'. "
                "Expected 'alltoall' or 'alltoall_deredundancy'."
            )
        if hasattr(transformer_block.mlp, 'experts') and experts_mesh is not None:
            parallelize_module(
                module=transformer_block.mlp.experts,
                device_mesh=experts_mesh,
                parallelize_plan=experts_plan,
            )

    return model


def apply_moe_ep_overlap_tp(
        model,
        overlap,
        tp_mesh=None,
        ep_mesh=None,
        moe_token_dispatcher_type: str = "alltoall",
        expert_async_d2h: bool = False,
):
    """Apply EP with comm/compute overlap hooks to MoE layers in all pipeline model parts.

    Wraps every MoE layer's ``experts`` module with :class:`OverlapExpertParallel`
    (instead of plain :class:`ExpertParallel`) and tags the **last MoE layer in
    each pipeline stage** with ``is_last_layer=True`` so its closing D hook is
    ``"D_LAST"``.  The CHUNK_START / CHUNK_END bracketing happens at the
    OVERLAP_B_F callback level (:func:`_make_overlap_b_f_callback`), covering
    the whole ``forward_one_chunk`` record including embedding and lm-head/loss.

    This function is called once per call to :func:`apply_pp` when
    ``parallelism.pipeline_parallel_overlap_b_f`` is ``True`` and
    ``moe_token_dispatcher_type`` is ``"alltoall"``.

    Args:
        model:     Single pipeline-stage model (``nn.Cell``).
        overlap:   Shared
                   :class:`~hyper_parallel.core.pipeline_parallel.comm_compute_overlap.CommComputeOverlap`
                   instance for this rank (one per rank, shared across all chunks).
        tp_mesh:   TP device mesh (passed through to EP mesh selection).
        ep_mesh:   EP device mesh.
        moe_token_dispatcher_type: Must be ``"alltoall"``; overlap is only
                   supported for the standard all-to-all dispatcher.

    Returns:
        ``model`` (mutated in place).
    """
    if moe_token_dispatcher_type != "alltoall":
        raise ValueError(
            f"EP comm/compute overlap only supports moe_token_dispatcher_type='alltoall', "
            f"got '{moe_token_dispatcher_type}'."
        )

    if ep_mesh is None and tp_mesh is None:
        raise ValueError("At least one of ep_mesh or tp_mesh must be provided.")

    experts_mesh = ep_mesh if ep_mesh is not None else tp_mesh
    coordinator = overlap.coordinator

    transformer_layers = list(model.model.decoder.layers)
    if getattr(model.model, "mtp", None) is not None:
        transformer_layers.extend(
            mtp_layer.transformer_layer for mtp_layer in model.model.mtp.layers
        )

    # Collect MoE layers to apply overlap strategy
    moe_layers = [
        layer for layer in transformer_layers
        if hasattr(layer.mlp, "experts")
    ]
    last_idx = len(moe_layers) - 1

    # Expert weights live on the meta device during model build; sharding them
    # into DTensors must happen under the meta DeviceCtx (same as the baseline
    # apply_moe_ep_tp path). The CHUNK hook registration below is Python-level
    # and must stay OUTSIDE the meta context.
    with ms.DeviceCtx("meta"):
        for idx, layer in enumerate(moe_layers):
            is_last = idx == last_idx
            strategy = OverlapExpertParallel(
                coordinator=coordinator,
                is_last_layer=is_last,
                # Honor the yaml fusion flag (same as the non-overlap apply_moe_ep_tp
                # path). Verified bit-identical to manual permute under overlap on
                # pp2/ep2 + recompute:full + MTP; other overlap variants (interleaved
                # pp, no-recompute, larger ep) not yet A/B'd.
                moe_permute_fusion=model.model.config.moe_permute_fusion,
                async_d2h=expert_async_d2h,
            )
            parallelize_module(
                module=layer.mlp.experts,
                device_mesh=experts_mesh,
                parallelize_plan=strategy,
            )

    # CHUNK_START / CHUNK_END bracketing is done at the OVERLAP_B_F callback
    # level (see _make_overlap_b_f_callback) rather than via decoder Cell
    # hooks: the decoder boundary leaves preprocess/embedding (entry) and
    # lm-head/loss (exit) recording OUTSIDE the bracket, concurrent with the
    # BWD thread's loss.bwd/lm-head.bwd replay — MS PyNative does not support
    # concurrent FWD-record + BWD-replay and corrupts memory/grads.
    return model


# ---------------------------------------------------------------------------
# GPT FSDP implementation
# ---------------------------------------------------------------------------

def _resolve_dense_fsdp_config(parallelism, base_fsdp_config, parallel_dims):
    """Resolve the FSDP config + shard degree for dense (non-expert) weights.

    Reads ``parallelism.dense_fsdp_shard_size``. When it is unset (``None``) or equal to
    the full fsdp domain (``dp_shard * cp``), returns ``(base_fsdp_config, full_degree)``
    unchanged -- the default, pre-feature behavior. Otherwise returns a copy of
    ``base_fsdp_config`` whose ``mesh`` is the reduced ``[replicate, shard]`` sub-mesh
    (see :meth:`ParallelDims.get_fsdp_shard_mesh`), together with the reduced shard
    degree ``k``. The degree is fed to :func:`_collect_layer_replicate_params` so params
    whose leading dim is not divisible by ``k`` are replicated rather than sharded.

    Only dense weights consume the returned config; routed experts keep their own
    ``efsdp_config`` and are not affected.
    """
    full_degree = parallel_dims.fsdp
    shard_size = getattr(parallelism, "dense_fsdp_shard_size", None)
    if shard_size is None or shard_size == full_degree:
        return base_fsdp_config, full_degree
    dense_mesh = parallel_dims.get_fsdp_shard_mesh(shard_size)
    return {**base_fsdp_config, "mesh": dense_mesh}, shard_size


def apply_fsdp(
        model: nn.Cell,
        parallel_dims: Any,
        parallelism: Any,
) -> None:
    """Apply FSDP/HSDP to a GPT model.

    Directly operates on GPT model structure (embedding -> decoder.layers ->
    final_layernorm + output_layer -> root).

    Gradients are always accumulated and reduced in fp32: the HSDP mixed-precision
    policy reduces in fp32 (``reduce_dtype=float32``) and accumulates the reduced
    gradient onto a fp32 ``param.main_grad`` buffer
    (``apply_grad_on_fp32_main_grad=True``), aligning with Megatron-LM.

    Args:
        model: The GPTModel instance.
        parallel_dims: ParallelDims instance.
        parallelism: Parallelism configuration.
    """
    logger.info("Applying FSDP/HSDP to GPT model...")

    # Unwrap to GPTModel
    gpt_model = _unwrap_gptmodel(model)

    # --- Resolve dp_mesh ---
    # The ``fsdp`` axis already folds ``cp`` (fsdp = dp_shard * cp), so CP needs
    # no special-casing here: when dp_replicate is on we shard over the 2D
    # [dp_replicate, fsdp] HSDP mesh, otherwise over the 1D fsdp mesh. Folding CP
    # into a single ``cp_enabled`` branch would have dropped dp_replicate on
    # HSDP+CP runs (sharding only over dp_shard*cp instead of the full domain).
    if parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
        logger.info("Using HSDP mesh [dp_replicate, fsdp] (2D)")
    else:
        dp_mesh = parallel_dims.get_mesh("fsdp")
        logger.info("Using FSDP mesh [fsdp] (1D)")

    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_mesh(edp_mesh_names)

    # --- Build FSDP config ---
    reshard_policy = getattr(parallelism, "reshard_after_forward_policy", "default")
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_policy, parallel_dims.pp_enabled
    )

    cpu_offload = getattr(parallelism, "cpu_offload", False)
    # Always accumulate + reduce gradients in fp32: reduce-scatter / all-reduce run in
    # fp32 and the reduced gradient is accumulated onto a fp32 ``param.main_grad`` buffer
    # (Megatron-LM main_grad semantics), keeping ``param.grad`` at None.
    mp_policy = MixedPrecisionPolicy(
        reduce_dtype=ms.float32, apply_grad_on_fp32_main_grad=True
    )
    fsdp_config = {
        "mesh": dp_mesh,
        "offload_policy": CPUOffloadPolicy() if cpu_offload else OffloadPolicy(),
        "mp_policy": mp_policy,
    }
    efsdp_config = {
        "mesh": edp_mesh,
        "offload_policy": CPUOffloadPolicy() if cpu_offload else OffloadPolicy(),
        "mp_policy": mp_policy,
    }

    # --- Optionally limit the FSDP shard degree of dense (non-expert) weights ---
    # When ``dense_fsdp_shard_size`` is set, ``fsdp_config`` is rebound to a reduced
    # ``[replicate, shard]`` sub-mesh, so every dense wrap below (embedding, norms,
    # router, shared experts, layers, output layer, root) shards over that many ranks
    # and HSDP-replicates over the rest of the DP domain. ``efsdp_config`` (routed
    # experts) is built above and left untouched. The (world-collective) sub-mesh is
    # built here once, on every rank, before any wrapping, so process-group creation
    # stays consistent across pipeline stages.
    fsdp_config, dense_shard_degree = _resolve_dense_fsdp_config(
        parallelism, fsdp_config, parallel_dims
    )
    if dense_shard_degree != parallel_dims.fsdp:
        logger.info(
            "Dense-weight FSDP limited to shard_size=%d (full fsdp domain=%d); "
            "routed experts keep the full efsdp domain.",
            dense_shard_degree, parallel_dims.fsdp,
        )

    embedding = getattr(gpt_model, "embedding", None)
    layers = list(gpt_model.decoder.layers)

    mtp = getattr(gpt_model, "mtp", None)
    tail_modules = [
        m for m in [
            getattr(gpt_model.decoder, "final_layernorm", None),
            getattr(gpt_model, "output_layer", None) if not mtp else None,
        ] if m is not None
    ]

    if not layers:
        raise ValueError(f"{type(model).__name__} has no decoder layers")

    # --- 1. Wrap embedding ---
    if embedding is not None:
        embed_replicate_params = _collect_module_replicate_params(embedding, dense_shard_degree)
        with ms.DeviceCtx("meta"):
            fully_shard(
                embedding,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
                replicate_params=embed_replicate_params,
            )

    # --- 2. Wrap transformer layers ---
    # Principle: small modules first, then larger modules
    for layer in layers:
        # 2a. Expert FSDP (small expert block)
        if hasattr(layer.mlp, "experts") and edp_mesh is not None:
            with ms.DeviceCtx("meta"):
                fully_shard(layer.mlp.experts, **efsdp_config, reshard_after_forward=reshard_after_forward)

        replicate_params = _collect_layer_replicate_params(layer, dense_shard_degree)

        # 2a'. Wrap MoE router (and shared-expert gate) independently. Their weights are
        # stored in moe_router_dtype (fp32) for gradient precision, so they must NOT share
        # an HSDP param-group with the layer's bf16 params — fully_shard asserts a single
        # original dtype per group. Same treatment as the fp32 embedding wrapped above.
        router = getattr(layer.mlp, "router", None)
        if router is not None:
            with ms.DeviceCtx("meta"):
                fully_shard(
                    router,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                    replicate_params=replicate_params
                )

        shared_experts = getattr(layer.mlp, "shared_experts", None)
        shared_gate = getattr(shared_experts, "shared_experts_gate", None) if shared_experts is not None else None
        if shared_gate is not None:
            gate_replicate_params = _collect_module_replicate_params(shared_gate, dense_shard_degree)
            with ms.DeviceCtx("meta"):
                fully_shard(shared_gate, **fsdp_config, reshard_after_forward=reshard_after_forward,
                            replicate_params=gate_replicate_params)

        # 2b. Wrap norms independently (small modules first)
        layer_norms = _collect_layer_norms(layer)
        for norm in layer_norms.values():
            with ms.DeviceCtx("meta"):
                fully_shard(norm, **fsdp_config)

        # 2c. Wrap HC FP32 projections independently; replicate scalar/vector params.
        _wrap_hc_modules(layer, fsdp_config, reshard_after_forward, dense_shard_degree)

        # 2c'. Wrap DSv4 hybrid-attention fp32 sub-modules independently so their
        #      fp32 norms / attn_sink do not share a param-group with bf16 weights.
        _wrap_dsv4_attention_modules(
            layer.self_attention, fsdp_config, reshard_after_forward, replicate_params
        )

        # 2d. Wrap the transformer layer (large module)

        with ms.DeviceCtx("meta"):
            fully_shard(
                layer,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
                replicate_params=replicate_params
            )

    # --- 3. Wrap the FP32 mHC head, final_layernorm and output_layer independently ---
    _wrap_hc_head(
        gpt_model.decoder,
        fsdp_config,
        reshard_policy == "always",
        dense_shard_degree,
    )

    # Principle: small modules (norms) first, then larger modules (output_layer)
    if tail_modules:
        for tail_module in tail_modules:
            if is_norm_module(tail_module):
                # Small module: wrap norm independently
                with ms.DeviceCtx("meta"):
                    fully_shard(tail_module, **fsdp_config)
            else:
                # Larger module: output_layer, do not reshard_after_forward by default
                # since FSDP would prefetch it immediately after the forward pass.
                tail_replicate_params = _collect_module_replicate_params(
                    tail_module, dense_shard_degree)
                with ms.DeviceCtx("meta"):
                    fully_shard(
                        tail_module,
                        **fsdp_config,
                        reshard_after_forward=reshard_policy == "always",
                        replicate_params=tail_replicate_params,
                    )

    if mtp:
        for layer in mtp.layers:
            # Expert FSDP (same as main decoder layers)
            if hasattr(layer.transformer_layer.mlp, "experts") and edp_mesh is not None:
                with ms.DeviceCtx("meta"):
                    fully_shard(
                        layer.transformer_layer.mlp.experts, **efsdp_config,
                        reshard_after_forward=reshard_after_forward,
                    )

            mtp_replicate_params = _collect_layer_replicate_params(layer.transformer_layer, dense_shard_degree)

            # Router (fp32) — independent wrap to avoid dtype conflict with bf16 params
            router = getattr(layer.transformer_layer.mlp, "router", None)
            if router is not None:
                with ms.DeviceCtx("meta"):
                    fully_shard(
                        router,
                        **fsdp_config,
                        reshard_after_forward=reshard_after_forward,
                        replicate_params=mtp_replicate_params
                    )

            # Shared experts gate (fp32) — independent wrap
            shared_experts = getattr(layer.transformer_layer.mlp, "shared_experts", None)
            shared_gate = getattr(shared_experts, "shared_experts_gate", None) if shared_experts is not None else None
            if shared_gate is not None:
                with ms.DeviceCtx("meta"):
                    fully_shard(shared_gate, **fsdp_config, reshard_after_forward=reshard_after_forward)

            # Norms inside transformer_layer (fp32) — independent wrap
            layer_norms = _collect_layer_norms(layer.transformer_layer)
            for norm in layer_norms.values():
                with ms.DeviceCtx("meta"):
                    fully_shard(norm, **fsdp_config)

            # MTP-specific norms (fp32) — independent wrap
            for norm_attr in ("enorm", "hnorm", "final_layernorm"):
                norm = getattr(layer, norm_attr, None)
                if norm is not None:
                    with ms.DeviceCtx("meta"):
                        fully_shard(norm, **fsdp_config)

            # HC modules inside transformer_layer — same independent wrap as the
            # main decoder layers (2c); otherwise the tiny mHC params get swept into
            # the generic MTP layer wrap and sharded on dim 0, which their shapes
            # do not allow.
            _wrap_hc_modules(
                layer.transformer_layer,
                fsdp_config,
                reshard_after_forward,
                dense_shard_degree,
            )

            # Keep the FP32 mHC head projection separate from the BF16 outer MTP
            # layer and replicate the head's scalar/vector parameters.
            _wrap_hc_head(layer, fsdp_config, reshard_after_forward, dense_shard_degree)

            # DSv4 hybrid-attention fp32 sub-modules — same independent wrap as the
            # main decoder layers (2c'), keeping fp32 norms / attn_sink out of the
            # MTP layer's bf16 param-group.
            _wrap_dsv4_attention_modules(
                layer.transformer_layer.self_attention, fsdp_config, reshard_after_forward,
                mtp_replicate_params,
            )

            # Wrap the MTP layer (remaining: eh_proj + transformer_layer residual, all bf16)
            with ms.DeviceCtx("meta"):
                fully_shard(
                    layer,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                    replicate_params=mtp_replicate_params,
                )

    # --- 4. Wrap root ---
    with ms.DeviceCtx("meta"):
        fully_shard(model, **fsdp_config)

    # --- 5. Disable gradient division ---
    if getattr(parallelism, "disable_gradient_division", True):
        disable_fsdp_gradient_division(model)

    # --- 6. Prefetch chains ---
    # Extend the prefetch chain with MTP layers so their FSDP all-gathers overlap
    # with the preceding layer's forward/backward compute.
    if mtp:
        if tail_modules:
            layers.extend(tail_modules)
        for mtp_layer in mtp.layers:
            layers.append(mtp_layer)
        _setup_gpt_prefetch(embedding, layers, [])
    else:
        _setup_gpt_prefetch(embedding, layers, tail_modules)

    if parallel_dims.dp_replicate_enabled:
        logger.info("Successfully applied HSDP (Hybrid Sharded Data Parallel)")
    else:
        logger.info("Successfully applied FSDP (Fully Sharded Data Parallel)")


def apply_context_parallel_attention(
    model: nn.Cell,
    cp_mesh: DeviceMesh,
    parallel_dims: Any,
    parallelism: Any,
) -> None:
    """Apply Hyper-Parallel context parallel hooks to GPT attention modules."""
    cp_size = parallel_dims.cp
    gpt_model = _unwrap_gptmodel(model)
    model_config = getattr(gpt_model, "config", None)

    method = getattr(model_config, "_mf_runtime_context_parallel_method", None)
    if method is None:
        method = getattr(parallelism, "context_parallel_method", "colossal")
    method = method.lower()

    async_enabled = bool(getattr(parallelism, "context_parallel_async", False))
    if model_config is not None:
        async_enabled = bool(getattr(model_config, "_mf_runtime_context_parallel_async", async_enabled))
    input_layout = getattr(model_config, "input_layout", None) if model_config is not None else None
    if async_enabled and method == "colossal":
        raise NotImplementedError(
            "Async context parallel currently supports 'ulysses' and 'hybrid' only. "
            "Use context_parallel_async=false for colossal CP."
        )

    requested_ulysses_degree = (
        getattr(model_config, "_mf_runtime_ulysses_degree_in_cp", None)
        if model_config is not None else None
    )
    if requested_ulysses_degree is None:
        requested_ulysses_degree = getattr(parallelism, "ulysses_degree_in_cp", None)

    if method == "colossal":
        ulysses_degree = 1
    elif method == "ulysses":
        ulysses_degree = cp_size if requested_ulysses_degree is None else requested_ulysses_degree
    else:
        ulysses_degree = requested_ulysses_degree

    if model_config is not None:
        setattr(model_config, "_mf_runtime_context_parallel_method", method)
        setattr(model_config, "_mf_runtime_context_parallel_async", async_enabled)
        setattr(model_config, "_mf_runtime_ulysses_degree_in_cp", ulysses_degree)
        setattr(model_config, "_mf_runtime_cp_rank_list", tuple(cp_mesh.rank_list))
        rotary_pos_emb = getattr(gpt_model, "rotary_pos_emb", None)
        if rotary_pos_emb is not None and hasattr(rotary_pos_emb, "use_position_ids"):
            setattr(rotary_pos_emb, "use_position_ids", True)
            # Context parallel forces explicit position_ids, so the rotary cos/sin carries a
            # per-batch dimension. The fused interleaved RoPE op (apply_rope_fusion) does not
            # support non-broadcast batch>1 cos/sin and fails in tiling. Reject the combo early.
            # Exception: with EOD attn-mask compression (TND layout), CP flattens the per-rank
            # position_ids to batch=1 (see prepare_context_parallel_input's TND slicing), so
            # cos/sin stays broadcastable and the fused op works -- allow it.
            if getattr(model_config, "apply_rope_fusion", False) and \
                    not getattr(model_config, "use_eod_attn_mask_compression", False):
                raise ValueError(
                    "Context parallel does not support apply_rope_fusion in pynative mode "
                    "unless EOD attn-mask compression (TND layout) is enabled: without TND, "
                    "CP feeds per-batch cos/sin to the fused RoPE op, which rejects it in tiling. "
                    "Set apply_rope_fusion=false, or enable use_eod_attn_mask_compression."
                )
        if async_enabled and method == "ulysses":
            num_heads = getattr(model_config, "num_attention_heads", None)
            if num_heads is not None:
                if num_heads % ulysses_degree != 0:
                    raise ValueError(
                        f"num_attention_heads ({num_heads}) must be divisible by "
                        f"ulysses_degree ({ulysses_degree})."
                    )

    cp_style = build_context_parallel_attention_style(
        method=method,
        cp_size=cp_size,
        ulysses_degree_in_cp=ulysses_degree,
        input_layout=input_layout,
        async_enabled=async_enabled,
    )

    decoder = getattr(gpt_model, "decoder", None)
    decoder_layers = getattr(decoder, "layers", None)
    if decoder_layers is None:
        raise ValueError("Unable to locate GPT decoder layers for context parallel application.")

    def _find_one_module(transformer_block, block_label, suffix):
        matches = [
            (name, cell) for name, cell in transformer_block.cells_and_names()
            if "self_attention" in name and name.endswith(suffix)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one self_attention {suffix} in {block_label}, "
                f"but got {[name for name, _ in matches]}."
            )
        return matches[0]

    def _apply_cp_to_block(transformer_block, block_label):
        cp_module_path, cp_module = _find_one_module(transformer_block, block_label, "core_attention")
        if isinstance(cp_module, CompressedSparseAttention):
            attention_path, attention_module = _find_one_module(transformer_block, block_label, "self_attention")
            DSv4HybridAttentionContextParallel(cp_mesh)._apply(attention_module, cp_mesh)
            logger.info(
                "Applied CSA context parallel to %s.%s through %s",
                block_label,
                cp_module_path,
                attention_path,
            )
            return

        if async_enabled:
            attention_path, attention_module = _find_one_module(transformer_block, block_label, "self_attention")
            cp_style.apply_to_attention(attention_module, cp_mesh)
            logger.info(
                "Applied async context parallel to %s.%s through %s handoff boundaries",
                block_label,
                cp_module_path,
                attention_path,
            )
            return

        cp_style._apply(cp_module, cp_mesh)
        logger.info("Applied context parallel to %s.%s", block_label, cp_module_path)

    for layer_idx, transformer_block in enumerate(decoder_layers):
        _apply_cp_to_block(transformer_block, f"decoder.layers.{layer_idx}")

    mtp = getattr(gpt_model, "mtp", None)
    if mtp:
        for layer_idx, mtp_layer in enumerate(mtp.layers):
            _apply_cp_to_block(mtp_layer, f"mtp.layers.{layer_idx}")

def _setup_moe_aux_loss_group(model, parallel_dims):
    """Attach TP/CP reduction groups to every sequence-sharded MoE router.

    This function equips each MoE router with the process group used to
    all-reduce ``tokens_per_expert`` before the aux loss is computed.

    TP sequence parallelism and CP both shard the router token view. Their
    per-axis groups recover global token statistics without gathering router
    activations. DP remains excluded because it owns independent samples.
    """
    if parallel_dims is None:
        return

    aux_groups: list = []
    aux_group_size = 1
    tp_mesh = parallel_dims.get_optional_mesh("tp")
    tp_group = None
    tp_size = 1
    if tp_mesh is not None:
        tp_group = tp_mesh.get_group()
        tp_size = tp_mesh.size()
        aux_groups.append(tp_group)
        aux_group_size *= tp_size
    cp_mesh = parallel_dims.get_optional_mesh("cp")
    cp_groups = []
    cp_size = 1
    if cp_mesh is not None:
        cp_groups.append(cp_mesh.get_group())
        cp_size = cp_mesh.size()
        aux_groups.extend(cp_groups)
        aux_group_size *= cp_size

    gpt_model = _unwrap_gptmodel(model)
    moe_routers = []
    for _, cell in gpt_model.cells_and_names():
        router = getattr(cell, "router", None)
        # Identify TopKRouters by their ``aux_loss_group`` attribute, without
        # importing the moe package (keeps this helper decoupled).
        if router is not None and hasattr(router, "moe_aux_loss_coeff"):
            moe_routers.append(router)

    mtp = getattr(gpt_model, "mtp", None)
    if mtp is not None:
        for _, cell in mtp.cells_and_names():
            router = getattr(cell, "router", None)
            if router is not None and hasattr(router, "moe_aux_loss_coeff"):
                moe_routers.append(router)

    if not moe_routers:
        return

    # Set global communication domain variables in moe_utils so both the
    # router forward pass (``get_tokens_per_expert_and_token_count``) and
    # the step-end aggregation (``track_moe_metrics``) share the same TP/CP
    # reduction group without per-router attribute plumbing.
    set_moe_aux_loss_group_info(aux_groups, aux_group_size)
    for router in moe_routers:
        router.enable_sequence_parallel(tp_group, tp_size, cp_groups, cp_size)

    logger.info(
        "[MoE-AuxLossGroup] set global TP/CP aux_loss groups "
        "(size=%d, ) for %d MoE router(s).",
        aux_group_size, len(moe_routers),
    )


def _apply_spmd_parallelism(
        model: nn.Cell,
        parallel_dims: Any,
        parallelism: Any,
        recompute: Any,
        recompute_comm: Any,
        swap: Any,
        overlap=None,
) -> nn.Cell:
    """Unified GPTModel parallelization entry point.

    Args:
        overlap: Optional
            :class:`~hyper_parallel.core.pipeline_parallel.comm_compute_overlap.CommComputeOverlap`
            instance.  When provided (and ``ep_enabled`` is ``True`` with
            ``moe_token_dispatcher_type='alltoall'``), replaces the plain
            :class:`ExpertParallel` strategy with
            :class:`OverlapExpertParallel` and registers
            ``CHUNK_START``/``CHUNK_END`` hooks on the model.
    """
    logger.info("Starting GPTModel parallelization for MCore architecture...")

    # Phase 1: TP
    tp_mesh = None
    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        with ms.DeviceCtx("meta"):
            apply_non_moe_tp(
                model,
                tp_mesh=tp_mesh,
                enable_ep=parallel_dims.ep_enabled,
                enable_mc2=getattr(parallelism, "enable_mc2", False),
            )

    # Phase 2: EP
    if parallel_dims.ep_enabled:
        ep_mesh = parallel_dims.get_mesh("ep")
        if overlap is not None and parallelism.moe_token_dispatcher_type == "alltoall":
            # Overlap path: OverlapExpertParallel + CHUNK_START/END hooks.
            # apply_moe_ep_overlap_tp wraps its own weight sharding in
            # ms.DeviceCtx("meta") internally and registers hooks outside it.
            apply_moe_ep_overlap_tp(
                model,
                overlap=overlap,
                tp_mesh=tp_mesh,
                ep_mesh=ep_mesh,
                moe_token_dispatcher_type=parallelism.moe_token_dispatcher_type,
                expert_async_d2h=getattr(parallelism, "expert_parallel_async_d2h", False),
            )
        else:
            with ms.DeviceCtx("meta"):
                apply_moe_ep_tp(
                    model,
                    tp_mesh=tp_mesh,
                    ep_mesh=ep_mesh,
                    moe_token_dispatcher_type=parallelism.moe_token_dispatcher_type,
                    npu_nums_per_device=parallelism.npu_nums_per_device,
                    expert_async_d2h=getattr(parallelism, "expert_parallel_async_d2h", False),
                )

    # Phase 3: CP
    if parallel_dims.cp_enabled:
        apply_context_parallel_attention(
            model=model,
            cp_mesh=parallel_dims.get_mesh("cp"),
            parallel_dims=parallel_dims,
            parallelism=parallelism,
        )

    # Phase 4: AC
    if recompute.mode != "None" or recompute_comm.enable or swap.enable:
        gptmodel = _unwrap_gptmodel(model)
        apply_ac(
            gptmodel.decoder,
            recompute,
            recompute_comm,
            swap,
            parallelism.pipeline_parallel,
            mtp_block=getattr(gptmodel, "mtp", None),
        )

    # Phase 5: FSDP/HSDP
    apply_fsdp(
        model,
        parallel_dims,
        parallelism,
    )
    _tag_dsv4_tp_replicated_grad_norm_params(model)

    for param_name, param in model.parameters_and_names():
        if isinstance(param, DTensor):
            logger.debug(param_name, param.layout, param.placements)
        else:
            logger.debug(f"{param_name} is not DTensor, shape is {param.shape}")

    # Phase 6: CP root I/O hooks. Apply after FSDP so the hooks live on the final
    # model boundary called by trainer.
    if parallel_dims.cp_enabled:
        apply_context_parallel_model_io(model, parallel_dims, parallelism)

    # Scope the qk_clip max-logit all-reduce to the dp x cp (loss_mesh) domain
    # instead of the whole world: tp/pp hold different heads/layers, so reducing
    # over loss_mesh is both correct and far cheaper than a world all-reduce.
    gpt_model = _unwrap_gptmodel(model)
    if parallel_dims.dp_cp_enabled and hasattr(gpt_model, "set_qk_clip_reduce_group"):
        try:
            gpt_model.set_qk_clip_reduce_group(parallel_dims.world_mesh.get_group("loss_mesh"))
        except (RuntimeError, ValueError, KeyError) as exc:
            logger.warning(
                "[QK-Clip] could not resolve loss_mesh reduce group (%s); "
                "falling back to world all-reduce.", exc)

    # Tag every MoE router with the dp x cp group so the seq_aux_loss /
    # global_aux_loss can all-reduce tokens_per_expert and aggregated probs
    # before applying the Switch formula.
    _setup_moe_aux_loss_group(model, parallel_dims)

    logger.info("GPTModel parallelization completed.")
    return model


def _make_overlap_b_f_callback(overlap):
    """Build the OVERLAP_B_F schedule callback for PP+EP comm/compute overlap.

    The callback drives :meth:`CommComputeOverlap.run` with the forward and
    backward closures for a paired ``OVERLAP_B_F`` schedule step.  It mirrors
    the PoC's ``_make_overlap_b_f_callback`` for the mindformers trainer.

    MindSpore PyNative notes:
    - ``_pynative_executor.set_enable_grad(True)`` must be called explicitly
      on the daemon BWD thread because the grad-enable flag is thread-local
      and the daemon does not inherit the main thread's enabled state.
    - MindSpore's current device is process-wide (not thread-local like
      PyTorch), so the BWD thread inherits the correct device automatically.
    - After ``backward_one_chunk``, one explicit
      ``coordinator.rendezvous(HookRole.COMPUTE)`` is called out-of-band to
      pair with ``CHUNK_END.fwd`` because MS autograd may skip
      ``CHUNK_START.bwd`` when the chunk input lacks ``requires_grad``.
    """

    def _callback(step, ctx):
        from mindspore.common.api import _pynative_executor  # pylint: disable=C0415
        from hyper_parallel.core.pipeline_parallel.hook_coordinator import HookRole  # pylint: disable=C0415

        bwd_step, fwd_step = step.sub_steps
        schedule = ctx.schedule
        fwd_stage = schedule._stage_dict[fwd_step.stage_index]  # pylint: disable=W0212
        bwd_stage = schedule._stage_dict[bwd_step.stage_index]  # pylint: disable=W0212
        fwd_mi, bwd_mi = fwd_step.micro_index, bwd_step.micro_index

        def fwd_fn():
            # CHUNK_START equivalent, at callback level so the bracket covers
            # the WHOLE chunk record — including preprocess/embedding/rotary
            # (entry) and lm-head/loss (exit), which live outside the decoder.
            # MS PyNative does not support concurrent FWD-record + BWD-replay;
            # decoder-level Cell hooks left those segments recording while the
            # BWD thread replayed loss.bwd / lm-head.bwd (its pre-D_LAST tail),
            # corrupting memory and grads. Pairing is unchanged: this COMPUTE
            # rendezvous pairs with D_LAST.bwd, parking FWD until the BWD
            # thread has passed its first hook.
            if overlap.coordinator.is_enabled():
                overlap.coordinator.rendezvous(HookRole.COMPUTE)
            # wait_fwd_recv pops the cached recv handle (overlap_p2p) and waits;
            # no-op when nothing is cached. fwd_handle_cache is touched only by
            # the main thread here, bwd_handle_cache only by the daemon below.
            schedule.wait_fwd_recv(fwd_stage.stage_index, fwd_mi)
            out = fwd_stage.forward_one_chunk(
                fwd_mi, ctx.arg_mbs[fwd_mi], ctx.kwarg_mbs[fwd_mi],
            )
            schedule.update_losses(fwd_stage, out, ctx.losses)
            # CHUNK_END equivalent: release the C_last comm event, then pair
            # with the bwd_fn out-of-band rendezvous below.
            if overlap.coordinator.is_enabled():
                overlap.coordinator.notify_dispatched(HookRole.COMM)
                overlap.coordinator.rendezvous(HookRole.COMPUTE)

        def bwd_fn():
            # MS PyNative grad-enable is thread-local; enable explicitly on
            # the daemon BWD thread so value_and_grad does not raise.
            _pynative_executor.set_enable_grad(True)
            schedule.wait_bwd_recv(bwd_stage.stage_index, bwd_mi)
            bwd_stage.backward_one_chunk(bwd_mi)
            # Pair-8 BWD partner taken out-of-band: MS autograd may skip
            # CHUNK_START.bwd when the chunk input has no requires_grad.
            # One explicit rendezvous(COMPUTE) here matches CHUNK_END.fwd.
            if overlap.coordinator.is_enabled():
                overlap.coordinator.rendezvous(HookRole.COMPUTE)

        # Activation-recompute compatibility: fire the BWD chunk's forward
        # re-run serially on THIS (main) thread, BEFORE overlap.run enables the
        # coordinator and spawns the BWD daemon. Two reasons it must be here:
        #   1. The coordinator is still disabled, so the re-run's A/B/C/D sync
        #      hooks are no-ops (the is_enabled() gate) — no stray rendezvous to
        #      desync the protocol.
        #   2. backward_one_chunk then reuses the cached recomputed activations
        #      instead of re-running the forward on the daemon thread, which
        #      would be concurrent FWD-record + BWD-replay (unsupported by MS
        #      PyNative) and would re-fire the hooks, deadlocking the coordinator.
        # No-op when the chunk has no checkpoint_wrapper'd blocks (recompute
        # off). Mirrors the reference PoC's recompute path.
        bwd_stage.recompute_one_chunk(bwd_mi)

        # Diagnostic: EP_OVERLAP_SEQ=1 runs bwd then fwd sequentially on the main
        # thread WITHOUT enabling the coordinator (all A/B/C/D/CHUNK sync hooks
        # stay passthrough). This isolates the OverlapExpertParallel dispatch/
        # combine math from the dual-thread synchronization: if grads match the
        # baseline here, the math is correct and the bug is in the threading.
        if os.environ.get("EP_OVERLAP_SEQ") == "1":
            # Diagnostic isolation: run bwd then fwd sequentially with the
            # coordinator disabled (sync hooks passthrough). Validates the
            # OverlapExpertParallel dispatch/combine math without dual-threading.
            bwd_fn()
            fwd_fn()
        else:
            overlap.run(fwd_fn=fwd_fn, bwd_fn=bwd_fn)

    return _callback


def apply_pp(
        model,
        pp_mesh,
        parallel_dims,
        parallelism,
        recompute,
        recompute_comm,
        swap,
        gradient_accumulation_steps: int = 1,
):
    """Apply pipeline parallelism to the GPTModel.

    When ``parallelism.pipeline_parallel_overlap_b_f`` is ``True`` and
    ``parallel_dims.ep_enabled`` is ``True`` with
    ``moe_token_dispatcher_type='alltoall'``, creates a single
    :class:`~hyper_parallel.core.pipeline_parallel.comm_compute_overlap.CommComputeOverlap`
    orchestrator per rank, passes it to :func:`_apply_spmd_parallelism` so
    :func:`apply_moe_ep_overlap_tp` installs :class:`OverlapExpertParallel` on
    every MoE layer, and registers it on the schedule for the
    ``OVERLAP_B_F`` callback.
    """
    layer_setting = PpLayerSetting(model.config.num_hidden_layers, parallelism)

    model_cls = type(model)
    builder = StageModelBuilder(layer_setting)
    # tp_mesh lets each stage resolve the rank group for TP/SP-sharded activations
    # crossing the PP boundary (the pp_mesh's flat world root can't resolve "tp").
    tp_mesh = parallel_dims.get_optional_mesh("tp")
    stages, model_parts = builder.build_stages(model_cls, model.config, pp_mesh, tp_mesh)
    # PP rebuilds fresh per-stage models from config, discarding any LoRA injected into
    # the original model. Re-inject into each stage on meta device BEFORE SPMD parallelism
    # so the adapters pick up TP/FSDP layouts. strict=False: an embedding-only stage may
    # hold no target layer; the "no adapters anywhere" guard is in the Trainer.
    lora_cfg = getattr(model.config, "_mf_lora_config", None)
    del model
    if lora_cfg is not None:
        for part in model_parts:
            build_lora_model(part, lora_cfg, strict=False)

    # Create one CommComputeOverlap orchestrator per rank when B/F EP overlap
    # is requested.  One orchestrator is shared across ALL chunks on this rank
    # so that the coordinator's barrier/event state is consistent across the
    # interleaved chunk sequence.
    use_ep_overlap = (
        getattr(parallelism, "pipeline_parallel_overlap_b_f", False)
        and parallel_dims.ep_enabled
        and getattr(parallelism, "moe_token_dispatcher_type", "alltoall") == "alltoall"
    )
    overlap = None
    if use_ep_overlap:
        from hyper_parallel.core.pipeline_parallel.comm_compute_overlap import CommComputeOverlap  # pylint: disable=C0415
        overlap = CommComputeOverlap()
        logger.info("PP+EP overlap enabled: created CommComputeOverlap orchestrator")

    for part in model_parts:
        _apply_spmd_parallelism(
            part,
            parallel_dims,
            parallelism,
            recompute,
            recompute_comm,
            swap,
            overlap=overlap,
        )

    # Tag the MTP-shared input embedding so its gradient is summed across the
    # embedding-owning PP stages before the optimizer step (handled in the
    # grad-norm helper), keeping the embedding identical to the single-card run.
    # Run ONCE over all chunks this rank owns (not per-chunk inside
    # _apply_spmd_parallelism): under interleaving the main and MTP embeddings
    # live in different chunks/stages, so ownership must be aggregated across
    # the whole rank for the cross-PP-stage sync group to form.
    _setup_mtp_embedding_grad_sync(model_parts, parallel_dims)

    # Adjust the last-stage loss backward scaling so PP gradients match the single-card case.
    # DTensor repeat compensation is handled by ScaledLossPipelineStage from the actual output layout.
    # MoE/MTP losses are scaled separately by their own auto-scalers and are not affected here.
    main_loss_sense = get_loss_sense(
        parallelism=parallelism,
        enable_parallel=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        apply_gradient_accumulation=True,
        )
    loss_scale = float(main_loss_sense.asnumpy().item())
    for stage in stages:
        stage.loss_scale = loss_scale

    micro_batch_num = parallelism.pipeline_parallel_microbatch_size
    schedule_type = _infer_schedule_type(parallelism)
    has_moe = all(
        any(hasattr(layer.mlp, "experts") for layer in part.model.decoder.layers)
        for part in model_parts
    )
    schedule = _create_schedule(schedule_type, stages, micro_batch_num, parallelism, swap=swap.enable, has_moe=has_moe)
    logger.info(f"Pipeline schedule: {schedule_type}, exec_order: {schedule.exec_order}")

    # Register the OVERLAP_B_F callback when EP overlap is active.
    if use_ep_overlap and overlap is not None:
        from hyper_parallel.core.pipeline_parallel.scheduler import MetaStepType  # pylint: disable=C0415
        schedule.register_custom_function(
            MetaStepType.OVERLAP_B_F,
            _make_overlap_b_f_callback(overlap),
        )
        logger.info("Registered OVERLAP_B_F callback on schedule")

    has_first = any(s.stage_index == 0 for s in stages)
    has_last = any(s.stage_index == layer_setting.num_virtual_stages - 1 for s in stages)

    return model_parts, schedule, has_first, has_last

# ---------------------------------------------------------------------------
# GPT parallelization entry point
# ---------------------------------------------------------------------------

def parallelize_gptmodel(
        model: nn.Cell,
        parallel_dims: Any,
        parallelism: Any,
        recompute: Any,
        recompute_comm: Any,
        swap: Any,
        gradient_accumulation_steps: int = 1,
) -> List[nn.Cell]:
    """Apply pipeline parallelism to the GPTModel."""

    if parallel_dims.pp_enabled:
        pp_mesh = parallel_dims.get_mesh("pp")
        return apply_pp(
            model,
            pp_mesh=pp_mesh,
            parallel_dims=parallel_dims,
            parallelism=parallelism,
            recompute=recompute,
            recompute_comm=recompute_comm,
            swap=swap,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    _apply_spmd_parallelism(model, parallel_dims, parallelism, recompute, recompute_comm, swap)

    return [model], None, False, False
