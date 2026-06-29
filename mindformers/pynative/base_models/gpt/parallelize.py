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
from mindspore.mint.distributed import get_rank, all_gather, new_group

from hyper_parallel import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, Partial
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
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)
from mindformers.pynative.distributed.mc2_style import (
    MC2ColwiseParallel,
    MC2RowwiseParallel,
)
from mindformers.pynative.distributed.context_parallel import (
    apply_context_parallel_model_io,
    build_context_parallel_attention_style,
)
from mindformers.pynative.layers.layer_norm import FusedLayerNorm, FusedRMSNorm
from mindformers.pynative.distributed.tensor_parallel import NoParallel
from mindformers.pynative.distributed.expert_parallel import ExpertParallel, DeredundancyExpertParallel
from mindformers.pynative.distributed.ep_overlap import OverlapExpertParallel
from mindformers.pynative.distributed.pipeline_parallel import PpLayerSetting, StageModelBuilder, _create_schedule, _infer_schedule_type
from mindformers.pynative.pet.lora_adapter import build_lora_model
from mindformers.pynative.distributed.parallelize import parallelize_module
from mindformers.pynative.distributed.activation_checkpoint import apply_ac
from mindformers.pynative.distributed.utils import distribute_module, get_loss_sense
from mindformers.pynative.base_models.gpt.gpt_model import GPTModel
from mindformers.pynative.transformers.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttention,
)
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


def _setup_mtp_embedding_grad_sync(model, parallel_dims):
    """Tag the MTP-shared input embedding for cross-PP-stage gradient sync.

    Under pipeline parallelism with MTP enabled, the input embedding is
    replicated on two PP stages: stage 0 (the main forward) and the last stage
    (which hosts the MTP block and reuses the embedding to embed the shifted MTP
    tokens). The single-card baseline keeps ONE embedding tensor that receives
    ``main_grad + mtp_grad``; under PP the stage-0 copy only sees ``main_grad``
    and the MTP-stage copy only sees ``mtp_grad``. With no gradient sync the two
    copies drift apart and the MTP loss diverges from single-card (a slow,
    one-sided, compounding error).

    This builds a process group over the embedding-owning PP ranks (per
    dp/tp/cp/ep coordinate) and tags the local ``word_embeddings.weight`` with
    ``_embedding_grad_sync_group`` / ``_pp_replica_count``. The grad-norm helper
    (`_calculate_global_grad_norm` -> `_get_grad_factor`) then all-reduces the
    tagged gradient before computing the norm/step and counts the replicated
    embedding exactly once, with no changes required in the trainer.
    """
    if parallel_dims is None or not parallel_dims.pp_enabled:
        return

    gpt_model = _unwrap_gptmodel(model)
    cfg = gpt_model.get_gpt_transformer_config()
    if not getattr(cfg, "mtp_num_layers", 0):
        return

    # Locate this rank's input-embedding weight (present on stage 0 and the MTP
    # stage). At most one on a non-interleaved run.
    embed_weights = [
        param for name, param in model.parameters_and_names()
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
        "[MTP-EmbedSync] rank %d tagged embedding for grad-sync group %s",
        local_rank, embed_ranks,
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


def _distribute_router(
        module,
        router_mesh,
        input_layouts,
        output_layouts,
        use_local_output=True
):
    """
    Distribute a router module by defining custom input/output transformations
    for DTensor-based parallel execution.

    Args:
        module: The router module to be distributed.
        router_mesh: Device mesh used for distribution.
        input_layouts (tuple): Expected placements for each input tensor.
        output_layouts (tuple): Desired placements for each output tensor.
        use_local_output (bool): Whether to convert DTensor outputs back to local tensors.

    This function wraps the module with input/output hooks to ensure:
    - Inputs are converted to DTensor with correct placements.
    - Outputs are redistributed and optionally converted back to local tensors.
    """
    def input_fn(device_mesh, mod, args):
        """
        Transform inputs into DTensor with correct placements before forward pass.
        """
        _ = mod
        hidden_state, expert_bias, input_ids = args

        # Handle hidden_state: convert or redistribute to expected layout
        if not isinstance(hidden_state, DTensor):
            hidden_state = DTensor.from_local(
                hidden_state, device_mesh, (input_layouts[0],)
            )
        elif hidden_state.placements != input_layouts[0]:
            hidden_state = hidden_state.redistribute(
                placements=(input_layouts[0],), device_mesh=device_mesh
            )

        # Handle expert_bias if provided
        if expert_bias is not None:
            if not isinstance(expert_bias, DTensor):
                expert_bias = DTensor.from_local(
                    expert_bias, device_mesh, (input_layouts[1],)
                )
            elif expert_bias.placements != input_layouts[1]:
                expert_bias = expert_bias.redistribute(
                    placements=(input_layouts[1],), device_mesh=device_mesh
                )

        # Handle input_ids if provided (for hash routing)
        if input_ids is not None and len(input_layouts) > 2:
            if not isinstance(input_ids, DTensor):
                input_ids = DTensor.from_local(
                    input_ids, device_mesh, (input_layouts[2],)
                )
            elif input_ids.placements != input_layouts[2]:
                input_ids = input_ids.redistribute(
                    placements=(input_layouts[2],), device_mesh=device_mesh
                )

        if input_ids is not None:
            return hidden_state, expert_bias, input_ids
        return hidden_state, expert_bias

    def output_fn(device_mesh, mod, args, outputs):
        """
        Transform outputs to desired placements and optionally convert to local tensors.
        """
        _, _ = mod, args
        prepared_outputs = []

        # Ensure each output matches the expected layout
        for out, out_layout in zip(outputs, output_layouts):
            if out.placements != out_layout:
                out = out.redistribute(
                    placements=(out_layout,),
                    device_mesh=device_mesh,
                )

            # Convert to local tensor if required
            prepared_outputs.append(
                out.to_local() if use_local_output else out
            )

        return tuple(prepared_outputs)

    # Apply distribution to the module with custom input/output handlers
    distribute_module(
        module=module,
        device_mesh=router_mesh,
        parameter_shard_plan={},  # No parameter sharding defined here
        input_fn=input_fn,
        output_fn=output_fn
    )


def _apply_mtp_concat_tp(
    module,
    tp_mesh,
):
    """Distribute the MTP (Multi-Token Prediction) hidden-states concatenation module."""
    def input_fn(device_mesh, mod, args):
        """
        Transform inputs into DTensor with correct placements before forward pass.
        """
        _ = mod
        (hidden_states,) = args
        cur_hidden_states = []
        for h in hidden_states:
            if isinstance(h, DTensor):
                hidden_state = h.redistribute(
                    placements=(Replicate(),),
                    device_mesh=device_mesh,
                )
            else:
                hidden_state = DTensor.from_local(
                    h, device_mesh, (Replicate(),)
                )
            cur_hidden_states.append(hidden_state)
        return cur_hidden_states

    def output_fn(device_mesh, mod, args, outputs):
        """
        Transform outputs to desired placements and optionally convert to local tensors.
        """
        _, _ = mod, args
        hidden_states = outputs
        if isinstance(hidden_states, DTensor):
            hidden_states = hidden_states.redistribute(
                placements=(Replicate(),),
                device_mesh=device_mesh,
            )
        return hidden_states

    distribute_module(
        module=module,
        device_mesh=tp_mesh,
        parameter_shard_plan={},  # No parameter sharding defined here
        input_fn=input_fn,
        output_fn=output_fn
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


def _collect_layer_replicate_params(layer, shard_size):
    """Collect non-module parameters that must be replicated (not sharded) under FSDP."""
    replicate_params = []
    if hasattr(layer.self_attention.core_attention, "max_logits_val"):
        replicate_params.append(layer.self_attention.core_attention.max_logits_val)

    if hasattr(layer.mlp, "tokens_per_expert"):
        replicate_params.append(layer.mlp.tokens_per_expert)

    if getattr(layer.mlp, "enable_expert_bias", False):
        replicate_params.append(layer.mlp.expert_bias)

    if hasattr(layer.mlp, "router"):
        if hasattr(layer.mlp.router, "tid2eid") and layer.mlp.router.tid2eid is not None:
            replicate_params.append(layer.mlp.router.tid2eid)

    for param_name, param in layer.parameters_and_names():
        if isinstance(param, DTensor):
            shape = param.local_shape
        else:
            shape = param.shape

        if shape[0] % shard_size != 0:
            replicate_params.append(param)
            logger.warning("The shape[0]=%s of parameter %s is not divisible by data_parallel_shard=%s, " \
                 "then this parameter will not be applied fsdp.", shape[0], param_name, shard_size)

    return replicate_params


def _collect_hc_replicate_params(hc_module):
    """Collect small HC params as replicate_params (avoid sharding issues with delayed init)."""
    replicate_params = []
    for attr in ("alpha_pre", "alpha_post", "alpha_res", "bias"):
        p = getattr(hc_module, attr, None)
        if p is not None:
            replicate_params.append(p)
    return replicate_params


def _wrap_hc_modules(host, fsdp_config, reshard_after_forward):
    """Independently FSDP-wrap a host's mHC sub-modules (attn_hc/ffn_hc).

    Their params are tiny and must be replicated rather than swept into the host's
    generic wrap and sharded on dim 0 (their shapes do not allow it): 1-D params
    shard on dim 0, the rest on dim 1. Shared by the main decoder layers and the
    MTP transformer layers so both get the same treatment.
    """
    def shard_plan(param):
        if len(param.shape) == 1:
            return Shard(0)
        return Shard(1)

    for hc_attr in ("attn_hc", "ffn_hc"):
        hc_module = getattr(host, hc_attr, None)
        if hc_module is None:
            continue
        with ms.DeviceCtx("meta"):
            fully_shard(hc_module, **fsdp_config, shard_placement_fn=shard_plan,
                        reshard_after_forward=reshard_after_forward,
                        replicate_params=_collect_hc_replicate_params(hc_module))


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


def _get_dsv4_tp(transformer_layer, shard_plans):
    """get dsv4 tp plan"""
    # Fused CSA kernel has CANN shape constraints and no TP benefit from head-sharding
    # non-fused small-op path does benefit.
    sp_layout, norm_plan, rowwise_output_plan, attn_qkv_shard = shard_plans
    distribute_param_plan = []
    rope_freqs_layout = Replicate() if transformer_layer.self_attention.config.apply_rope_fusion else None
    core_attn = transformer_layer.self_attention.core_attention
    csa_q_layout = Replicate() if getattr(core_attn, "apply_dsa_kernel_fusion", False) else Shard(2)
    attn_sink_plan = Replicate() if getattr(core_attn, "apply_dsa_kernel_fusion", False) else Shard(0)
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
        input_layouts=(Partial(),),
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


def _apply_layers_tp(
    transformer_layer,
    tp_mesh,
    enable_ep,
    shard_plans,
    enable_mc2=False,
):
    """Apply tensor-parallel sharding to a single GPT transformer layer."""

    sp_layout, norm_plan, rowwise_output_plan, attn_qkv_shard = shard_plans
    colwise_parallel, prepare_module_input, prepare_module_output, prepare_module_input_output = (
        ColwiseParallel,
        PrepareModuleInput,
        PrepareModuleOutput,
        PrepareModuleInputOutput,
    )

    # MC2 fuses sequence-sharded all-gather into the column-parallel matmul. It
    # only applies when the module's input is sequence-sharded (sp_layout). In
    # this layer, MLP linear_fc1 is the qualifying entry: its input is sharded
    # and the all-gather can be folded into the matmul. Attention's linear_qkv
    # is NoParallel here (no TP weight shard) and linear_qb consumes a
    # replicated LoRA path, so neither path benefits from MC2.
    mc2_mlp_colwise_input = sp_layout if enable_mc2 else None
    mc2_mlp_colwise_parallel = MC2ColwiseParallel if enable_mc2 else colwise_parallel
    mlp_desired_input = sp_layout if enable_mc2 else Replicate()

    has_q_lora = transformer_layer.self_attention.config.q_lora_rank is not None
    q_pre_attn_layout = Shard(2) if has_q_lora else Replicate()
    attn_input_shard = attn_qkv_shard if has_q_lora else Replicate()
    # The fused RoPE kernel is dispatched as a distributed op and requires its cos/sin
    # (derived from the rotary freqs) to be DTensors, so the freqs input is replicated at
    # the module boundary. The non-fused mul/add path instead broadcasts plain freqs and
    # must keep them as plain tensors (None) to preserve q_pe's head sharding, so the freqs
    # slot is only converted when apply_rope_fusion is enabled.
    if isinstance(transformer_layer.self_attention, DSv4HybridSelfAttention):
        layer_plan, distribute_param_plan = _get_dsv4_tp(transformer_layer, shard_plans)
    else:
        rope_freqs_layout = Replicate() if transformer_layer.self_attention.config.apply_rope_fusion else None
        distribute_param_plan = []
        # Rotary embedding for q_pe (head-structured tensor from linear_qb when LoRA is
        # on, otherwise from linear_qkv's replicated slice). The second positional input is
        # the rotary freqs tensor (mscale is passed as a kwarg and stays a Python scalar).
        q_rotary_emb_plan = prepare_module_input_output(
            input_layouts=(q_pre_attn_layout, rope_freqs_layout),
            desired_input_layouts=(q_pre_attn_layout, rope_freqs_layout),
            output_layouts=(q_pre_attn_layout,),
            desired_output_layouts=(q_pre_attn_layout,),
            use_local_output=False,
        )
        # Rotary embedding for k_pe (pre-tile 1-head tensor from linear_qkv, stays Replicate).
        # k_pe comes from linear_qkv whose all-reduce is necessary, so k_pe remains Replicate.
        k_rotary_emb_plan = prepare_module_input_output(
            input_layouts=(Replicate(), rope_freqs_layout),
            desired_input_layouts=(Replicate(), rope_freqs_layout),
            output_layouts=(Replicate(),),
            desired_output_layouts=(Replicate(),),
            use_local_output=False,
        )

        # Q arrives at core_attention as attn_input_shard (= attn_qkv_shard when LoRA
        # produces a head-sharded Q, otherwise Replicate). K/V come in as Replicate and
        # get reduce-scattered here to attn_qkv_shard.
        attention_kernel_plan = prepare_module_input(
            input_layouts=(attn_input_shard, Replicate(), Replicate(), None),
            desired_input_layouts=(attn_qkv_shard, attn_qkv_shard, attn_qkv_shard, None),
            use_local_output=False,
        )

        layer_plan = {
            "input_layernorm": norm_plan,
            "self_attention": prepare_module_input(
                input_layouts=(sp_layout,),
                desired_input_layouts=(
                    Replicate()
                ),
                use_local_output=False,
            ),
            # linear_qkv is kept replicated so its output (which is split into q_a /
            # compressed_kv / k_pe along the last dim) needs no AllGather. The
            # downstream up-projections (linear_qb / linear_kvb) carry the actual TP
            # weight sharding for this attention block.
            "self_attention.linear_qkv": NoParallel(use_local_output=False),
            "self_attention.linear_kvb": colwise_parallel(output_layouts=Replicate(), use_local_output=False),
            "self_attention.k_layernorm": NoParallel(use_local_output=False),
            # NOTE: use_local_output=True so that the inputs to FlexAttention are plain Tensors
            "self_attention.core_attention": attention_kernel_plan,
            "self_attention.linear_proj": rowwise_output_plan,
            "pre_mlp_layernorm": norm_plan,
            "self_attention.apply_rotary_emb_q": q_rotary_emb_plan,
            "self_attention.apply_rotary_emb_k": k_rotary_emb_plan,
        }

        if transformer_layer.self_attention.q_rank == 0:
            raise ValueError(
                "q_rank should be greater than 0, but got 0!"
            )

        if has_q_lora:
            layer_plan.update(
                {
                    "self_attention.linear_qb": colwise_parallel(
                        output_layouts=Shard(2), use_local_output=False),
                    "self_attention.q_layernorm": NoParallel(use_local_output=False),
                }
            )

    if hasattr(transformer_layer, "attn_hc"):
        layer_plan.update(
            {
                "attn_hc": SequenceParallel(sequence_dim=0, use_local_output=False),
                "ffn_hc": SequenceParallel(sequence_dim=0, use_local_output=False),
            }
        )

    if hasattr(transformer_layer.mlp, "linear_fc1"):
        layer_plan.update(
            {
                "mlp": prepare_module_input(
                    input_layouts=(sp_layout,),
                    desired_input_layouts=(mlp_desired_input,),
                    use_local_output=False,
                ),
                "mlp.linear_fc1": mc2_mlp_colwise_parallel(
                    input_layouts=mc2_mlp_colwise_input, use_local_output=False),
                "mlp.linear_fc2": rowwise_output_plan,
            }
        )

    if hasattr(transformer_layer.mlp, "shared_experts"):
        layer_plan.update(
            {
                "mlp.shared_experts": prepare_module_output(
                    output_layouts=(sp_layout,),
                    desired_output_layouts=(Replicate(),),
                    use_local_output=False
                ),
            }
        )
        layer_plan["mlp.shared_experts.linear_fc1"] = colwise_parallel(use_local_output=False)
        layer_plan["mlp.shared_experts.linear_fc2"] = rowwise_output_plan

    core_attention = transformer_layer.self_attention.core_attention

    if hasattr(core_attention, "max_logits_val"):
        distribute_param_plan.append([core_attention, "max_logits_val", (Shard(0),)])
    if hasattr(transformer_layer.mlp, "experts"):
        layer_plan["mlp"] = prepare_module_input_output(
            input_layouts=(sp_layout,),
            desired_input_layouts=(Replicate(),),
            output_layouts=(Replicate(),),
            desired_output_layouts=(sp_layout,),
            use_local_input=False,
            use_local_output=False
        )

        layer_plan["mlp.router.moe_aux_loss_auto_scaler"] = prepare_module_output(
            output_layouts=(Replicate(),),
            desired_output_layouts=(sp_layout,),
            use_local_output=False
        )

        # distribute parameters for MoE layer
        distribute_param_plan.append([transformer_layer.mlp, "tokens_per_expert", (Replicate(),)])
        if getattr(transformer_layer.mlp, "enable_expert_bias", False):
            distribute_param_plan.append([transformer_layer.mlp, "expert_bias", (Replicate(),)])
        if hasattr(transformer_layer.mlp, "router"):
            if getattr(transformer_layer.mlp.router, "is_hash_layer", False):
                distribute_param_plan.append([transformer_layer.mlp.router, "tid2eid", (Replicate(),)])

        if not enable_ep:
            # shard expert weight if not enable_ep
            distribute_param_plan.extend([
                [transformer_layer.mlp.experts, "weight1", (Replicate(),)],
                [transformer_layer.mlp.experts, "weight2", (Replicate(),)],
            ])

        # special parallel for router
        _distribute_router(
            transformer_layer.mlp.router,
            router_mesh=tp_mesh,
            input_layouts=(Replicate(), Replicate(), Replicate()),
            output_layouts=(Replicate(), Replicate(), Replicate()),
        )

    if distribute_param_plan:
        for sub_module, param_name, sub_plan in distribute_param_plan:
            _distribute_param(
                sub_module,
                param_name=param_name,
                device_mesh=tp_mesh,
                placements=sub_plan
            )

    # LoRA: attach adapter layouts to the target styles (model-side; styles stay generic).
    _attach_lora_layouts(transformer_layer, layer_plan)

    parallelize_module(
        module=transformer_layer,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
    )


def apply_non_moe_tp(
        model: nn.Cell,
        tp_mesh: DeviceMesh,
        enable_loss_parallel: bool = False,
        enable_float8_tensorwise_tp: bool = False,
        enable_sp: bool = True,
        enable_ep: bool = False,
        enable_mc2: bool = False,
):
    """Apply tensor parallelism."""
    # MC2 (all_gather_matmul / matmul_reduce_scatter) fuses the tensor-parallel
    # collective into the matmul and is only meaningful when the sequence is
    # sharded (sequence parallel). Disable it otherwise to keep semantics correct.
    enable_mc2 = enable_mc2 and enable_sp
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    sp_layout = Shard(0) if enable_sp else Replicate()

    embed_plan = RowwiseParallel(
        input_layouts=Replicate(),
        output_layouts=Shard(1),
        use_local_output=False,
    )

    rowwise_parallel, prepare_module_output = (
        MC2RowwiseParallel if enable_mc2 else RowwiseParallel,
        PrepareModuleOutput,
    )

    model_plan = {}

    if hasattr(model.model, "embedding"):
        model_plan.update(
            {
                "model.embedding.word_embeddings": embed_plan,
            }
        )

    if getattr(model.model.decoder, "final_layernorm", None):
        model_plan.update(
            {
                "model.decoder.final_layernorm": SequenceParallel(
                    sequence_dim=0, use_local_output=False) if enable_sp else NoParallel(),
            }
        )

    if hasattr(model.model, "output_layer"):
        model_plan.update(
            {
                "model.output_layer": ColwiseParallel(
                    input_layouts=sp_layout,
                    output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                    use_local_output=False,
                ),
            }
        )

    if hasattr(model.model, "mtp_loss_auto_scaler"):
        model_plan.update(
            {
                "model.mtp_loss_auto_scaler": prepare_module_output(
                    output_layouts=(Replicate(),),
                    desired_output_layouts=(Replicate(),),
                    use_local_output=False,
                )
            }
        )

    parallelize_module(
        model,
        tp_mesh,
        model_plan,
    )

    decoder_layers = model.model.decoder.layers
    attn_input_layout = getattr(
        decoder_layers.get_first_cell().self_attention.config, "input_layout", None
        ) if decoder_layers else None
    if attn_input_layout == "TND":
        attn_qkv_shard = Shard(1)
    else:
        attn_qkv_shard = Shard(2)

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    norm_plan = SequenceParallel(sequence_dim=0, use_local_output=False) if enable_sp else NoParallel()
    rowwise_output_plan = rowwise_parallel(
        output_layouts=sp_layout, use_local_output=False,  # use_local_output=enable_sp
    )

    for transformer_layer in model.model.decoder.layers:
        _apply_layers_tp(
            transformer_layer,
            tp_mesh,
            enable_ep,
            (sp_layout, norm_plan, rowwise_output_plan, attn_qkv_shard,),
            enable_mc2=enable_mc2,
        )

    if getattr(model.model, "mtp"):
        _apply_mtp_concat_tp(
            model.model.mtp.concat_hidden_states,
            tp_mesh,
        )

        for layer in model.model.mtp.layers:
            parallelize_module(
                layer,
                tp_mesh,
                {
                    "enorm": SequenceParallel(
                        sequence_dim=0, use_local_output=False) if enable_sp else NoParallel(),
                    "hnorm": SequenceParallel(
                        sequence_dim=0, use_local_output=False) if enable_sp else NoParallel(),
                    "eh_proj": RowwiseParallel(
                        input_layouts=Shard(0), output_layouts=Shard(0), use_local_output=False),
                    "final_layernorm": SequenceParallel(
                        sequence_dim=0, use_local_output=False) if enable_sp else NoParallel(),
                },
            )

            _apply_layers_tp(
                layer.transformer_layer,
                tp_mesh,
                enable_ep,
                (sp_layout, norm_plan, rowwise_output_plan, attn_qkv_shard,),
                enable_mc2=enable_mc2,
            )

    if hasattr(model.model, "loss") and (hasattr(model.model.loss, "log_softmax") and
                                         hasattr(model.model.loss, "nll_loss")):
        # apply parallelism to loss custom backend functions
        # NOTE: should support loss parallel in future
        layer_plan = {
            "log_softmax": PrepareModuleInputOutput(
                input_layouts=(Replicate(),),
                desired_input_layouts=(Replicate(),),
                output_layouts=(Replicate(),),
                desired_output_layouts=(Replicate(),),
                use_local_input=True,
                use_local_output=False,
            ),
            "nll_loss": PrepareModuleInputOutput(
                input_layouts=(Replicate(), None),
                desired_input_layouts=(Replicate(), None),
                output_layouts=(Replicate(),),
                desired_output_layouts=(Replicate(),),
                use_local_input=True,
                use_local_output=False,
            ),
        }
        parallelize_module(
            module=model.model.loss,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        f"Tensor Parallelism to the model (MC2 fusion {'ENABLED' if enable_mc2 else 'disabled'})"
    )


def apply_moe_ep_tp(
        model: nn.Cell,
        tp_mesh: DeviceMesh = None,
        ep_mesh: DeviceMesh = None,
        moe_token_dispatcher_type: str = "all_to_all",
        npu_nums_per_device: int = 8,
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
            experts_plan = ExpertParallel(model.model.config.moe_permute_fusion)
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
                moe_permute_fusion=False,  # overlap path always uses manual permute
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
        with ms.DeviceCtx("meta"):
            fully_shard(embedding, **fsdp_config, reshard_after_forward=reshard_after_forward)

    # --- 2. Wrap transformer layers ---
    # Principle: small modules first, then larger modules
    for layer in layers:
        # 2a. Expert FSDP (small expert block)
        if hasattr(layer.mlp, "experts") and edp_mesh is not None:
            with ms.DeviceCtx("meta"):
                fully_shard(layer.mlp.experts, **efsdp_config, reshard_after_forward=reshard_after_forward)

        replicate_params = _collect_layer_replicate_params(layer, parallel_dims.fsdp)

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
            with ms.DeviceCtx("meta"):
                fully_shard(shared_gate, **fsdp_config, reshard_after_forward=reshard_after_forward)

        # 2b. Wrap norms independently (small modules first)
        layer_norms = _collect_layer_norms(layer)
        for norm in layer_norms.values():
            with ms.DeviceCtx("meta"):
                fully_shard(norm, **fsdp_config)

        # 2c. Wrap HC modules independently (small modules, all params replicated)
        _wrap_hc_modules(layer, fsdp_config, reshard_after_forward)

        # 2d. Wrap the transformer layer (large module)

        with ms.DeviceCtx("meta"):
            fully_shard(
                layer,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
                replicate_params=replicate_params
            )

    # --- 3. Wrap final_layernorm and output_layer independently ---
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
                with ms.DeviceCtx("meta"):
                    fully_shard(
                        tail_module,
                        **fsdp_config,
                        reshard_after_forward=reshard_policy == "always"
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

            mtp_replicate_params = _collect_layer_replicate_params(layer.transformer_layer, parallel_dims.fsdp)

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
            _wrap_hc_modules(layer.transformer_layer, fsdp_config, reshard_after_forward)

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
                enable_loss_parallel=getattr(parallelism, "enable_loss_parallel", False),
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
            )
        else:
            with ms.DeviceCtx("meta"):
                apply_moe_ep_tp(
                    model,
                    tp_mesh=tp_mesh,
                    ep_mesh=ep_mesh,
                    moe_token_dispatcher_type=parallelism.moe_token_dispatcher_type,
                    npu_nums_per_device=parallelism.npu_nums_per_device,
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

    # Tag the MTP-shared input embedding so its gradient is summed across the
    # embedding-owning PP stages before the optimizer step (handled in the
    # grad-norm helper), keeping the embedding identical to the single-card run.
    _setup_mtp_embedding_grad_sync(model, parallel_dims)

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

    # Adjust the last-stage loss backward scaling so PP gradients match the single-card case (loss / grad_accum).
    # Note: DP/TP/CP are already included in get_loss_sense as 1/(dp*tp*cp)/grad_accum.
    # However, PipelineStage.get_last_stage_sens also divides the loss DTensor by repeat_num.
    # (repeat_num equals TP for a TP-replicated scalar loss.)
    # To avoid double-scaling TP, we multiply TP back here.
    # This yields an effective scale of 1/(dp*cp)/grad_accum.
    # MoE/MTP losses are scaled separately by their own auto-scalers and are not affected here.
    main_loss_sense = get_loss_sense(
        parallelism=parallelism,
        enable_parallel=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        apply_gradient_accumulation=True,
        )
    loss_scale = float(main_loss_sense.asnumpy().item()) * int(parallel_dims.tp)
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
