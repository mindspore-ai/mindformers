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

from typing import Any, List

import mindspore as ms
from mindspore import nn, Parameter
from mindspore.mint.distributed import get_world_size
from mindspore.ops.communication import set_comm_ops_inplace

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
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)
from mindformers.pynative.distributed.context_parallel import (
    apply_context_parallel_model_io,
    build_context_parallel_attention_style,
)
from mindformers.pynative.layers.layer_norm import FusedLayerNorm, FusedRMSNorm
from mindformers.pynative.distributed.tensor_parallel import NoParallel
from mindformers.pynative.distributed.expert_parallel import ExpertParallel, DeredundancyExpertParallel
from mindformers.pynative.distributed.parallelize import parallelize_module
from mindformers.pynative.distributed.activation_checkpoint import apply_ac
from mindformers.pynative.distributed.utils import distribute_module
from mindformers.pynative.base_models.gpt.gpt_model import GPTModel
from mindformers.pynative.transformers.moe.moe_utils import _MoEAuxLossAutoScaler
from mindformers.pynative.transformers.multi_token_prediction import _MTPLossAutoScaler
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
        hidden_state, expert_bias = args

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
    norm_cls = (FusedLayerNorm, FusedRMSNorm)
    for attr_name in ("input_layernorm", "pre_mlp_layernorm", "pre_cross_attn_layernorm"):
        norm = getattr(layer, attr_name, None)
        if isinstance(norm, norm_cls):
            norms[attr_name] = norm
    for attr_name in ("q_layernorm", "k_layernorm"):
        norm = getattr(layer.self_attention, attr_name, None)
        if isinstance(norm, norm_cls):
            norms[attr_name] = norm
    return norms


def _collect_layer_replicate_params(layer):
    """Collect non-module parameters that must be replicated (not sharded) under FSDP."""
    replicate_params = []
    if hasattr(layer.self_attention.core_attention, "max_logits_val"):
        replicate_params.append(layer.self_attention.core_attention.max_logits_val)

    if hasattr(layer.mlp, "tokens_per_expert"):
        replicate_params.append(layer.mlp.tokens_per_expert)

    if getattr(layer.mlp, "enable_expert_bias", False):
        replicate_params.append(layer.mlp.expert_bias)

    return replicate_params


def _collect_hc_replicate_params(hc_module):
    """Collect small HC params as replicate_params (avoid sharding issues with delayed init)."""
    replicate_params = []
    for attr in ("alpha_pre", "alpha_post", "alpha_res", "bias"):
        p = getattr(hc_module, attr, None)
        if p is not None:
            replicate_params.append(p)
    return replicate_params


def _apply_layers_tp(
    transformer_layer,
    tp_mesh,
    enable_ep,
    shard_plans,
):
    """Apply tensor-parallel sharding to a single GPT transformer layer."""

    sp_layout, attention_kernel_plan, norm_plan, rowwise_output_plan, rotary_emb_plan = shard_plans
    colwise_parallel, prepare_module_input, prepare_module_output, prepare_module_input_output = (
        ColwiseParallel,
        PrepareModuleInput,
        PrepareModuleOutput,
        PrepareModuleInputOutput,
    )

    distribute_param_plan = []
    layer_plan = {
        "input_layernorm": norm_plan,
        "self_attention": prepare_module_input(
            input_layouts=(sp_layout),
            desired_input_layouts=(
                Replicate()
            ),
            use_local_output=False,
        ),
        # NOTE: NoParallel() without local_output_grad_placements keeps the output as a
        # DTensor so that the intermediate results k is generated as a DTensor and its
        # gradient is correctly handled by the autograd engine.
        "self_attention.linear_qkv": colwise_parallel(output_layouts=Replicate(), use_local_output=False),
        "self_attention.linear_kvb": colwise_parallel(output_layouts=Replicate(), use_local_output=False),
        "self_attention.k_layernorm": NoParallel(use_local_output=False),
        # NOTE: use_local_output=True so that the inputs to FlexAttention are plain Tensors
        "self_attention.core_attention": attention_kernel_plan,
        "self_attention.linear_proj": rowwise_output_plan,
        "pre_mlp_layernorm": norm_plan,
        "self_attention.apply_rotary_emb": rotary_emb_plan,
        "self_attention.apply_rotary_emb2": rotary_emb_plan,
    }

    if transformer_layer.self_attention.q_rank == 0:
        raise ValueError(
            "q_rank should be greater than 0, but got 0!"
        )

    if transformer_layer.self_attention.config.q_lora_rank is not None:
        layer_plan.update(
            {
                "self_attention.linear_qb": colwise_parallel(output_layouts=Replicate(), use_local_output=False),
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
                    desired_input_layouts=(Replicate(),),
                    use_local_output=False,
                ),
                "mlp.linear_fc1": colwise_parallel(use_local_output=False),
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
            input_layouts=(Replicate(), Replicate()),
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
):
    """Apply tensor parallelism."""
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

    rowwise_parallel, prepare_module_input, prepare_module_output, prepare_module_input_output = (
        RowwiseParallel,
        PrepareModuleInput,
        PrepareModuleOutput,
        PrepareModuleInputOutput,
    )

    model_plan = {
        "model.embedding.word_embeddings": embed_plan,
        "model.decoder.final_layernorm": SequenceParallel(
            sequence_dim=0, use_local_output=False) if enable_sp else NoParallel(),
        "model.output_layer": ColwiseParallel(
            input_layouts=sp_layout,
            output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
            use_local_output=False,
        ),

    }

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
        decoder_layers[0].self_attention.config, "input_layout", None
    ) if decoder_layers else None
    attn_qkv_shard = Shard(1) if attn_input_layout == "TND" else Shard(2)
    attention_kernel_plan = prepare_module_input(
        input_layouts=(Replicate(), Replicate(), Replicate(), None),
        desired_input_layouts=(attn_qkv_shard, attn_qkv_shard, attn_qkv_shard, None),
        use_local_output=False,
    )
    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    norm_plan = SequenceParallel(sequence_dim=0, use_local_output=False) if enable_sp else NoParallel()
    rowwise_output_plan = rowwise_parallel(
        output_layouts=sp_layout, use_local_output=False,  # use_local_output=enable_sp
    )
    rotary_emb_plan = prepare_module_input_output(
        input_layouts=(Replicate(), None),
        desired_input_layouts=(Replicate(), None),
        output_layouts=(Replicate(),),
        desired_output_layouts=(Replicate(),),
        use_local_output=False,
    )

    for transformer_layer in model.model.decoder.layers:
        _apply_layers_tp(
            transformer_layer,
            tp_mesh,
            enable_ep,
            (sp_layout, attention_kernel_plan, norm_plan, rowwise_output_plan, rotary_emb_plan,),
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
                (sp_layout, attention_kernel_plan, norm_plan, rowwise_output_plan, rotary_emb_plan,),
            )

    if hasattr(model.model.loss, "log_softmax") and hasattr(model.model.loss, "nll_loss"):
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
        "Tensor Parallelism to the model"
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


# ---------------------------------------------------------------------------
# GPT FSDP implementation
# ---------------------------------------------------------------------------

def apply_fsdp(
        model: nn.Cell,
        parallel_dims: Any,
        parallelism: Any,
        accumulate_allreduce_grads_in_fp32: bool = True,
) -> None:
    """Apply FSDP/HSDP to a GPT model.

    Directly operates on GPT model structure (embedding -> decoder.layers ->
    final_layernorm + output_layer -> root).

    Args:
        model: The GPTModel instance.
        parallel_dims: ParallelDims instance.
        parallelism: Parallelism configuration.
    """
    logger.info("Applying FSDP/HSDP to GPT model...")

    # Unwrap to GPTModel
    gpt_model = _unwrap_gptmodel(model)

    # --- Resolve dp_mesh ---
    if parallel_dims.cp_enabled:
        dp_mesh = parallel_dims.world_mesh["fsdp"]
        logger.info(
            "Using FSDP mesh with CP enabled: dp_shard=%s, cp=%s, fsdp=%s, fsdp_rank_list=%s",
            parallel_dims.dp_shard,
            parallel_dims.cp,
            parallel_dims.fsdp,
            tuple(getattr(dp_mesh, "rank_list", ())),
        )
    elif parallel_dims.dp_replicate_enabled:
        # change `dp_shard` to `fsdp` if hyper-parallel support flatten
        dp_mesh = parallel_dims.get_mesh(["dp_replicate", "dp_shard"])
        logger.info("Using HSDP mesh [dp_replicate, fsdp] (2D)")
    else:
        dp_mesh = parallel_dims.get_mesh("dp_shard")
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
    reduce_dtype = ms.float32 if accumulate_allreduce_grads_in_fp32 else None
    fsdp_config = {
        "mesh": dp_mesh,
        "offload_policy": CPUOffloadPolicy() if cpu_offload else OffloadPolicy(),
        "mp_policy": MixedPrecisionPolicy(reduce_dtype=reduce_dtype),
    }
    efsdp_config = {
        "mesh": edp_mesh,
        "offload_policy": CPUOffloadPolicy() if cpu_offload else OffloadPolicy(),
        "mp_policy": MixedPrecisionPolicy(reduce_dtype=reduce_dtype),
    }

    modules = {
        "embedding": getattr(gpt_model, "embedding", None),
        "layers": list(gpt_model.decoder.layers),
    }

    if getattr(gpt_model, "mtp"):
        modules.update(
            {
                "tail_modules": [
                    m for m in [
                        getattr(gpt_model.decoder, "final_layernorm", None),
                    ] if m is not None
                ],
                "mtp": getattr(gpt_model, "mtp", None),
            }
        )
    else:
        modules.update(
            {
                "tail_modules": [
                    m for m in [
                        getattr(gpt_model.decoder, "final_layernorm", None),
                        getattr(gpt_model, "output_layer", None),
                    ] if m is not None
                ],
            }
        )

    embedding = modules["embedding"]
    layers = modules["layers"]
    tail_modules = modules["tail_modules"]
    mtp = modules.get("mtp")

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

        # 2a'. Wrap MoE router (and shared-expert gate) independently. Their weights are
        # stored in moe_router_dtype (fp32) for gradient precision, so they must NOT share
        # an HSDP param-group with the layer's bf16 params — fully_shard asserts a single
        # original dtype per group. Same treatment as the fp32 embedding wrapped above.
        router = getattr(layer.mlp, "router", None)
        if router is not None:
            with ms.DeviceCtx("meta"):
                fully_shard(router, **fsdp_config, reshard_after_forward=reshard_after_forward)
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
        def shard_plan(param):
            if len(param.shape) == 1:
                return Shard(0)
            return Shard(1)

        if hasattr(layer, "attn_hc"):
            with ms.DeviceCtx("meta"):
                fully_shard(layer.attn_hc, **fsdp_config, shard_placement_fn=shard_plan,
                            reshard_after_forward=reshard_after_forward,
                            replicate_params=_collect_hc_replicate_params(layer.attn_hc))
        if hasattr(layer, "ffn_hc"):
            with ms.DeviceCtx("meta"):
                fully_shard(layer.ffn_hc, **fsdp_config, shard_placement_fn=shard_plan,
                            reshard_after_forward=reshard_after_forward,
                            replicate_params=_collect_hc_replicate_params(layer.ffn_hc))

        # 2d. Wrap the transformer layer (large module)
        replicate_params = _collect_layer_replicate_params(layer)
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
        with ms.DeviceCtx("meta"):
            for tm in tail_modules:
                if isinstance(tm, (FusedLayerNorm, FusedRMSNorm)):
                    # Small module: wrap norm independently
                    with ms.DeviceCtx("meta"):
                        fully_shard(tm, **fsdp_config)
                else:
                    # Larger module: output_layer, do not reshard_after_forward by default
                    # since FSDP would prefetch it immediately after the forward pass.
                    with ms.DeviceCtx("meta"):
                        fully_shard(tm,
                                    **fsdp_config,
                                    reshard_after_forward=reshard_policy == "always")

    if mtp:
        for layer in mtp.layers:
            if hasattr(layer.transformer_layer.mlp, "experts") and edp_mesh is not None:
                with ms.DeviceCtx("meta"):
                    fully_shard(
                        layer.transformer_layer.mlp.experts, **efsdp_config,
                        reshard_after_forward=reshard_after_forward,
                    )

            mtp_replicate_params = _collect_layer_replicate_params(layer.transformer_layer)
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
                setattr(model_config, "_mf_runtime_flash_attention_head_num", num_heads // ulysses_degree)

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
        accumulate_allreduce_grads_in_fp32: bool = True,
) -> nn.Cell:
    """Unified GPTModel parallelization entry point."""
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
            )

    # Phase 2: EP
    if parallel_dims.ep_enabled:
        ep_mesh = parallel_dims.get_mesh("ep")
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
            cp_mesh=parallel_dims.world_mesh["cp"],
            parallel_dims=parallel_dims,
            parallelism=parallelism,
        )

    # Phase 4: AC
    if recompute.mode != "None" or recompute_comm.enable or swap.enable:
        apply_ac(
            _unwrap_gptmodel(model).decoder,
            recompute,
            recompute_comm,
            swap,
        )

    # Phase 5: FSDP/HSDP
    if parallel_dims.fsdp_enabled:
        apply_fsdp(model, parallel_dims, parallelism, accumulate_allreduce_grads_in_fp32)

    for param_name, param in model.parameters_and_names():
        if isinstance(param, DTensor):
            logger.debug(param_name, param.layout, param.placements)
        else:
            logger.debug(f"{param_name} is not DTensor, shape is {param.shape}")

    # Phase 6: CP root I/O hooks. Apply after FSDP so the hooks live on the final
    # model boundary called by trainer.
    if parallel_dims.cp_enabled:
        apply_context_parallel_model_io(model, parallel_dims, parallelism)

    # Set the loss scale for the auxiliary loss of the MoE layer.
    main_loss_sense = 1. / (get_world_size() / int(parallelism.pipeline_parallel))
    _MoEAuxLossAutoScaler.set_loss_scale(main_loss_sense)
    _MTPLossAutoScaler.set_loss_scale(main_loss_sense)

    logger.info("GPTModel parallelization completed.")
    return model
