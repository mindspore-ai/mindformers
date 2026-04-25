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
from mindspore import nn

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
from mindformers.pynative.base_models.gpt.gpt_model import GPTModel
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


def _setup_gpt_prefetch(
    embedding: nn.Cell,
    final_layernorm: nn.Cell,
    output_layer: nn.Cell,
    transformer_layers: List[nn.Cell],
) -> None:
    """Set up forward and backward prefetch chains for GPT FSDP modules.

    Args:
        embedding: The embedding module.
        final_layernorm: The final layer norm module.
        output_layer: The output projection module.
        transformer_layers: List of transformer layers.
    """
    if not transformer_layers:
        return

    if final_layernorm is None and output_layer is None:
        logger.warning(
            "No final_layernorm or output_layer found for prefetch setup. "
            "Prefetch chain will be incomplete."
        )
    if embedding is None:
        logger.warning("No embedding found for prefetch setup.")

    # --- Forward prefetch: embedding -> layer[0] -> ... -> [final_layernorm, output_layer] ---
    if embedding is not None and isinstance(embedding, HSDPModule):
        embedding.set_modules_to_forward_prefetch([transformer_layers[0]])

    next_layers = transformer_layers[1:] + [None]
    for layer, next_layer in zip(transformer_layers, next_layers):
        if not isinstance(layer, HSDPModule):
            continue
        if next_layer is not None:
            layer.set_modules_to_forward_prefetch([next_layer])
        elif final_layernorm is not None or output_layer is not None:
            layer.set_modules_to_forward_prefetch(
                [m for m in [final_layernorm, output_layer] if m is not None]
            )

    # --- Backward prefetch: output_layer -> layer[N-1] -> ... -> embedding ---
    reversed_layers = list(reversed(transformer_layers))
    prev_layers = reversed_layers[1:] + [None]

    last_tail = output_layer if output_layer is not None else final_layernorm
    if last_tail is not None and isinstance(last_tail, HSDPModule):
        last_tail.set_modules_to_backward_prefetch([reversed_layers[0]])

    for layer, prev_layer in zip(reversed_layers, prev_layers):
        if not isinstance(layer, HSDPModule):
            continue
        if prev_layer is not None:
            layer.set_modules_to_backward_prefetch([prev_layer])
        elif embedding is not None and isinstance(embedding, HSDPModule):
            layer.set_modules_to_backward_prefetch([embedding])


# ---------------------------------------------------------------------------
# GPT FSDP implementation
# ---------------------------------------------------------------------------

def apply_fsdp_to_gptmodel(
    model: nn.Cell,
    parallel_dims: Any,
    parallelism_config: Any,
) -> None:
    """Apply FSDP/HSDP to a GPT model.

    Directly operates on GPT model structure (embedding -> decoder.layers ->
    final_layernorm + output_layer -> root).

    Args:
        model: The GPTModel instance.
        parallel_dims: ParallelDims instance.
        parallelism_config: Parallelism configuration.
    """
    logger.info("Applying FSDP/HSDP to GPT model...")

    # --- Resolve dp_mesh ---
    if parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
        logger.info("Using HSDP mesh [dp_replicate, fsdp] (2D)")
    else:
        dp_mesh = parallel_dims.get_mesh("fsdp")
        logger.info("Using FSDP mesh [fsdp] (1D)")

    # --- Resolve dtypes ---
    param_dtype_str = getattr(parallelism_config, "param_dtype", None)
    reduce_dtype_str = getattr(parallelism_config, "reduce_dtype", None)

    # --- Build FSDP config ---
    reshard_policy = getattr(parallelism_config, "reshard_after_forward_policy", "default")
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_policy, parallel_dims.pp_enabled
    )

    cpu_offload = getattr(parallelism_config, "cpu_offload", False)
    fsdp_config = {
        "mesh": dp_mesh,
        "offload_policy": CPUOffloadPolicy() if cpu_offload else OffloadPolicy(),
    }

    if param_dtype_str is not None:
        param_dtype = _DTYPE_MAP.get(param_dtype_str, ms.bfloat16)
        reduce_dtype = _DTYPE_MAP.get(reduce_dtype_str, ms.float32) if reduce_dtype_str is not None else ms.float32
        fsdp_config["mp_policy"] = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)

    unwrapped_model = _unwrap_gptmodel(model)

    embedding = getattr(unwrapped_model, "embedding", None)
    layers = list(unwrapped_model.decoder.layers)
    final_layernorm = getattr(unwrapped_model.decoder, "final_layernorm", None)
    output_layer = getattr(unwrapped_model, "output_layer", None)

    if not layers:
        raise ValueError(f"{type(unwrapped_model).__name__} has no decoder layers")

    # --- 1. Wrap embedding ---
    if embedding is not None:
        fully_shard(embedding, **fsdp_config, reshard_after_forward=reshard_after_forward)

    # --- 2. Wrap transformer layers ---
    for layer in layers:
        fully_shard(layer, **fsdp_config, reshard_after_forward=reshard_after_forward)

    # --- 3. Wrap final_layernorm and output_layer together ---
    if final_layernorm is not None or output_layer is not None:
        # As an optimization, do not reshard_after_forward the last layers by default
        # since FSDP would prefetch them immediately after the forward pass.
        fully_shard(
            [m for m in [final_layernorm, output_layer] if m is not None],
            **fsdp_config,
            reshard_after_forward=(reshard_policy == "always"),
        )

    # --- 4. Wrap root ---
    fully_shard(model, **fsdp_config)

    # --- 5. Disable gradient division ---
    if getattr(parallelism_config, "disable_gradient_division", True):
        disable_fsdp_gradient_division(model)

    # --- 6. Prefetch chains ---
    _setup_gpt_prefetch(embedding, final_layernorm, output_layer, layers)

    if parallel_dims.dp_replicate_enabled:
        logger.info("Successfully applied HSDP (Hybrid Sharded Data Parallel)")
    else:
        logger.info("Successfully applied FSDP (Fully Sharded Data Parallel)")


# ---------------------------------------------------------------------------
# GPT parallelization entry point
# ---------------------------------------------------------------------------

def parallelize_gptmodel(
    model: nn.Cell,
    parallel_dims: Any,
    training_config: Any,
    parallelism_config: Any,
) -> nn.Cell:
    """Unified GPTModel parallelization entry point."""
    logger.info("Starting GPTModel parallelization for MCore architecture...")

    # Phase 1: TP
    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "Tensor Parallelism (TP) is not yet implemented for GPTModel. "
            "Please set tensor_parallel=1 in your config."
        )

    # Phase 2: CP
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "Context Parallelism (CP) is not yet implemented for GPTModel. "
            "Please set context_parallel=1 in your config."
        )

    # Phase 3: AC
    if getattr(training_config, "activation_checkpointing", False):
        raise NotImplementedError(
            "Activation Checkpointing (AC) is not yet implemented for GPTModel. "
            "Please set activation_checkpointing=false in your config."
        )

    # Phase 4: FSDP/HSDP
    if parallel_dims.fsdp_enabled:
        apply_fsdp_to_gptmodel(model, parallel_dims, parallelism_config)

    logger.info("GPTModel parallelization completed.")
    return model
