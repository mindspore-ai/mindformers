# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Modification points:
# 1. Migrated num_floating_point_operations function from Megatron-LM's training.py
#    for FLOPs calculation of transformer models.
# 2. Migrated is_linear_attention_variant helper function from Megatron-LM's training.py
#    for checking linear attention variants.
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
"""Check Model Input Config."""
import json
import os
from dataclasses import dataclass
from functools import wraps
from typing import Union, Optional, Dict
import mindspore.common.dtype as mstype
import mindspore as ms
from mindspore import nn
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.communication import get_rank, get_group_size

from mindformers.parallel_core.transformer_config import TransformerConfig
from ..tools.utils import get_predict_run_mode, is_pynative, get_output_root_path
from ..version_control import get_lazy_inline, get_predict_lazy_inline
from ..tools.logger import logger

# pylint: disable=W0212
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "mindspore_model.ckpt"
WEIGHTS_INDEX_NAME = "mindspore_model.ckpt.index.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
PROCESSOR_NAME = "processor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
MAX_INT32 = 2147483647
DEFAULT_CHECKPOINT_SAVE_FOLDER = os.path.join(get_output_root_path(), 'checkpoint_save')

str_to_ms_type = {
    "float16": mstype.float16,
    "float32": mstype.float32,
    "bfloat16": mstype.bfloat16,
    "int8": mstype.int8
}

format_type = {
    "nz": 29,
}

def get_current_rank_stage():
    """get current pipeline stage."""
    pipeline_stages = ms.get_auto_parallel_context('pipeline_stages')
    rank_id = get_rank()
    device_num = get_group_size()
    per_stage_device_num = device_num // pipeline_stages
    return rank_id // per_stage_device_num


def get_model_parameters(cell: nn.Cell, only_trainable=True):
    """get all parameters in cell."""
    params = []
    for _, sub_cell in cell.cells_and_names():
        if isinstance(sub_cell, nn.Cell):
            for param in sub_cell.trainable_params():
                params.append(param)
            if not only_trainable:
                for param in sub_cell.untrainable_params():
                    params.append(param)
    return params

def is_current_pipeline_stage(layer: nn.Cell, current_pipeline_stage):
    """judge the layer belongs to the current pipeline state."""
    if not hasattr(layer, "pipeline_stage"):
        raise ValueError(f"You should set the pipeline_stage for the {type(layer)}")
    if current_pipeline_stage == layer.pipeline_stage:
        return True
    return False

def convert_mstype(ms_type: str = "float16"):
    """Convert the string type to MindSpore type."""
    if isinstance(ms_type, mstype.Float):
        return ms_type
    ms_type = str(ms_type).lower()
    if ms_type == "float16":
        return mstype.float16
    if ms_type == "float32":
        return mstype.float32
    if ms_type == "bfloat16":
        return mstype.bfloat16
    if ms_type == "int8":
        return mstype.int8
    raise KeyError(f"Supported data type keywords include: "
                   f"[float16, float32, bfloat16, int8], but get {ms_type}")


def reverse_dict(d: dict):
    new_d = {}
    for k, v in d.items():
        if v in new_d:
            raise ValueError("Different keys in dict have same values.")
        new_d[v] = k
    return new_d


def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False


def check_fine_grain_interleave_valid(fine_grain_interleave, parallel_config):
    """Check the fine grain interleave condition"""
    if fine_grain_interleave is None or parallel_config is None:
        return False
    return fine_grain_interleave > 1 and parallel_config.model_parallel > 1


def check_use_3d_tensor_parallel_valid(config):
    """Check the use_3d_tensor_parallel condition"""
    use_3d_tensor_parallel = getattr(config, "use_3d_tensor_parallel", False)
    is_config_valid = config is not None and config.parallel_config is not None
    if not use_3d_tensor_parallel or not is_config_valid:
        return False
    if not config.use_flash_attention:
        raise ValueError("When the use_3d_tensor_parallel = True, the use_flash_attention must be True ")
    if config.parallel_config.get_ulysses_cp_num() > 1:
        raise ValueError("Currently, when the use_3d_tensor_parallel = True, "
                         "the cp_ds of the ulysses context parallel must be 1")
    if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
        raise ValueError("Currently, when the use_3d_tensor_parallel = True, the auto parallel is not supported")
    if config.moe_config is not None and config.moe_config.expert_num > 1:
        raise ValueError("Currently, when the use_3d_tensor_parallel = True, the MoE is not supported")
    if not config.parallel_config.use_seq_parallel:
        raise ValueError("Currently, when the use_3d_tensor_parallel = True, the use_seq_parallel must be True")
    if check_fine_grain_interleave_valid(config.fine_grain_interleave, config.parallel_config):
        raise ValueError("Currently, when the use_3d_tensor_parallel = True, "
                         "the fine_grain_interleave is not supported")
    tp_x = getattr(config, "tp_x", 1)
    tp_y = getattr(config, "tp_y", 1)
    tp_z = getattr(config, "tp_z", 1)
    model_parallel = config.parallel_config.model_parallel
    if model_parallel > 1 and tp_x * tp_y * tp_z != config.parallel_config.model_parallel:
        raise ValueError(f"tp_x * tp_y * tp_z should be equal to model_parallel, but got "
                         f"tp_x={tp_x}, tp_y={tp_y}, tp_z={tp_z}, model_parallel={model_parallel}.")
    if model_parallel > 1:
        logger.info(f"use_3d_tensor_parallel is True, (tp_x, tp_y, tp_z): ({tp_x}, {tp_y}, {tp_z})")
        return True
    return False


def check_swap_enabled(swap_config):
    if isinstance(swap_config, dict):
        return swap_config["swap"]
    return swap_config.swap


def jit(func):
    """jit decorator."""

    @wraps(func)
    def decorator(*args, **kwargs):
        if not get_predict_run_mode():
            raise ValueError("Jit is only supported in predict mode now.")
        if is_pynative():
            return func(*args, **kwargs)
        return ms.jit(func, jit_level='O0', infer_boost='on')(*args, **kwargs)

    return decorator


def dict_from_json_file(json_file: Union[str, os.PathLike]):
    """method to read json."""
    if not os.path.exists(json_file):
        raise ValueError(
            f"{json_file} does not exist. Please check files in given path."
        )
    json_file = os.path.realpath(json_file)
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)

ms_type_to_str = reverse_dict(str_to_ms_type)

lazy_inline = get_lazy_inline
predict_lazy_inline = get_predict_lazy_inline


def is_hybrid_model(args):
    """Returns True if the model is a hybrid Mamba-Transformer model."""
    return args.hybrid_layer_pattern is not None


def is_linear_attention_variant(experimental_attention_variant):
    """Check if the experimental attention variant is a linear attention variant."""
    linear_attention_variants = ["gated_delta_net"]
    return experimental_attention_variant in linear_attention_variants


def convert_transformer_config_to_args_for_tflops(
        config: TransformerConfig, seq_length: Optional[int] = None
) -> TransformerConfig:
    """Get the transformer config for tflops calculation."""
    if seq_length is not None:
        config.seq_length = seq_length

    default_args = {
        'hybrid_layer_pattern': None,
        'attention_output_gate': False,
        'experimental_attention_variant': None,
        'linear_attention_freq': None,
        'linear_key_head_dim': 128,
        'linear_value_head_dim': 128,
        'linear_num_key_heads': 16,
        'linear_num_value_heads': 32,
        'linear_conv_kernel_dim': 128,
        'mamba_state_dim': 128,
        'mamba_head_dim': 64,
        'mamba_num_groups': 8,
        'mamba_num_heads': None,
        'group_query_attention': False,
        'swiglu': False,
        'padded_vocab_size': getattr(config, 'vocab_size', None),
        'num_experts': getattr(config, 'num_moe_experts', None),
        'moe_latent_size': None,
    }
    for key, value in default_args.items():
        if not hasattr(config, key):
            setattr(config, key, value)

    if config.num_attention_heads != config.num_query_groups and not config.multi_latent_attention:
        config.group_query_attention = True

    if config.hidden_act in ['silu', 'swiglu']:
        config.swiglu = True

    return config


class Symbols:
    """Symbols for different layer types and pattern separators."""

    MAMBA = "M"
    ATTENTION = "*"
    MLP = "-"
    MOE = 'E'
    PIPE = '|'
    MTP_SEPARATOR = "/"
    VALID_LAYERS = {MAMBA, ATTENTION, MLP, MOE}


@dataclass
class ParsedHybridPattern:
    """Result of parsing a unified hybrid pattern string.

    A unified pattern encodes both the main decoder pattern and the MTP pattern
    in a single string using "/" as a separator. The main pattern may also
    contain "|" pipe symbols to define pipeline stage boundaries for flexible
    virtual pipeline parallelism (fVPP).

    Format: "<main_pattern>/<mtp_pattern>/<mtp_pattern>/..."

    Examples:
        - "M*M*" -> main="M*M*", mtp=None, depths=0 (no MTP)
        - "M*M*/MM/MM" -> main="M*M*", mtp="MM", depths=2
        - "MMMM/*M/*M/*M" -> main="MMMM", mtp="*M", depths=3
        - "M-M-|M-M*-/MM/MM" -> main="M-M-|M-M*-" (2 PP stages), mtp="MM", depths=2

    The "/" symbol introduces MTP patterns. Each repeated pattern after the main
    decoder represents one MTP prediction depth.

    The "|" symbol in the main pattern defines pipeline stage boundaries.

    Attributes:
        main_pattern: The main decoder layer pattern (e.g., "M*M*" or "M-M-|M-M*-")
        mtp_pattern: The MTP layer pattern per depth (e.g., "MM"), or None if no MTP
        mtp_num_depths: Number of MTP prediction depths (0 if no MTP)
    """

    main_pattern: Optional[str]
    mtp_pattern: Optional[str]
    mtp_num_depths: int


def get_hybrid_layer_counts(pattern: str) -> Dict[str, int]:
    """Count layers by type across the full hybrid pattern (main + MTP).

    Parses the pattern to extract main and MTP components, then counts
    each layer type. Main pattern '|' separators are skipped. MTP layers
    are counted once per MTP depth.

    Args:
        pattern: Full hybrid layer pattern string.

    Returns:
        Dictionary mapping layer symbol to count. Keys are Symbols.ATTENTION,
        Symbols.MAMBA, Symbols.MLP, and Symbols.MOE.

    Examples:
        >>> get_hybrid_layer_counts("M*M*")
        {'*': 2, 'M': 2, '-': 0, 'E': 0}

        >>> get_hybrid_layer_counts("M-M-|M-M*-/MM/MM")
        {'*': 1, 'M': 8, '-': 4, 'E': 0}
    """
    parsed = parse_hybrid_pattern(pattern)
    counts = {Symbols.ATTENTION: 0, Symbols.MAMBA: 0, Symbols.MLP: 0, Symbols.MOE: 0}

    # Count main decoder layers (skip '|' pipe separators)
    if parsed.main_pattern:
        for char in parsed.main_pattern:
            if char in counts:
                counts[char] += 1

    # Count MTP layers (pattern repeated mtp_num_depths times)
    if parsed.mtp_pattern and parsed.mtp_num_depths > 0:
        for char in parsed.mtp_pattern:
            if char in counts:
                counts[char] += parsed.mtp_num_depths

    return counts


def parse_hybrid_pattern(pattern: Optional[str]) -> ParsedHybridPattern:
    """Parse a unified hybrid pattern string into main and MTP components.

    The pattern uses "/" as a separator between the main decoder pattern and
    MTP patterns. Each MTP pattern after the separator represents one prediction
    depth. The main pattern may contain "|" pipe symbols for pipeline stage
    boundaries.

    Format: "<main_pattern>/<mtp_pattern>/<mtp_pattern>/..."

    Args:
        pattern: Unified pattern string, e.g., "M*M*/MM/MM" or just "M*M*"

    Returns:
        ParsedHybridPattern with main_pattern, mtp_pattern, and mtp_num_depths

    Raises:
        ValueError: If MTP patterns are inconsistent (all must be identical)
        ValueError: If pattern contains invalid layer symbols

    Examples:
        >>> parse_hybrid_pattern("M*M*")
        ParsedHybridPattern(main_pattern="M*M*", mtp_pattern=None, mtp_num_depths=0)

        >>> parse_hybrid_pattern("M*M*/MM/MM")
        ParsedHybridPattern(main_pattern="M*M*", mtp_pattern="MM", mtp_num_depths=2)

        >>> parse_hybrid_pattern("MMMM/*M/*M/*M")
        ParsedHybridPattern(main_pattern="MMMM", mtp_pattern="*M", mtp_num_depths=3)

        >>> parse_hybrid_pattern("M-M-|M-M*-/MM/MM")
        ParsedHybridPattern(main_pattern="M-M-|M-M*-", mtp_pattern="MM", mtp_num_depths=2)
    """
    if pattern is None:
        return ParsedHybridPattern(main_pattern=None, mtp_pattern=None, mtp_num_depths=0)

    parts = pattern.split(Symbols.MTP_SEPARATOR)

    if len(parts) == 1:
        # No MTP separator found - pattern is main decoder only
        main_pattern = parts[0]
        _validate_pattern(main_pattern, "main", allow_pipe=True)
        return ParsedHybridPattern(main_pattern=main_pattern, mtp_pattern=None, mtp_num_depths=0)

    # First part is main decoder pattern
    main_pattern = parts[0]
    if main_pattern:
        _validate_pattern(main_pattern, "main", allow_pipe=True)

    # Remaining parts are MTP patterns (one per depth)
    mtp_parts = parts[1:]

    if not mtp_parts or all(p == "" for p in mtp_parts):
        # No MTP patterns after separator
        return ParsedHybridPattern(
            main_pattern=main_pattern if main_pattern else None, mtp_pattern=None, mtp_num_depths=0
        )

    # Validate all MTP patterns are identical
    mtp_pattern = mtp_parts[0]
    for i, part in enumerate(mtp_parts[1:], start=2):
        if part != mtp_pattern:
            raise ValueError(
                f"All MTP patterns must be identical. "
                f"Pattern 1 is '{mtp_pattern}', but pattern {i} is '{part}'. "
                f"Full pattern: '{pattern}'"
            )

    _validate_pattern(mtp_pattern, "MTP", allow_pipe=False)

    return ParsedHybridPattern(
        main_pattern=main_pattern if main_pattern else None,
        mtp_pattern=mtp_pattern,
        mtp_num_depths=len(mtp_parts),
    )


def _validate_pattern(pattern: str, pattern_name: str, allow_pipe: bool = False) -> None:
    """Validate that a pattern contains only valid layer symbols.

    Args:
        pattern: Layer pattern string to validate
        pattern_name: Name of pattern for error messages (e.g., "main" or "MTP")
        allow_pipe: Whether to allow the pipe '|' separator (for main patterns)

    Raises:
        ValueError: If pattern contains invalid symbols
    """
    valid_chars = Symbols.VALID_LAYERS | {Symbols.PIPE} if allow_pipe else Symbols.VALID_LAYERS
    for char in pattern:
        if char not in valid_chars:
            raise ValueError(
                f"In {pattern_name} pattern, '{char}' is not a valid layer symbol. "
                f"Valid symbols are: {valid_chars}"
            )


def num_floating_point_operations(args, batch_size):
    def mlp_layer_flops(batch_size, seq_len, hidden_size, expansion=4.0, swiglu=False):
        """Calculate FLOPs for an MLP layer."""
        scale_factor = 3.0 / 2.0 if swiglu else 1.0
        return 4 * expansion * scale_factor * batch_size * seq_len * hidden_size**2

    def moe_layer_flops(batch_size, seq_len, hidden_size, moe_ffn_hidden_size,
                        shared_expert_ffn_hidden_size, num_experts_routed_to,
                        moe_latent_size=None, swiglu=False):
        """Calculate FLOPs for an MoE layer."""
        scale_factor = 3.0 / 2.0 if swiglu else 1.0
        if moe_latent_size is None:
            routed_flops = (4 * batch_size * seq_len * hidden_size *
                            moe_ffn_hidden_size * num_experts_routed_to * scale_factor)
        else:
            # Routed experts run on moe_latent_size.
            routed_flops = (4 * batch_size * seq_len * moe_latent_size *
                            moe_ffn_hidden_size * num_experts_routed_to * scale_factor)
            # Up proj and down proj.
            routed_flops += (4 * batch_size * seq_len * hidden_size * moe_latent_size)
        shared_flops = 4 * batch_size * seq_len * hidden_size * shared_expert_ffn_hidden_size * scale_factor
        return routed_flops + shared_flops

    def attn_layer_flops(
        batch_size, seq_len, hidden_size, num_heads, gqa=True, gqa_groups=8, kv_channels=None
    ):
        """Calculate FLOPs for an attention layer."""
        p = (kv_channels * num_heads / hidden_size) if kv_channels else 1
        g = gqa_groups if gqa else num_heads
        return (
            4
            * batch_size
            * seq_len
            * hidden_size
            * p
            * (hidden_size + (hidden_size * (g / num_heads)) + (seq_len / 2))
        )

    def mamba_layer_flops(batch_size, seq_len, hidden_size, state_dim=16,
                          head_dim=64, num_groups=1, num_heads=128):
        """Calculate FLOPs for a Mamba layer."""
        # Note (rwaleffe): flops estimate for scan should be updated based on new SSD kernels,
        # but small percent of overall layer flops
        d_in = 2 * hidden_size
        if num_heads:
            nheads = num_heads
        else:
            nheads = d_in // head_dim
        return (
            (
                2
                * batch_size
                * seq_len
                * hidden_size
                * (2 * d_in + 2 * num_groups * state_dim + nheads)
            )  # in_proj
            + (7 * batch_size * seq_len * d_in * state_dim)  # scan
            + (2 * batch_size * seq_len * d_in * hidden_size)  # out_proj
        )

    def hybrid_flops(batch_size, seq_len, hidden_size,
                     num_attn_layers, num_mamba_layers, num_mlp_layers, num_moe_layers,
                     mamba_state_dim=128, mamba_head_dim=64,
                     mamba_num_groups=8, mamba_num_heads=128,
                     num_attn_heads=32, gqa=True,
                     gqa_groups=8, kv_channels=None,
                     mlp_expansion=4.0, swiglu=False,
                     moe_latent_size=None,
                     moe_ffn_hidden_size=2048, shared_expert_ffn_hidden_size=2048, num_experts_routed_to=1,
                     vocab_size=256000, mtp_num_layers=0):
        """Calculate total FLOPs for the hybrid model."""
        flops_fwd = (
                num_attn_layers * attn_layer_flops(batch_size, seq_len, hidden_size,
                                                   num_attn_heads, gqa, gqa_groups, kv_channels) +
                num_mlp_layers * mlp_layer_flops(batch_size, seq_len, hidden_size,
                                                 mlp_expansion, swiglu) +
                num_mamba_layers * mamba_layer_flops(batch_size, seq_len, hidden_size,
                                                     mamba_state_dim, mamba_head_dim,
                                                     mamba_num_groups, mamba_num_heads) +
                num_moe_layers * moe_layer_flops(batch_size, seq_len, hidden_size, moe_ffn_hidden_size,
                                                 shared_expert_ffn_hidden_size, num_experts_routed_to,
                                                 moe_latent_size, swiglu) +
                (2 * batch_size * seq_len * hidden_size * vocab_size * (1 + mtp_num_layers))  # logits computation
        )
        return flops_fwd * 3

    def transformer_flops():
        """Calculate FLOPs for a standard Transformer model."""
        # Group Query Attention.
        if not args.group_query_attention:
            args.num_query_groups = args.num_attention_heads
        # MoE.
        if args.num_experts is None:
            # Every Transformer MLP is dense.
            num_dense_layers = args.num_layers
            num_moe_layers = 0
            num_experts_routed_to = 0
            last_layer_is_moe = 0
        else:
            # Calculate number of dense and MoE Transformer MLPs.
            if isinstance(args.moe_layer_freq, int):
                moe_layer_pattern = [
                    1 if (i % args.moe_layer_freq == 0) else 0 for i in range(args.num_layers)
                ]
            elif isinstance(args.moe_layer_freq, list):
                moe_layer_pattern = args.moe_layer_freq
            else:
                raise RuntimeError("Illegal --moe-layer-freq argument provided!")
            assert len(moe_layer_pattern) == args.num_layers, (
                f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
                f"expected {args.num_layers}, "
                f"current moe layer pattern: {args.moe_layer_freq}"
            )
            num_moe_layers = sum(moe_layer_pattern)  # Number of 1s in `moe_layer_pattern`.
            num_dense_layers = args.num_layers - num_moe_layers
            num_experts_routed_to = args.moe_router_topk
            last_layer_is_moe = moe_layer_pattern[-1]

        if args.mtp_num_layers is not None:
            mtp_num_layers = args.mtp_num_layers
            num_moe_layers += last_layer_is_moe * mtp_num_layers
            num_dense_layers += (1 - last_layer_is_moe) * mtp_num_layers
            num_layers = args.num_layers + mtp_num_layers
        else:
            mtp_num_layers = 0
            num_layers = args.num_layers

        moe_ffn_hidden_size = (
            args.moe_ffn_hidden_size
            if args.moe_ffn_hidden_size is not None
            else args.ffn_hidden_size
        )
        moe_latent_size = args.moe_latent_size
        shared_expert_ffn_hidden_size = (
            0
            if args.moe_shared_expert_intermediate_size is None
            else args.moe_shared_expert_intermediate_size
        )

        # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
        #       backward wgrad [weight gradient], backward dgrad [data gradient]).
        forward_backward_expansion_factor = 3
        # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
        fma_expansion_factor = 2
        # - 3x (SwiGLU enabled): h->2*ffn_h GEMM and ffn_h->h GEMM are stacked.
        # - 2x (SwiGLU disabled): h->ffn_h GEMM and ffn_h->h GEMM are stacked.
        ffn_expansion_factor = 3 if args.swiglu else 2

        if args.multi_latent_attention:
            assert not args.group_query_attention
            # Basic arithmetic
            # let B is batch size, s is seq_len, h is embedding dim,
            # for one self_attnetion block (prenorm is not included)
            # qkv projection:  6Bsh^2
            # attn:            2Bs^2h
            # attn over value: 2Bs^2h
            # oproj:           2Bsh^2
            #
            # references
            # https://arxiv.org/abs/2305.10403
            # https://arxiv.org/abs/2205.05198
            ## MLA
            if args.q_lora_rank is None:
                q_term = (
                    args.hidden_size
                    * args.num_attention_heads
                    * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                )
            else:
                q_term = args.q_lora_rank * (
                    args.hidden_size
                    + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                    + 1
                )
            standard_self_attn_term = (
                forward_backward_expansion_factor
                * fma_expansion_factor
                * (
                    ## q lora + rope + q norm
                    q_term
                    ## kv lora + rope + kv norm
                    + args.kv_lora_rank
                    * (
                        args.hidden_size
                        + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim)
                        + 1
                    )
                    + args.hidden_size * args.qk_pos_emb_head_dim
                    ## o proj
                    + (args.num_attention_heads * args.v_head_dim) * args.hidden_size
                    ## core attn
                    + args.seq_length
                    * (args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim))
                    / 2  # causal mask (only half of the mask is non-zero)
                    + args.seq_length * args.num_attention_heads * args.v_head_dim / 2
                )
            )

        else:
            ## MHA or GQA
            query_projection_size = args.kv_channels * args.num_attention_heads
            key_projection_size = args.kv_channels * args.num_query_groups
            value_projection_size = args.kv_channels * args.num_query_groups
            gate_projection_size = query_projection_size if args.attention_output_gate else 0
            standard_self_attn_term = (
                forward_backward_expansion_factor
                * fma_expansion_factor
                * (
                    ## qkv proj
                    args.hidden_size
                    * (
                        query_projection_size
                        + key_projection_size
                        + value_projection_size
                        + gate_projection_size
                    )
                    ## core attention
                    + query_projection_size
                    * args.seq_length
                    / 2  # causal mask (only half of the mask is non-zero)
                    * 2  # QK^T and (QK^T)V
                    ## out proj
                    + query_projection_size
                    * args.hidden_size
                )
            )

        if is_linear_attention_variant(args.experimental_attention_variant):
            # Calculate number of dense and MoE Transformer MLPs.
            if isinstance(args.linear_attention_freq, int):
                linear_attention_pattern = [
                    # [1,1,...,1,0,1,1,...,1,0,...]
                    0 if ((i + 1) % args.linear_attention_freq == 0)
                    else 1 for i in range(num_layers)
                ]
            elif isinstance(args.linear_attention_freq, list):
                linear_attention_pattern = args.linear_attention_freq
                assert len(linear_attention_pattern) == num_layers, (
                    f"Invalid length of linear_attention_pattern: {len(linear_attention_pattern)}, "
                    f"expected {num_layers}, "
                    f"current linear attention pattern: {args.linear_attention_freq}"
                )
            elif args.linear_attention_freq is None:
                # This should be caught by config validation, but raise here as a safety check
                raise ValueError(
                    f"Linear attention type {args.experimental_attention_variant} is specified "
                    "but linear_attention_freq is None. "
                    "Please set linear_attention_freq to specify the LA/SDPA layer pattern."
                )
            else:
                raise ValueError(
                    f"Invalid linear_attention_freq: {type(args.linear_attention_freq)},"
                    f" {args.linear_attention_freq}"
                )
            num_linear_attention_layers = sum(linear_attention_pattern)
            num_standard_attention_layers = num_layers - num_linear_attention_layers

            if args.experimental_attention_variant == "gated_delta_net":
                # Calculate the FLOPs for the gated delta net attention.
                qk_head_dim = args.linear_key_head_dim
                v_head_dim = args.linear_value_head_dim
                num_qk_heads = args.linear_num_key_heads
                num_v_heads = args.linear_num_value_heads
                qk_dim = qk_head_dim * num_qk_heads
                v_dim = v_head_dim * num_v_heads
                linear_self_attn_term = (
                    forward_backward_expansion_factor
                    * fma_expansion_factor
                    * (
                        ## in proj
                        args.hidden_size
                        * (2 * qk_dim + 2 * v_dim + 2 * num_v_heads)
                        ## conv1d
                        + args.linear_conv_kernel_dim
                        * (2 * qk_dim + v_dim)
                        ## gated delta rule
                        + num_v_heads
                        * (v_head_dim ** 2)
                        * 4  # KK^T, VK^T, S(a(I-bKK^T)), and SQ
                        ## out proj
                        + args.hidden_size
                        * v_dim
                    )
                )
            else:
                raise ValueError(
                    "Invalid experimental_attention_variant: "
                    f"{args.experimental_attention_variant}"
                )
        else:
            num_linear_attention_layers = 0
            linear_self_attn_term = 0
            num_standard_attention_layers = num_layers

        self_attn_term = (
            linear_self_attn_term * num_linear_attention_layers
            + standard_self_attn_term * num_standard_attention_layers
        )

        total_floating_point_operations = (
            batch_size
            * args.seq_length
            * (
                # MLP
                forward_backward_expansion_factor
                * fma_expansion_factor
                * args.hidden_size
                * (
                    # dense layer (deepseek v2, v3 style)
                    (args.ffn_hidden_size * ffn_expansion_factor)
                    * num_dense_layers
                    # routed experts
                    + (
                        (moe_ffn_hidden_size * num_experts_routed_to * ffn_expansion_factor)
                        if moe_latent_size is None
                        else (
                            (
                                moe_ffn_hidden_size
                                * num_experts_routed_to
                                * ffn_expansion_factor
                                * moe_latent_size
                                / args.hidden_size
                            )  # Routed experts run on moe_latent_size.
                            + 2 * moe_latent_size  # Up proj and down proj.
                        )
                    )
                    * num_moe_layers
                    # Shared Experts.
                    + (shared_expert_ffn_hidden_size * ffn_expansion_factor)
                    * num_moe_layers
                )
                # Self Attention
                + self_attn_term
                # MTP norms and proj
                + forward_backward_expansion_factor
                * fma_expansion_factor
                * mtp_num_layers
                * (
                    # MTP eh norm + final norm
                    3 * args.hidden_size
                    # MTH eh proj
                    + 2 * args.hidden_size * args.hidden_size
                )
                # Logit.
                + forward_backward_expansion_factor
                * fma_expansion_factor
                * args.hidden_size
                * args.padded_vocab_size
                * (mtp_num_layers + 1)  # MTP + final logit
            )
        )
        return total_floating_point_operations

    # Main entrypoint for FLOPs calculation.
    if is_hybrid_model(args):
        # Calculate the number of each type of layer.
        from operator import itemgetter

        num_attn_layers, num_mamba_layers, num_mlp_layers, num_moe_layers = itemgetter(
            Symbols.ATTENTION, Symbols.MAMBA, Symbols.MLP, Symbols.MOE
        )(get_hybrid_layer_counts(args.hybrid_layer_pattern))

        mtp_num_layers = args.mtp_num_layers
        if mtp_num_layers is None:
            mtp_num_layers = 0
        # Compute hybrid model FLOPs.
        return hybrid_flops(
            batch_size=batch_size,
            seq_len=args.seq_length,
            hidden_size=args.hidden_size,
            num_attn_layers=num_attn_layers,
            num_mamba_layers=num_mamba_layers,
            num_mlp_layers=num_mlp_layers,
            num_moe_layers=num_moe_layers,
            mamba_state_dim=args.mamba_state_dim,
            mamba_head_dim=args.mamba_head_dim,
            mamba_num_groups=args.mamba_num_groups,
            mamba_num_heads=args.mamba_num_heads,
            num_attn_heads=args.num_attention_heads,
            gqa=args.group_query_attention,
            gqa_groups=args.num_query_groups,
            kv_channels=args.kv_channels,
            mlp_expansion=args.ffn_hidden_size / args.hidden_size,
            swiglu=args.swiglu,
            moe_latent_size=args.moe_latent_size,
            moe_ffn_hidden_size=(args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None
                                 else args.ffn_hidden_size),
            shared_expert_ffn_hidden_size=(0 if args.moe_shared_expert_intermediate_size is None
                                           else args.moe_shared_expert_intermediate_size),
            num_experts_routed_to=args.moe_router_topk,
            vocab_size=args.padded_vocab_size,
            mtp_num_layers=mtp_num_layers,
        )
    # Compute standard Transformer model FLOPs.
    return transformer_flops()
