# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Modified some config parameters to adapt to MindSpore Transformer.
"""Transformer Config"""

import enum
import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import mindspore as ms
from mindspore import dtype

from mindformers.modules.transformer.transformer import TransformerOpParallelConfig
from mindformers.parallel_core.utils.init_method import init_method_normal, scaled_init_method_normal
from mindformers.tools.logger import logger

ms_dtype_mapping = {
    # Common floating point numbers types
    "float64": dtype.float64,
    "fp64": dtype.float64,
    "float32": dtype.float32,
    "fp32": dtype.float32,
    "bfloat16": dtype.bfloat16,
    "bf16": dtype.bfloat16,
    "float16": dtype.float16,
    "fp16": dtype.float16,
    # Signed integer types
    "int8": dtype.int8,
    "int16": dtype.int16,
    "int32": dtype.int32,
    "int64": dtype.int64,
    # Unsigned integer types
    "uint8": dtype.uint8,
    "uint16": dtype.uint16,
    "uint32": dtype.uint32,
    "uint64": dtype.uint64,
    # Complex number types
    "complex64": dtype.complex64,
    "complex128": dtype.complex128
}


def convert_str_to_mstype(type_str) -> dtype:
    """
    Utils for convert type string to mstype.

    Args:
        type_str (Union[str, dtype]): A string describing the dtype, or mindspore.dtype.

    Returns:
        A dtype of `mindspore.dtype` .
    """
    if not isinstance(type_str, str):
        raise TypeError(f"The type of 'type_str' must 'string', but got '{type(type_str)}'.")

    if type_str in ms_dtype_mapping:
        return ms_dtype_mapping[type_str]

    raise ValueError(f"The value of 'type_str' must be in {list(ms_dtype_mapping.keys())}, "
                     f"but got '{type_str}'.")


class ParamUsage:
    TRAINING = "training"
    INFERENCE = "inference"
    COMMON = "common"


class ParamSource:
    MEGATRON = "megatron-lm"
    HF = "huggingface"
    MF = "mindformers"


class ParamMode:
    GRAPH = "graph"
    PYNATIVE = "pynative"
    COMMON = "common"

@dataclass
class TransformerConfig:
    """
    Configuration object for MindSpore Transformer's transformers.

    The initialization function has an argument for each parameter, including those in ModelParallelConfig.
    """
    ########################################################
    # General Configuration Items For MindSpore Transformers
    ########################################################

    batch_size: int = field(
        default=1,
        metadata={
            "description": "Per batch size for training and inference. Default: 1.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    parallel_config: Union[dict, TransformerOpParallelConfig] = field(
        default=None,
        metadata={
            "description": "Configs which contains parallel settings.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    pet_config: dict = field(
        default=None,
        metadata={
            "description": "Config for Pattern-Exploiting Training (PET).",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    context_parallel_algo: str = field(
        default="colossalai_cp",
        metadata={
            "description": "Algorithm to use for context parallelism. Can be \"colossalai_cp\", \"ulysses_cp\" or "
                           "\"hybrid_cp\". Only effective when context_parallel > 1",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    is_dynamic: bool = field(
        default=False,
        metadata={
            "description": "Whether model is dynamic shape.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    compute_dtype: str = field(
        default="bfloat16",
        metadata={
            "description": "Linear layer compute dtype.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    layernorm_compute_dtype: str = field(
        default="float32",
        metadata={
            "description": "LayerNorm compute dtype.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    rotary_dtype: str = field(
        default="float32",
        metadata={
            "description": "Custom rotary position embedding compute dtype.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    rotary_cos_format: str = field(
        default="rotate_half",
        metadata={
            "description": "Custom cosine position embedding format.",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    bias_swiglu_fusion: bool = field(
        default=False,
        metadata={
            "description": "If True, use fused swiglu kernel.",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    mla_qkv_concat: bool = field(
        default=True,
        metadata={
            "description": "If True, Multi Latent Attention computes q_compressed, k, kv_compressed in a single "
                           "linear transformation; if False, computes them separately.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    use_contiguous_weight_layout_attention: bool = field(
        default=False,
        metadata={
            "description": "Determines the weight arrangement in SelfAttention's QKV linear projection. Only affects "
                           "SelfAttention layers. When True: Uses contiguous layout: [Q_weights, K_weights, "
                           "V_weights]. When False (default): Uses interleaved head layout. Note: This affects tensor "
                           "memory layout but not mathematical equivalence.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    use_interleaved_weight_layout_mlp: bool = field(
        default=True,
        metadata={
            "description": "Determines the weight arrangement in MLP's linear_fc1 projection. Only affects MLP "
                           "layers. When True (default): Uses interleaved layout. When False: Uses contiguous layout. "
                           "Note: This affects tensor memory layout but not mathematical equivalence.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    normalization: str = field(
        default="LayerNorm",
        metadata={
            "description": "Which norm to use for normalization layers, valid options are `LayerNorm` and `RMSNorm`.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    fused_norm: bool = field(
        default=True,
        metadata={
            "description": "Whether to use fused-normalization, only support (only effective during training for "
                           "now). When True (default): If normalization = \"LayerNorm\" and fused_norm = True, "
                           "use the FusedNorm operator. If normalization = \"RMSNorm\" and fused_norm = True, "
                           "use the FusedRMSNorm operator.",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    add_bias_linear: bool = field(
        default=True,
        metadata={
            "description": "Include a bias term in all linear layers (QKV projections, after core attention, "
                           "and two in MLP layer).",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    add_mlp_fc1_bias_linear: bool = field(
        default=None,
        metadata={
            "description": "Include a bias term in fc1 linear in MLP layer.",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    add_mlp_fc2_bias_linear: bool = field(
        default=None,
        metadata={
            "description": "Include a bias term in fc2 linear in MLP layer.",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    gated_linear_unit: bool = field(
        default=False,
        metadata={
            "description": "Use a gated linear unit for the first linear layer in the MLP.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    ################################################################
    # Flash Attention Configuration Items for MindSpore Transformers
    ################################################################

    use_flash_attention: bool = field(
        default=True,
        metadata={
            "description": "If true, use flash attention for the attention layer.",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    rotary_seq_len_interpolation_factor: float = field(
        default=None,
        metadata={
            "description": "RoPE scaling used for linear interpolation of longer sequences. This value must be a "
                           "floating point number greater than 1.0.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    rope_scaling: dict = field(
        default=None,
        metadata={
            "description": "Dictionary containing the scaling configuration for the RoPE embeddings.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.HF,
            "mode": ParamMode.COMMON
        }
    )

    use_rope_scaling: bool = field(
        default=False,
        metadata={
            "description": "Whether to use RoPE scaling.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    input_layout: str = field(
        default="BNSD",
        metadata={
            "description": "Specifies the layout of input query, key and value. The value can be \"BSH\", \"BNSD\", "
                           "\"SBH\", \"BSND\" or \"TND\". \"TND\" is an experimental format. More details, "
                           "please refer MindSpore API Document: mindspore.ops.flash_attention_score",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    sparse_mode: int = field(
        default=0,
        metadata={
            "description": "Indicates sparse mode: 0: defaultMask; 1: allMask; 2: leftUpCausal; 3: rightDownCausal; "
                           "4: band; 5: prefix (not implemented); 6: global (not implemented); 7: dilated (not "
                           "implemented); 8: block_local (not implemented).",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    use_alibi_mask: bool = field(
        default=False,
        metadata={
            "description": "The value is True if alibi_mask is passed.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    use_attn_mask_compression: bool = field(
        default=False,
        metadata={
            "description": "If true, use attention mask compression for the attention layer.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    use_eod_attn_mask_compression: bool = field(
        default=False,
        metadata={
            "description": "If true, use end of sequence attention mask compression for the attention layer.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    use_attention_mask: bool = field(
        default=True,
        metadata={
            "description": "If true, use attention mask for the attention layer.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    use_ring_attention: bool = field(
        default=False,
        metadata={
            "description": "If true, use ring attention for the attention layer.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    fp16_lm_cross_entropy: bool = field(
        default=False,
        metadata={
            "description": "If true, use fp16 for cross entropy loss calculation.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    untie_embeddings_and_output_weights: bool = field(
        default=True,
        metadata={
            "description": "If true, untie the embeddings and output weights.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    hidden_act: str = field(
        default="gelu",
        metadata={
            "description": "Activation function to use for the non-linearity in the MLP.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.HF,
            "mode": ParamMode.COMMON
        }
    )

    mask_func_type: str = field(
        default="attn_mask_fill",
        metadata={
            "description": "Mask function type to use for the attention layer.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    track_max_attention_logit: bool = field(
        default=False,
        metadata={
            "description": "Whether to monitor the maximum attention logit value during training.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    ####################################################
    # MoE Configuration Items For MindSpore Transformers
    ####################################################
    comp_comm_parallel: bool = field(
        default=False,
        metadata={
            "description": "Whether to enable ffn compute and communication parallel, which can reduce pure "
                           "communicattion time by splitting and overlapping compute and communication.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    comp_comm_parallel_degree: int = field(
        default=2,
        metadata={
            "description": "The split number of compute and communication. The larger the numbers,the more overlap "
                           "there will be but will consume more memory. This parameter is effective only when "
                           "comp_comm_parallel enable.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    norm_topk_prob: bool = field(
        default=True,
        metadata={
            "description": "If True, use top-k probability for normalization.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.HF,
            "mode": ParamMode.COMMON
        }
    )

    use_fused_ops_topkrouter: bool = field(
        default=False,
        metadata={
            "description": "If True, use fused ops for top-k routing.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    use_shared_expert_gating: bool = field(
        default=False,
        metadata={
            "description": "If True, use shared expert gating.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    topk_method: str = field(
        default="greedy",
        metadata={
            "description": "Method to use for top-k routing.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.HF,
            "mode": ParamMode.COMMON
        }
    )

    npu_nums_per_device: int = field(
        default=8,
        metadata={
            "description": "Set NPU ranks for each device.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    use_pad_tokens: bool = field(
        default=True,
        metadata={
            "description": "If True, gmm pads an additional protection token to avoid 0-token calculation.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    callback_moe_droprate: bool = field(
        default=False,
        metadata={
            "description": "Whether to print each expert's load information through callback.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    first_k_dense_replace: int = field(
        default=None,
        metadata={
            "description": "Number of dense layers in shallow layers("
                           "embed->dense->dense->...->dense->moe->moe...->lm_head).",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.HF,
            "mode": ParamMode.COMMON
        }
    )

    moe_router_enable_expert_bias: bool = field(
        default=False,
        metadata={
            "description": "TopK routing with dynamic per-expert bias in the aux-loss-free load balancing strategy. "
                           "The routing decision is based on the sum of the routing scores and the expert bias. See "
                           "https://arxiv.org/abs/2408.15664 for details.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_router_force_expert_balance: bool = field(
        default=False,
        metadata={
            "description": "Whether to force expert balance in router. This option is only used in performance "
                           "testing, not for general use.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    moe_router_score_function: str = field(
        default="softmax",
        metadata={
            "description": "Score function for MoE routing. Can be \"softmax\" or \"sigmoid\".",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_router_fusion: bool = field(
        default=False,
        metadata={
            "description": "Fuse ops in routing and aux loss calculation.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )
    ################################################
    # Training Parameters for MindSpore Transformers
    ################################################

    use_eod_reset: bool = field(
        default=False,
        metadata={
            "description": "Whether to use eod reset.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    print_separate_loss: bool = field(
        default=True,
        metadata={
            "description": "Print lm_loss, extra_loss and mtp_loss separately.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    #################################################
    # Inference Parameters for MindSpore Transformers
    #################################################

    vocab_size: int = field(
        default=128000,
        metadata={
            "description": "Vocabulary size of the model.",
            "usage": ParamUsage.COMMON,
            "source": [ParamSource.MF, ParamSource.HF, ParamSource.MEGATRON],
            "mode": ParamMode.COMMON
        }
    )

    seq_length: int = field(
        default=4096,
        metadata={
            "description": "Model Seq Length",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    pad_token_id: int = field(
        default=0,
        metadata={
            "description": "Model pad token id.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.HF,
            "mode": ParamMode.COMMON
        }
    )

    ignore_token_id: int = field(
        default=-100,
        metadata={
            "description": "Model ignore token id when training.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    max_position_embeddings: int = field(
        default=4096,
        metadata={
            "description": "Maximum sequence length that the model can handle.",
            "usage": ParamUsage.COMMON,
            "source": [ParamSource.HF, ParamSource.MEGATRON],
            "mode": ParamMode.COMMON
        }
    )

    sandwich_norm: bool = field(
        default=False,
        metadata={
            "description": "Whether to apply `normalization` type of normalization to the transformer layer.",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    tie_word_embeddings: bool = field(
        default=False,
        metadata={
            "description": "Whether to share the input and output embedding weights.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.HF,
            "mode": ParamMode.COMMON
        }
    )

    block_size: int = field(
        default=16,
        metadata={
            "description": "Size of each memory block used in PagedAttention.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    num_blocks: int = field(
        default=512,
        metadata={
            "description": "Size of each memory block used in PagedAttention.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    pre_process: bool = field(
        default=True,
        metadata={
            "description": "When using pipeline parallel, indicate whether it's the first stage.",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    post_process: bool = field(
        default=True,
        metadata={
            "description": "When using pipeline parallel, indicate whether it's the last stage.",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    dispatch_global_max_bs: int = field(
        default=0,
        metadata={
            "description": "Maximum global batch size in MoE dispatch with AlltoAll",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    attn_reduce_scatter: bool = field(
        default=False,
        metadata={
            "description": "Whether to enable attn_reduce_scatter",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    attn_allgather: bool = field(
        default=False,
        metadata={
            "description": "Whether to enable attn_allgather",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    attn_allreduce: bool = field(
        default=True,
        metadata={
            "description": "Whether to enable attn_allreduce",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    ffn_reduce_scatter: bool = field(
        default=False,
        metadata={
            "description": "Whether to enable ffn_reduce_scatter",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    ffn_allgather: bool = field(
        default=False,
        metadata={
            "description": "Whether to enable ffn_allgather",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    ffn_allreduce: bool = field(
        default=True,
        metadata={
            "description": "Whether to enable ffn_allreduce",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    use_alltoall: bool = field(
        default=False,
        metadata={
            "description": "Whether to enable use_alltoall",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    use_fused_mla: bool = field(
        default=False,
        metadata={
            "description": "Whether to enable use_fused_mla",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    quantization_config: dict = field(
        default=None,
        metadata={
            "description": "Quantization configuration.",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    disable_lazy_inline: bool = field(
        default=False,
        metadata={
            "description": "Whether to disable Lazy Inline compilation acceleration.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    coeff: float = field(
        default=0.1,
        metadata={
            "description": "Calculate the relative scaling coefficient of mscale in Yarn Rope.",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    ###################
    # Model parallelism
    ###################

    data_parallel_size: int = field(
        default=1,
        metadata={
            "description": "Data parallelism. The training data is partitioned into multiple micro-batches, with each "
                           "batch assigned to a distinct device for distributed parallel processing. Each accelerator "
                           "independently processes different batches in parallel, and the gradients or outputs are "
                           "subsequently synchronized and aggregated across devices.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    tensor_model_parallel_size: int = field(
        default=1,
        metadata={
            "description": "Intra-layer model parallelism. Splits tensors across NPU ranks.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    pipeline_model_parallel_size: int = field(
        default=1,
        metadata={
            "description": "Inter-layer model parallelism. Splits transformer layers across NPU ranks.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    virtual_pipeline_model_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "description": "Interleaved pipeline parallelism is used to improve performance by reducing the pipeline "
                           "bubble. Considers a transformer block as a list of smaller transformer (virtual) blocks. "
                           "The number of virtual blocks per pipeline model parallel rank is the virtual model "
                           "parallel size.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    sequence_parallel: bool = field(
        default=False,
        metadata={
            "description": "Makes tensor parallelism more memory efficient for LLMs (20B+) by parallelizing layer "
                           "norms and dropout sequentially. See Reducing Activation Recomputation in Large "
                           "Transformer Models (https://arxiv.org/abs/2205.05198) for more details.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    context_parallel_size: int = field(
        default=1,
        metadata={
            "description": "Splits network input along sequence dimension across NPU ranks.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    hierarchical_context_parallel_sizes: int = field(
        default=1,
        metadata={
            "description": "Reserved interface. Degrees of the hierarchical context parallelism. Users should provide "
                           "a list to specify the sizes for different levels. Taking the a2a+p2p cp comm type as "
                           "example, it contains groups of two levels, so the first value of the list indicates the "
                           "group size of the a2a communication type, and the second value indicates the group size "
                           "of the p2p communication type.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    expert_model_parallel_size: int = field(
        default=1,
        metadata={
            "description": "Distributes Moe Experts across sub data parallel dimension.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    expert_tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "description": "Intra-layer tensor model parallelism for expert layer. Splits tensors across NPU ranks.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    # Mindformers New
    micro_batch_num: Optional[int] = field(
        default=1,
        metadata={
            "description": "MicroBatch size for Pipeline Parallel. Default: 1.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    seq_split_num: Optional[int] = field(
        default=1,
        metadata={
            "description": "Sequence split number in sequence pipeline parallel mode. Default: 1.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    gradient_aggregation_group: int = field(
        default=4,
        metadata={
            "description": "The fusion group size of the optimizer state sharding. Default: 4.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    offset: Optional[Union[int, list]] = field(
        default=0,
        metadata={
            "description": "Offset of transformer layer when set pipeline stage number. Default: 0.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    ulysses_degree_in_cp: int = field(
        default=1,
        metadata={
            "description": "The number of parallel slices of the Ulysses sequence. For configuration method of "
                           "distributed parallel parameters, refer to the contents of the Parallel Configuration "
                           "section in MindSpore Transformers configuration description: ("
                           "https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/configuration.html)",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    vocab_emb_dp: Optional[bool] = field(
        default=False,
        metadata={
            "description": "Whether to split the vocabulary only along the dp dimension. This setting is not "
                           "supported to be configured as True at present; otherwise, it will be converted to False "
                           "automatically.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    ###################
    # Training
    ###################

    params_dtype: str = field(
        default="float32",
        metadata={
            "description": "dtype used when initializing the weights.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    ###################
    # CPU Offloading
    ###################
    cpu_offloading: bool = field(
        default=False,
        metadata={
            "description": "Enable offload of the transformer block or not. Default: False.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    # MindFormers New
    cpu_offloading_num_layers: Optional[Union[list, dict]] = field(
        default=None,
        metadata={
            "description": "Configuration for layer swapping. Each item in the list specifies the `backward_prefetch` "
                           "value for a specific layer. Default: None.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    op_swap: Optional[Union[list, dict]] = field(
        default=None,
        metadata={
            "description": "Configuration for operator swapping. Each item in the list specifies the "
                           "`backward_prefetch` value for operators matching a specific pattern. Default: None.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    default_prefetch: int = field(
        default=1,
        metadata={
            "description": "Number of operators to prefetch activations before the backward FlashAttention (FA) "
                           "operator. In the context of static graph execution, since the activation values that have "
                           "been offloaded need to be retrieved again during the backward pass, and retrieving data "
                           "from CPU back to NPU incurs latency.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    ####################
    # Model Architecture
    ####################

    num_layers: int = field(
        default=0,
        metadata={
            "description": "Number of transformer layers in a transformer block.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    mtp_num_layers: Optional[int] = field(
        default=None,
        metadata={
            "description": "Number of Multi-Token Prediction (MTP) Layers.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    mtp_loss_scaling_factor: Optional[float] = field(
        default=None,
        metadata={
            "description": "Weighting factor of Multi-Token Prediction (MTP) loss.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    hidden_size: int = field(
        default=0,
        metadata={
            "description": "Transformer hidden size.",
            "usage": ParamUsage.COMMON,
            "source": [ParamSource.HF, ParamSource.MEGATRON],
            "mode": ParamMode.COMMON
        }
    )

    num_attention_heads: int = field(
        default=0,
        metadata={
            "description": "Number of transformer attention heads.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    softmax_scale: Optional[float] = field(
        default=None,
        metadata={
            "description": "Softmax scale for attention scaling.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    num_query_groups: Optional[int] = field(
        default=None,
        metadata={
            "description": "Number of query groups for group query attention. If None, normal attention is used.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    ffn_hidden_size: Optional[int] = field(
        default=None,
        metadata={
            "description": "Transformer Feed-Forward Network hidden size.This is set to 4*hidden_size if not provided.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    kv_channels: Optional[int] = field(
        default=None,
        metadata={
            "description": "Projection weights dimension in multi-head attention. This is set to hidden_size // "
                           "num_attention_heads if not provided.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    hidden_dropout: float = field(
        default=0.1,
        metadata={
            "description": "Dropout probability for transformer hidden state.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    attention_dropout: float = field(
        default=0.1,
        metadata={
            "description": "Post attention dropout probability.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    fp32_residual_connection: bool = field(
        default=False,
        metadata={
            "description": "If true, move residual connections to fp32.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    apply_residual_connection_post_layernorm: bool = field(
        default=False,
        metadata={
            "description": "If True, uses the original BERT residue connection ordering.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    layernorm_epsilon: float = field(
        default=1e-5,
        metadata={
            "description": "Epsilon value for any LayerNorm operations.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    add_qkv_bias: bool = field(
        default=False,
        metadata={
            "description": "Add a bias term only for QKV projections.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    activation_func: str = field(
        default="gelu",
        metadata={
            "description": "Activation function to use for the non-linearity in the MLP.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    num_moe_experts: Optional[int] = field(
        default=None,
        metadata={
            "description": "Number of experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to "
                           "None for no MoE.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    rotary_interleaved: bool = field(
        default=False,
        metadata={
            "description": "True is rotate pairs of even and odd dimensions (RoFormer style), False is rotate pairs "
                           "of first half and second half (LLaMa style). Default to False.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    calculate_per_token_loss: bool = field(
        default=False,
        metadata={
            "description": "Whether cross entropy loss is calculated over the actual number of non-padded tokens in "
                           "the global batch, versus the default behavior of assuming all tokens are non-padded.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    multi_latent_attention: bool = field(
        default=False,
        metadata={
            "description": "Whether to use multi-latent attention.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    position_embedding_type: str = field(
        default="rope",
        metadata={
            "description": "Position embedding type to use for the attention layer.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    nope_layer_interval: int = field(
        default=None,
        metadata={
            "description": "Interval for inserting NoPE (No Position Embedding) layers among RoPE layers.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    rotary_base: float = field(
        default=10000.0,
        metadata={
            "description": "Rotary base for the rotary embeddings, used by rope and yarn. Mindformers required.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    partial_rotary_factor: float = field(
        default=1.0,
        metadata={
            "description": "rotaty partial dim",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    qk_layernorm: bool = field(
        default=False,
        metadata={
            "description": "Whether to apply `normalization` type of normalization to the query and key embeddings.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    ####################
    # attention variant
    ####################
    experimental_attention_variant: str = field(
        default=None,
        metadata={
            "description": "Type of attention variant to use. Currently support ['dsa']."
                           "Defaults: `None`, which means that no attention variant will be applied.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.GRAPH
        }
    )

    dsa_indexer_n_heads: int = field(
        default=None,
        metadata={
            "description": "Number of DSA indexer heads.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.GRAPH
        }
    )

    dsa_indexer_head_dim: int = field(
        default=None,
        metadata={
            "description": "Dimension per DSA indexer head.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.GRAPH
        }
    )

    dsa_indexer_topk: int = field(
        default=None,
        metadata={
            "description": "Number of top-k tokens to select in DSA indexer.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.GRAPH
        }
    )

    dsa_indexer_loss_coeff: float = field(
        default=None,
        metadata={
            "description": "Coefficient for the DSA indexer KL divergence loss. Set to 0 to disable indexer loss.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.GRAPH
        }
    )

    dsa_indexer_use_sparse_loss: bool = field(
        default=False,
        metadata={
            "description": "Whether to use sparse DSA indexer loss."
                           "If True, the indexer loss will be computed using the top-k indices.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.GRAPH
        }
    )

    ####################
    # Manifold-Constrained Hyper-Connections (mHC)
    ####################

    enable_hyper_connections: bool = False
    """If True, enable Manifold-Constrained Hyper-Connections (mHC) between Attention and FFN layers."""

    num_residual_streams: int = 4
    """Number of residual streams (tile factor n) used by mHC."""

    mhc_sinkhorn_iterations: int = 20
    """Number of Sinkhorn-Knopp iterations for mHC residual matrix normalization."""

    mhc_init_gating_factor: float = 0.01
    """Initial value of alpha gating factors for mHC projection."""

    ####################
    # Initialization
    ####################

    init_method: Optional[Callable] = field(
        default=None,
        metadata={
            "description": "Method to initialize weights. Note that bias is always set to zero. Should be a function "
                           "that takes a single Tensor and initializes it. If None, will be set to "
                           "init_method_normal(init_method_std) which is torch nn init normal with mean=0.0 and "
                           "std=init_method_std.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    output_layer_init_method: Optional[Callable] = field(
        default=None,
        metadata={
            "description": "Method to initialize weights of the output layer of both attention and MLP blocks. If "
                           "None, will be set to scaled_init_method_normal(init_method_std) which is torch nn init "
                           "normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_layers).",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    init_method_std: float = field(
        default=0.02,
        metadata={
            "description": "Standard deviation of the zero mean normal for the default initialization method, "
                           "not used if init_method and output_layer_init_method are provided.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    param_init_std_rules: List[dict[str, Union[str, float]]] = field(
        default=None,
        metadata={
            "description": "Configuration for decoupled weight initialization.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    ####################
    # Mixed-Precision
    ####################

    apply_query_key_layer_scaling: bool = field(
        default=False,
        metadata={
            "description": "If true, scale Q * K^T by 1 / layer-number. This improve numeric stability when training "
                           "with fp16.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    attention_softmax_in_fp32: bool = field(
        default=True,
        metadata={
            "description": "If True, run attention masking and softmax in fp32. This should be True if "
                           "apply_query_key_layer_scaling is True.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    softmax_compute_dtype: str = field(
        default='float32',
        metadata={
            "description": "Data type for computing softmax during attention computation.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    ####################
    # Fusion
    ####################

    bias_dropout_fusion: bool = field(
        default=False,
        metadata={
            "description": "If True, uses bias dropout fusion.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    apply_rope_fusion: bool = field(
        default=False,
        metadata={
            "description": "If True, use fused RoPE kernel.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    ####################
    # Recompute
    ####################

    recompute: Optional[Union[bool, list, tuple]] = field(
        default=False,
        metadata={
            "description": "Whether enable recompute. Default: False.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    select_recompute: Optional[Union[bool, list]] = field(
        default=False,
        metadata={
            "description": "Turn on recomputation to recompute only for the operators in the attention layer. "
                           "Default: False.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    parallel_optimizer_comm_recompute: Optional[bool] = field(
        default=False,
        metadata={
            "description": "Whether to recompute AllGather communication introduced in parallel by the optimizer. "
                           "Default: False.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    select_comm_recompute: Optional[bool] = field(
        default=False,
        metadata={
            "description": "Whether to slice the Cell outputs retained in memory. Default: False.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    mp_comm_recompute: Optional[bool] = field(
        default=True,
        metadata={
            "description": "Whether to recompute communications introduced by model parallel. Default: True.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    recompute_slice_activation: bool = field(
        default=False,
        metadata={
            "description": "Whether to output slices for Cells kept in memory. Default: False.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    select_recompute_exclude: Optional[Union[bool, list]] = field(
        default=False,
        metadata={
            "description": "Disable recomputation for the specified operator, valid only for the Primitive operators.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    select_comm_recompute_exclude: Optional[Union[bool, list]] = field(
        default=False,
        metadata={
            "description": "Disable communication recomputation for the specified operator, valid only for the "
                           "Primitive operators.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    ####################
    # MoE
    ####################

    moe_shared_expert_intermediate_size: Optional[int] = field(
        default=None,
        metadata={
            "description": "Shared expert total ffn hidden size. It should be equal to 'num_shared_experts * "
                           "ffn_size_of_each_shared_expert' if there are multiple shared experts. None means no "
                           "shared expert.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_shared_expert_overlap: bool = field(
        default=False,
        metadata={
            "description": "Enable overlapping between shared expert computations and dispatcher communications. "
                           "Without this, the shared epxerts execute after the routed experts.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_layer_freq: Optional[Union[int, List[int]]] = field(
        default=1,
        metadata={
            "description": "Frequency between MoE layers and Dense layers. Accepts either: An integer N: Represents a "
                           "1:N ratio, meaning one expert layer for every N-1 dense layers. A list that defines a "
                           "custom pattern, e.g.: [1,1,1,0,1,1,1,0,1,1,1,0]",
            "usage": ParamUsage.COMMON,
            "source": [ParamSource.HF, ParamSource.MEGATRON],
            "mode": ParamMode.COMMON
        }
    )

    moe_ffn_hidden_size: Optional[int] = field(
        default=None,
        metadata={
            "description": "MoE Feed-Forward Network hidden size",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_router_load_balancing_type: str = field(
        default="sub_seq_aux_loss",
        metadata={
            "description": "The load balancing strategy for the router. \"sub_seq_aux_loss\" (Legacy), "
                           "\"seq_aux_loss\" (DeepSeekV2/V3), \"gbs_aux_loss\" (Qwen3MoE). The default is "
                           "\"sub_seq_aux_loss\".",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_router_topk: int = field(
        default=2,
        metadata={
            "description": "Number of experts to route to for each token.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_router_num_groups: Optional[int] = field(
        default=None,
        metadata={
            "description": "Number of groups to divide experts into for group-limited routing. When using "
                           "group-limited routing: Experts are divided into equal-sized groups; for each token, "
                           "groups are selected; from these groups, individual experts are chosen. Device-limited or "
                           "node-limited routing use cases.",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_router_group_topk: Optional[int] = field(
        default=None,
        metadata={
            "description": "Number of selected groups for group-limited routing.",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_router_pre_softmax: bool = field(
        default=False,
        metadata={
            "description": "Enable pre-softmax(pre-sigmoid) routing for MoE, which means softmax is before the top-k "
                           "selection. By default, softmax is done after top-k.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_router_topk_scaling_factor: Optional[float] = field(
        default=None,
        metadata={
            "description": "Scaling factor for routing score in top-k selection, only works when "
                           "moe_router_pre_softmax enabled. Defaults to None, which means no scaling.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_router_dtype: str = field(
        default="float32",
        metadata={
            "description": "Data type for routing and expert output weighted averaging. Using fp32 or fp64 can "
                           "improve stability especially when the number of experts is large (e.g. finegrained-moe). "
                           "None means no changes for dtype.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_router_bias_update_rate: float = field(
        default=1e-3,
        metadata={
            "description": "The expert bias is updated based on the number of assigned tokens to each expert in a "
                           "global batch, where the bias is increased for the experts with less assigned tokens and "
                           "decreased for the experts with more assigned tokens. The default value 1e-3 is same as "
                           "that used in DeepSeekV3.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_grouped_gemm: bool = field(
        default=False,
        metadata={
            "description": "When there are multiple experts per rank, compress multiple local (potentially small) "
                           "gemms in a single kernel launch to improve the utilization and performance by leveraging "
                           "the Grouped GEMM feature introduced since CUTLASS 2.8.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_aux_loss_coeff: float = field(
        default=0.,
        metadata={
            "description": "Scaling coefficient for the aux loss. A starting value of 1e-2 is recommended.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    moe_z_loss_coeff: Optional[float] = field(
        default=None,
        metadata={
            "description": "Scaling coefficient for the z-loss. A starting value of 1e-3 is recommended.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    group_wise_a2a: bool = field(
        default=False,
        metadata={
            "description": "Whether to enable group-wise alltoall communication, which can reduce communication time "
                           "by converting part of intercommunication into intra communication. This parameter is "
                           "effective only when model parallel > 1 and data_parallel equal to expert parallel.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    moe_token_dispatcher_type: str = field(
        default="alltoall",
        metadata={
            "description": "The type of token dispatcher to use. The default is 'alltoall'. Options are 'alltoall', "
                           "'alltoall_deredundency' and 'alltoall_zero_redundancy'.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_expert_capacity_factor: Optional[float] = field(
        default=None,
        metadata={
            "description": "The capacity factor for each expert, None means no token will be dropped. The default is "
                           "None.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_token_drop_policy: str = field(
        default='probs',
        metadata={
            "description": "The policy to drop tokens. Can be either \"probs\" or \"position\". If \"probs\", "
                           "the tokens with the lowest probabilities will be dropped. If \"position\", tokens at the "
                           "end of each batch will be dropped.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    moe_permute_fusion: bool = field(
        default=False,
        metadata={
            "description": "Fuse token rearrangement ops during token dispatching.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    moe_apply_probs_on_input: bool = field(
        default=False,
        metadata={
            "description": "Apply probs on input of experts instead of applying after activation and glu.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    # MindFormers New
    shared_expert_num: int = field(
        default=0,
        metadata={
            "description": "Number of shared experts.",
            "usage": ParamUsage.INFERENCE,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    print_expert_load: bool = field(
        default=False,
        metadata={
            "description": "Whether to print expert load information. When enabled, detailed expert load statistics "
                           "will be printed during training. Contributed by Xirang Intelligent Computing Team of "
                           "State Cloud.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    enable_expert_relocation: bool = field(
        default=False,
        metadata={
            "description": "Enable dynamic expert relocation for load balancing in MoE models. When enabled, "
                           "experts will be dynamically redistributed across devices based on their load history to "
                           "improve training efficiency and load balance. Contributed by Xirang Intelligent Computing "
                           "Team of State Cloud.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    expert_relocation_initial_iteration: int = field(
        default=20,
        metadata={
            "description": "The initial iteration to start expert relocation. Expert relocation will begin after this "
                           "many training iterations. Contributed by Xirang Intelligent Computing Team of State Cloud.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    expert_relocation_freq: int = field(
        default=50,
        metadata={
            "description": "Frequency of expert relocation in training iterations. Expert relocation will be "
                           "performed every N iterations after the initial iteration. Contributed by Xirang "
                           "Intelligent Computing Team of State Cloud.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    ##################
    # Context Parallel
    ##################

    cp_comm_type: Optional[Union[str, List[str]]] = field(
        default=None,
        metadata={
            "description": "Reserved interface. Inter-NPU communication type for context parallelism. str: all layers "
                           "share same type. List[str]: each layer has its separate type. Each layer can be \"p2p\" "
                           "or \"all_gather\" or \"a2a\" or \"a2a+p2p\". It uses A2A communications in low-level CP "
                           "groups, and P2P communications in high-level CP groups.",
            "usage": ParamUsage.TRAINING,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    window_size: Optional[Tuple[int, int]] = field(
        default=None,
        metadata={
            "description": "If not None, then will use sliding window attention. The size of the window is specified "
                           "by the numbers inside the tuple; -1 is special value meaning \"infinite window size\".",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    window_attn_skip_freq: Optional[Union[int, List[int]]] = field(
        default=None,
        metadata={
            "description": "Frequency of full attention layers among sliding window attention layers. Accepts either: "
                           "An integer N: Represents a (N-1):1 ratio, one full attention layer after (N-1) SWA "
                           "layers. A list that defines a custom pattern, e.g.: [1,1,1,1,0,0,0,0], where 1 represents "
                           "SWA.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    model_architecture: str = field(
        default="decoder_only",
        metadata={
            "description": "The structure of model. Support 'decoder-only', 'yoco'.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    num_encoder_layers: int = field(
        default=None,
        metadata={
            "description": "The number of encoder layers. The num_encoder_layers or the num_decoder_layers should be "
                           "set while use the 'yoco' model structure.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    num_decoder_layers: int = field(
        default=None,
        metadata={
            "description": "The number of decoder layers. The num_encoder_layers or the num_decoder_layers should be "
                           "set while use the 'yoco' model structure.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MF,
            "mode": ParamMode.COMMON
        }
    )

    def __post_init__(self):
        """
        Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more
        details.
        """
        self.params_dtype = convert_str_to_mstype(self.params_dtype)
        if self.sequence_parallel and self.tensor_model_parallel_size <= 1:
            raise ValueError("Can not use sequence parallelism without tensor parallelism")

        self.compute_dtype = convert_str_to_mstype(self.compute_dtype)
        self.layernorm_compute_dtype = convert_str_to_mstype(self.layernorm_compute_dtype)
        self.rotary_dtype = convert_str_to_mstype(self.rotary_dtype)
        self.moe_router_dtype = convert_str_to_mstype(self.moe_router_dtype)
        self.softmax_compute_dtype = convert_str_to_mstype(self.softmax_compute_dtype)

        if not isinstance(self.hidden_dropout, float) or not 0 <= self.hidden_dropout < 1:
            raise ValueError(f"hidden_dropout should be a float within [0, 1), but get {self.hidden_dropout}.")
        if not isinstance(self.attention_dropout, float) or not 0 <= self.attention_dropout < 1:
            raise ValueError(f"attention_dropout should be a float within [0, 1), but get {self.attention_dropout}.")

        if self.vocab_emb_dp:
            logger.warning("vocab_emb_dp is not supported in MCore, it will be converted to False automatically.")
            self.vocab_emb_dp = False

        if self.pad_token_id is None:
            self.pad_token_id = 0

        self.mtp_num_layers = self.mtp_num_layers or 0
        if self.mtp_num_layers is not None:
            if self.mtp_num_layers < 0 or not isinstance(self.mtp_num_layers, int):
                raise ValueError(
                    f"mtp_num_layers should be `None` or non-negative integer, but get {self.mtp_num_layers}."
                )
            if self.mtp_num_layers > 1:
                raise ValueError(
                    f"The current version only supports the scenario where `mtp_num_layers` = `1` is configured. "
                    f"But get {self.mtp_num_layers}."
                )

        if self.num_attention_heads % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads

        if self.context_parallel_size > 1 and not self.use_flash_attention:
            raise ValueError("context_parallel is only available for flash attention for now, "
                             "please set use_flash_attention=True.")

        if self.experimental_attention_variant == 'dsa':
            if not self.multi_latent_attention:
                raise ValueError("When experimental_attention_variant == 'dsa', multi_latent_attention should be True.")
            if not self.use_flash_attention:
                raise ValueError("DeepSeek Sparse Attention is only available for flash attention for now, "
                                 "please set use_flash_attention=True.")
            if ms.get_auto_parallel_context("pipeline_scheduler") == "zero_bubble_v":
                raise ValueError("When experimental_attention_variant == 'dsa', zero_bubble_v is not supported.")

        if self.use_flash_attention:
            if self.use_eod_attn_mask_compression and not self.use_ring_attention:
                self.input_layout = "TND"
                if self.attention_dropout != 0:
                    logger.warning("When use TND layout of flash attention, attention_dropout is ignored. Set to 0.")
                    self.attention_dropout = 0.
            elif self.experimental_attention_variant == 'dsa':
                self.input_layout = "BSND"
            elif self.context_parallel_size > 1:
                self.input_layout = "BSH"
            else:
                self.input_layout = "BNSD"

            if self.use_eod_attn_mask_compression and not self.use_ring_attention:
                self.sparse_mode = 3
            elif self.use_attn_mask_compression and not self.use_ring_attention:
                self.sparse_mode = 2
            else:
                self.sparse_mode = 0
        else:
            if self.use_eod_attn_mask_compression or self.use_attn_mask_compression:
                raise ValueError("When use mask compression, use_flash_attention must be True.")
            if self.use_ring_attention:
                raise ValueError("When use ring attention, use_flash_attention must be True.")

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.expert_model_parallel_size > 1 and self.num_moe_experts is None:
            raise ValueError('num_moe_experts must be non None to use expert-parallel.')

        if self.moe_token_dispatcher_type == "alltoall_deredundency" and \
                (self.expert_model_parallel_size < self.npu_nums_per_device):
            raise ValueError(
                f"expert_model_parallel_size must be greater than or equal to npu_nums_per_device when using "
                f"'alltoall_deredundency', but got expert_model_parallel_size={self.expert_model_parallel_size} "
                f"< npu_nums_per_device={self.npu_nums_per_device}."
            )

        if self.num_moe_experts is not None and self.num_moe_experts <= 0:
            raise ValueError('num_moe_experts must be non-negative.')

        if self.moe_ffn_hidden_size is None:
            self.moe_ffn_hidden_size = self.ffn_hidden_size

        if self.moe_shared_expert_intermediate_size is not None:
            if self.shared_expert_num == 0:
                logger.warning("The hidden-size of shared experts ('moe_shared_expert_intermediate_size') is set, "
                               "but get shared_expert_num = 0. The shared_expert_num will be ignored.")
            elif self.moe_shared_expert_intermediate_size != self.moe_ffn_hidden_size * self.shared_expert_num:
                logger.warning(
                    f'moe_shared_expert_intermediate_size should be '
                    f'num_shared_experts ({self.shared_expert_num}) * '
                    f'ffn_size_of_each_shared_expert ({self.moe_ffn_hidden_size}), '
                    f'but got {self.moe_shared_expert_intermediate_size}. '
                    f'moe_shared_expert_intermediate_size ({self.moe_shared_expert_intermediate_size}) will be applied.'
                )
        elif self.shared_expert_num > 0:
            self.moe_shared_expert_intermediate_size = self.moe_ffn_hidden_size * self.shared_expert_num

        if self.moe_expert_capacity_factor is not None:
            if self.moe_expert_capacity_factor < 0:
                self.moe_expert_capacity_factor = None
            if self.moe_router_load_balancing_type not in ["sub_seq_aux_loss", "seq_aux_loss", "gbs_aux_loss"]:
                raise ValueError(
                    'moe_expert_capacity_factor only works with supported load balancing types: '
                    'sub_seq_aux_loss, seq_aux_loss, gbs_aux_loss'
                )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.apply_rope_fusion:
            if self.multi_latent_attention:
                raise ValueError("multi_latent_attention does not support apply_rope_fusion.")

        if self.multi_latent_attention and self.rotary_interleaved:
            raise ValueError("rotary_interleaved does not work with multi_latent_attention.")

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std, self.params_dtype)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std,
                self.num_layers,
                self.params_dtype
            )

        if self.num_moe_experts is not None:
            assert not self.add_bias_linear, "Bias is not supported for MoE"

        if self.moe_router_enable_expert_bias and self.moe_router_score_function != "sigmoid":
            raise ValueError(
                "Expert bias for aux-loss-free routing only supports sigmoid score function."
                "Please set --moe-router-score-function sigmoid for sigmoid score function."
            )

        if (
                self.moe_router_topk == 1
                and self.moe_router_score_function == 'softmax'
                and not self.moe_router_pre_softmax
                and self.moe_router_load_balancing_type != 'sinkhorn'
        ):
            # Requires applying softmax before selecting the top-k when k is 1,
            # since softmax on a [num_tokens, 1] would yield a zero gradient.
            raise ValueError("Please use --moe-router-pre-softmax when topk is 1.")

        if self.moe_router_group_topk:
            if not self.moe_router_num_groups:
                raise ValueError(
                    "When using group limited routing, moe_router_num_groups must be specified."
                )
            assert self.num_moe_experts % self.moe_router_num_groups == 0, (
                f"num_moe_experts ({self.num_moe_experts}) should be divisible by "
                f"moe_router_num_groups ({self.moe_router_num_groups})."
            )
            assert self.moe_router_group_topk <= self.moe_router_num_groups, (
                f"moe_router_group_topk ({self.moe_router_group_topk}) should be smaller than "
                f"moe_router_num_groups ({self.moe_router_num_groups})."
            )
            assert self.moe_router_topk % self.moe_router_group_topk == 0, (
                f"`moe_router_topk` must be divisible by `moe_router_group_topk`. "
                f"Got moe_router_topk={self.moe_router_topk} and "
                f"moe_router_group_topk={self.moe_router_group_topk}."
            )

        if (
                self.num_moe_experts is not None
                and self.num_moe_experts >= 32
                and not self.moe_router_dtype
        ):
            logger.warning(
                "Using a large number of experts (e.g. >=32) without fp32 routing. "
                "Consider enabling moe_router_dtype for better numerical stability."
            )

        if self.first_k_dense_replace:
            moe_layer_freq_template = [0] * self.first_k_dense_replace + [1] * (
                    self.num_layers - self.first_k_dense_replace)
            if isinstance(self.moe_layer_freq, int) and not isinstance(self.moe_layer_freq, bool):
                if self.moe_layer_freq > 1:
                    raise ValueError(
                        "Configuration conflict: 'first_k_dense_replace' cannot be "
                        "used together with 'moe_layer_freq > 1'."
                    )
                self.moe_layer_freq = moe_layer_freq_template
            elif isinstance(self.moe_layer_freq, list):
                if self.moe_layer_freq != moe_layer_freq_template:
                    raise ValueError(
                        f"'moe_layer_freq' should be {moe_layer_freq_template}, "
                        f"but got {self.moe_layer_freq}"
                    )
            else:
                raise TypeError("'moe_layer_freq' should be <int> or <list[int]>, "
                                f"but got {type(self.moe_layer_freq)}")
            if self.first_k_dense_replace > self.num_layers:
                raise ValueError(
                    f"'first_k_dense_replace'({self.first_k_dense_replace}) should not be bigger "
                    f"than 'num_layers'({self.num_layers})."
                )
        elif self.moe_layer_freq != 1 or isinstance(self.moe_layer_freq, bool):
            if isinstance(self.moe_layer_freq, int) and not isinstance(self.moe_layer_freq, bool):
                if self.moe_layer_freq > self.num_layers:
                    raise ValueError(
                        f"'moe_layer_freq'({self.moe_layer_freq}) should not be bigger "
                        f"than 'num_layers'({self.num_layers})."
                    )
            elif isinstance(self.moe_layer_freq, list):
                if len(self.moe_layer_freq) != self.num_layers:
                    raise ValueError(
                        f"Length of 'moe_layer_freq'({self.moe_layer_freq}) "
                        f"must be equal to 'num_layers'({self.num_layers})."
                    )
                for num in self.moe_layer_freq:
                    if num not in (0, 1):
                        raise ValueError("Invalid 'moe_layer_freq', "
                                         f"numbers in 'moe_layer_freq'({self.moe_layer_freq}) must be equal to 1 or 0")
            else:
                raise TypeError("'moe_layer_freq' should be <int> or <list[int]>, "
                                f"but got {type(self.moe_layer_freq)}")

        self.is_dryrun = os.environ.get('MS_SIMULATION_LEVEL', '0') != '0'
        if self.is_dryrun:
            if self.num_moe_experts is not None and self.seq_length % self.num_moe_experts != 0:
                raise ValueError(
                    f"When using moe_dry_run, seq_length ({self.seq_length}) must be divisible by "
                    f"num_moe_experts ({self.num_moe_experts})"
                )
            if self.moe_token_dispatcher_type not in ("alltoall", "alltoall_deredundency"):
                raise ValueError(
                    "When using moe_dry_run, moe_token_dispatcher_type must be 'alltoall' or 'alltoall_deredundency'."
                )

        if self.position_embedding_type not in ["rope", "yarn", "none", "relative", "learned_absolute", "partial_rope"]:
            raise ValueError(
                f"The current value of position_embedding_type is {self.position_embedding_type},"
                " but position_embedding_type must be one of: 'rope', 'yarn', 'none', 'relative', 'learned_absolute'."
            )

        if isinstance(self.rope_scaling, dict):
            self.position_embedding_type = (self.rope_scaling.pop("type", None) or
                                            self.rope_scaling.pop("rope_type", None))
            self.rotary_scaling_factor = self.rope_scaling.pop("factor")
            self.max_position_embeddings = self.rope_scaling.pop("original_max_position_embeddings",
                                                                 None) or self.seq_length
            for k, v in self.rope_scaling.items():
                setattr(self, k, v)
            del self.rope_scaling

        if self.position_embedding_type == "none":
            self.nope_layer_interval = None

        if self.nope_layer_interval is None:
            pass
        elif not isinstance(self.nope_layer_interval, int):
            raise TypeError("nope_layer_interval must be a int, "
                            f"but got {type(self.nope_layer_interval)}.")
        elif self.nope_layer_interval <= 0:
            raise ValueError("nope_layer_interval must be larger than 0.")

        if self.bias_swiglu_fusion:
            if self.hidden_act != 'swiglu':
                raise ValueError(
                    "When using bias_swiglu_fusion, hidden_act must be swiglu."
                )
            self.hidden_act = 'fusedswiglu'

        if (self.moe_router_load_balancing_type is not None
                and not isinstance(self.moe_router_load_balancing_type, str)):
            raise TypeError("moe_router_load_balancing_type must be a string, "
                            f"but got {type(self.moe_router_load_balancing_type)}.")

        if self.moe_aux_loss_coeff is not None and not isinstance(self.moe_aux_loss_coeff, (float, int)):
            raise TypeError(f"moe_aux_loss_coeff must be a float or int, but got {type(self.moe_aux_loss_coeff)}.")

        if ms.get_auto_parallel_context("pipeline_scheduler") == "zero_bubble_v":
            if self.virtual_pipeline_model_parallel_size != 2:
                raise ValueError(
                    f"When zero_bubble_v is enabled, pp_interleave_num must be set to 2. "
                    f"But get {self.virtual_pipeline_model_parallel_size}.")
            if self.pipeline_model_parallel_size < 2:
                raise ValueError(
                    f"When zero_bubble_v is enabled, pp must be greater than or equal to 2. "
                    f"But get {self.pipeline_model_parallel_size}.")
            if self.micro_batch_num < 2 * self.pipeline_model_parallel_size:
                raise ValueError(
                    f"When zero_bubble_v is enabled, micro_batch_num({self.micro_batch_num}) >= 2 * stage_num"
                    f"({self.pipeline_model_parallel_size}) must be met.")
            if isinstance(self.recompute, (list, tuple)):
                if all(isinstance(item, (int, bool)) for item in self.recompute) or len(self.recompute) < 2:
                    raise ValueError(
                        "When zero_bubble_v is enabled, "
                        "'recompute' must provide explicit 2D configuration for each interleave, "
                        "such as [[stage0_recompute, stage1_recompute], [stage0_recompute, stage1_recompute]].")

        if self.recompute_slice_activation:
            raise ValueError("For recompute, `recompute_slice_activation` is not supported in Mcore.")

        self._validate_param_init_std_rules()

        if self.window_size is not None:
            if self.window_size[0] < -1 or self.window_size[1] < -1:
                raise ValueError("When number in window_size should not lower than -1."
                                 f"But get {self.window_size[0]} and {self.window_size[1]}")

        if self.model_architecture not in [model_architecture.value for model_architecture in ModelArchitecture]:
            raise ValueError("You should set the model_architecture in 'decoder_only' "
                             f"or 'yoco'. But get {self.model_architecture}.")

        if self.model_architecture == ModelArchitecture.DECODER_ONLY:
            self.num_decoder_layers = self.num_layers
            self.num_encoder_layers = 0
        if self.model_architecture == ModelArchitecture.YOCO:
            if self.num_encoder_layers is None and self.num_decoder_layers is None:
                raise ValueError("While use the 'yoco' model structure, "
                                 "You should set the num_encoder_layers or the num_decoder_layers.")
            self.num_decoder_layers = self.num_decoder_layers \
                if self.num_decoder_layers else self.num_layers - self.num_encoder_layers
            self.num_encoder_layers = self.num_encoder_layers \
                if self.num_encoder_layers else self.num_layers - self.num_decoder_layers

        if (self.num_encoder_layers and self.num_decoder_layers
                and self.num_encoder_layers + self.num_decoder_layers != self.num_layers):
            raise ValueError("The combination of  num_encoder_layers and num_decoder_layers "
                             f"should be equal to num_layers. But get num_encoder_layers: {self.num_encoder_layers}, "
                             f"num_decoder_layers: {self.num_decoder_layers}, num_layers: {self.num_layers}.")

    def _validate_param_init_std_rules(self):
        """Validate and compile decoupling initialization rules."""
        rules = self.param_init_std_rules

        # Only process if rules are provided (non-empty list)
        if rules:
            # Ensure rules is a list
            if not isinstance(rules, list):
                raise TypeError(
                    f"param_init_std_rules must be a list, "
                    f"but got {type(rules)}(value: {rules})"
                )

            for idx, rule in enumerate(rules):
                if not isinstance(rule, dict):
                    raise TypeError(f"Rule {idx} is not a dict, got {type(rule)}")

                target = rule.get("target")
                init_std = rule.get("init_method_std")

                # Validate 'target' field
                if not isinstance(target, str):
                    raise TypeError(f"Rule {idx}: 'target' must be a string, but got {type(target)}")

                # Validate 'init_method_std' field
                if not isinstance(init_std, (int, float)):
                    raise TypeError(
                        f"Rule {idx}: 'init_method_std' must be a number, "
                        f"but got {type(init_std)}"
                    )
                if init_std < 0:
                    raise ValueError(f"Rule {idx}: 'init_method_std' must be >= 0, but got {init_std}")

                # Compile the regex pattern and replace the string in-place
                compiled_pattern = re.compile(target)
                rule["target"] = compiled_pattern


@dataclass
class ModelArchitecture(enum.Enum):
    """
        Enumeration representing different types of model architecture.

        Attributes:
            DECODER_ONLY: Represents the model architecture only has the decoder layer.
            YOCO: Represents the model architecture is yoco.
    """
    DECODER_ONLY = "decoder_only"
    YOCO = "yoco"


@dataclass
class MLATransformerConfig(TransformerConfig):
    """
    Configuration object for MindSpore Transformer's Multi-Latent Attention (MLA) transformers.

    The initialization function has an argument for each parameter, including those in ModelParallelConfig.
    Included YaRN RoPE parameters that is fused in MLA.
    """
    multi_latent_attention: bool = field(
        default=True,
        metadata={
            "description": "Whether to use Multi-Latent Attention.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    q_lora_rank: int = field(
        default=512,
        metadata={
            "description": "Rank of Query tensor's low rank representation.",
            "usage": ParamUsage.COMMON,
            "source": [ParamSource.HF, ParamSource.MEGATRON],
            "mode": ParamMode.COMMON
        }
    )

    kv_lora_rank: int = field(
        default=512,
        metadata={
            "description": "Rank of Key and Value tensors' low rank representation.",
            "usage": ParamUsage.COMMON,
            "source": [ParamSource.HF, ParamSource.MEGATRON],
            "mode": ParamMode.COMMON
        }
    )

    qk_head_dim: int = field(
        default=128,
        metadata={
            "description": "Dimension of the head in the QK projection.q_head_dim = qk_head_dim + qk_pos_emb_head_dim.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    qk_pos_emb_head_dim: int = field(
        default=64,
        metadata={
            "description": "Dimension of the position embedding in the QK projection.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    v_head_dim: int = field(
        default=128,
        metadata={
            "description": "Dimension of the head in the V projection.",
            "usage": ParamUsage.COMMON,
            "source": [ParamSource.HF, ParamSource.MEGATRON],
            "mode": ParamMode.COMMON
        }
    )

    normalization: str = field(
        default="RMSNorm",
        metadata={
            "description": "Default normalization layer for MLA models is RMSNorm.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    rope_type: str = field(
        default="yarn",
        metadata={
            "description": "Type of RoPE to use. Default to yarn, options are rope and yarn.",
            "usage": ParamUsage.INFERENCE,
            "source": [ParamSource.HF, ParamSource.MEGATRON],
            "mode": ParamMode.COMMON
        }
    )

    rotary_percent: float = field(
        default=1.0,
        metadata={
            "description": "Rotary percent for the rotary embeddings, used by rope.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    rotary_scaling_factor: float = field(
        default=40.0,
        metadata={
            "description": "Rotary scaling factor for the rotary embeddings, used by yarn.",
            "usage": ParamUsage.COMMON,
            "source": ParamSource.MEGATRON,
            "mode": ParamMode.COMMON
        }
    )

    max_position_embeddings: int = field(
        default=4096,
        metadata={
            "description": "Maximum position embeddings for the original model, used by yarn.",
            "usage": ParamUsage.COMMON,
            "source": [ParamSource.HF, ParamSource.MEGATRON],
            "mode": ParamMode.COMMON
        }
    )

    beta_fast: float = field(
        default=32.0,
        metadata={
            "description": "Beta fast for YaRN RoPE, used by yarn.",
            "usage": ParamUsage.COMMON,
            "source": [ParamSource.HF, ParamSource.MEGATRON],
            "mode": ParamMode.COMMON
        }
    )

    beta_slow: float = field(
        default=1.0,
        metadata={
            "description": "Beta fast for YaRN RoPE, used by yarn.",
            "usage": ParamUsage.COMMON,
            "source": [ParamSource.HF, ParamSource.MEGATRON],
            "mode": ParamMode.COMMON
        }
    )

    mscale: float = field(
        default=0.707,
        metadata={
            "description": "Mscale for YaRN RoPE in Multi-Latent Attention, used by yarn.",
            "usage": ParamUsage.COMMON,
            "source": [ParamSource.HF, ParamSource.MEGATRON],
            "mode": ParamMode.COMMON
        }
    )

    mscale_all_dim: float = field(
        default=0.707,
        metadata={
            "description": "Mscale all dimensions for YaRN RoPE in Multi-Latent Attention, used by yarn.",
            "usage": ParamUsage.COMMON,
            "source": [ParamSource.HF, ParamSource.MEGATRON],
            "mode": ParamMode.COMMON
        }
    )

    def __post_init__(self):
        """
        Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more
        details.
        """
        super().__post_init__()

        if self.experimental_attention_variant == 'dsa':
            if self.kv_lora_rank != 512 or self.qk_pos_emb_head_dim != 64 or \
                    self.dsa_indexer_head_dim != 128 or self.dsa_indexer_n_heads != 64:
                raise ValueError("CurrentLy, when `experimental_attention_variant` == 'dsa', "
                                 "`kv_lora_rank` only supports 512, `qk_pos_emb_head_dim` only supports 64, "
                                 "`dsa_indexer_head_dim` only supports 128, `dsa_indexer_n_heads` only supports 64.")


default_transformer_config = TransformerConfig(num_attention_heads=1, num_layers=1)
