# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This file is derived from Megatron-LM and adapted for MindSpore.
# Modifications:
#     - Adapted to MindSpore framework: replaced torch with mindspore, nn.Module with nn.Cell.
#     - Used mindspore.mint and mindspore.ops for tensor operations.
#     - Integrated with mindformers.parallel_core for module specification and building.
#     - Added support for TND input layout.
#     - Utilized MindFormers' Rotary Embedding implementation.
"""
Multi-head Latent Attention (MLA) mechanism with KV compression and rotary position encoding.

This module implements the Multi-head Latent Attention mechanism with low-rank compression
for KV projections and rotary position encoding support.
"""
# MindSpore ``Cell.construct`` intentionally exposes operator-specific signatures.
# pylint: disable=arguments-differ
from dataclasses import dataclass
from typing import Union
import math

from mindspore import nn, Tensor, mint, ops
from hyper_parallel import DTensor

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.pynative.base_models.common.embeddings.rope_utils import ApplyRotaryPosEmb
from mindformers.pynative.base_models.common.embeddings.yarn_rotary_pos_embedding import _yarn_get_mscale
from mindformers.pynative.layers.identity_op import IdentityOp
from mindformers.pynative.dtensor_compat import local_shard


@dataclass
class MLASelfAttentionSubmodules:
    """
    Dataclass for MLA self-attention layer submodules.

    This dataclass defines the submodules required for building the MLA self-attention layer.

    Attributes:
        linear_qkv: Linear layer for combined query, key, and value projections.
            If q_lora_rank is not None, it concatenates linear_q_down_proj and linear_kv_down_proj;
            otherwise, it concatenates linear_q_proj and linear_kv_down_proj.
        linear_qb: Linear layer for query up projection.
        linear_kvb: Linear layer for key-value up projection.
        core_attention: Core attention mechanism implementation.
        linear_proj: Linear layer for final attention output projection.
        q_layernorm: Layer normalization for query projections (optional).
        k_layernorm: Layer normalization for key projections (optional).
    """
    linear_qkv: Union[ModuleSpec, type] = None
    linear_qb: Union[ModuleSpec, type] = None
    linear_kvb: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class MultiLatentAttention(nn.Cell):
    """
    Multi-head Latent Attention (MLA) with KV compression and rotary position encoding.

    Base class for Multi-head Latent Attention mechanism that implements KV compression
    and supports rotary position encoding. This class provides the core functionality
    for both self-attention and cross-attention variants.

    Args:
        config: Configuration object with MLA parameters.
        submodules: Submodules configuration for building the attention layer.
        layer_number: Layer index in the transformer stack.
        attention_type: Type of attention ("self" or "cross").
    """

    def __init__(
            self,
            config: MLATransformerConfig,
            submodules: Union[MLASelfAttentionSubmodules],
            layer_number: int,
            attention_type: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.layer_index = max(1, layer_number)
        self.attention_type = attention_type

        # model structure config
        self.use_flash_attention = self.config.use_flash_attention
        self.use_ring_attention = self.config.use_ring_attention
        self.use_eod_attn_mask_compression = self.config.use_eod_attn_mask_compression
        self.use_attn_mask_compression = self.config.use_attn_mask_compression
        self.seq_length = self.config.seq_length
        self.num_attention_heads = self.config.num_attention_heads
        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads
        self.qk_head_dim = self.config.qk_head_dim
        self.qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim
        self.q_head_dim = self.qk_head_dim + self.qk_pos_emb_head_dim
        self.kv_lora_rank = self.config.kv_lora_rank
        self.v_head_dim = self.config.v_head_dim
        self.input_layout = self.config.input_layout
        self.compute_dtype = self.config.compute_dtype
        self.use_tnd = config.input_layout == "TND"
        zero_pad_length = self.q_head_dim - self.v_head_dim
        if zero_pad_length < 0:
            raise ValueError("qk_head_dim + qk_pos_emb_head_dim should not less than v_head_dim")

        mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_index,
            softmax_scale=self.softmax_scale,
            **self.core_attention_extra_kwargs(),
        )

        self.linear_proj = build_module(
            submodules.linear_proj,
            input_size=self.query_projection_size,
            output_size=self.config.hidden_size,
            compute_dtype=config.compute_dtype,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
        )

        self.shape = ops.shape
        self.reshape = mint.reshape
        self.transpose = mint.transpose
        self.permute = mint.permute
        self.cast = ops.cast
        self.q_handoff = IdentityOp()
        self.k_handoff = IdentityOp()
        self.v_handoff = IdentityOp()

    @staticmethod
    def _to_local_tensor(tensor):
        """Return the local tensor for a DTensor, otherwise pass through."""
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    def apply_qk_clip(self, scales, fp32_param_map):
        """Scale this layer's MLA projections by its per-head QK-clip factors, with no collective."""
        root = mint.sqrt(scales)
        q_rows = self._row_scales(((self.qk_head_dim, root), (self.qk_pos_emb_head_dim, scales)))
        if self.config.q_lora_rank is None:
            # Without a q LoRA the query heads lead the concat qkv weight; its tail is not scaled.
            tail = self.linear_qkv.output_size - int(q_rows.shape[0])
            q_rows = mint.cat([q_rows, mint.ones((tail,), dtype=q_rows.dtype)])
            self._clip_projection(self.linear_qkv, q_rows, fp32_param_map)
        else:
            self._clip_projection(self.linear_qb, q_rows, fp32_param_map)
        self._clip_projection(
            self.linear_kvb,
            self._row_scales(((self.qk_head_dim, root), (self.v_head_dim, mint.ones_like(scales)))),
            fp32_param_map,
        )

    def _clip_projection(self, linear, row_scales, fp32_param_map):
        """Multiply one projection's local weight rows by their share of ``row_scales``."""
        # Muon keeps fp32 master copies; scale the value the optimizer reads.
        weight = fp32_param_map.get(linear.weight.name, linear.weight)
        local = self._to_local_tensor(weight)
        shards = int(row_scales.shape[0]) // int(local.shape[0])
        local.mul_(row_scales.reshape(shards, -1, 1)[local_shard(weight, 0, shards)])

    @staticmethod
    def _row_scales(segments):
        """Expand ``(rows, per-head factor)`` segments into one factor per weight row."""
        per_head = mint.cat(
            [mint.tile(mint.unsqueeze(factor, 1), (1, rows)) for rows, factor in segments],
            dim=1,
        )
        return per_head.reshape(-1)

    def core_attention_extra_kwargs(self):
        """Extra kwargs for the core-attention build. Empty by default.

        ``layer_number`` is handed to ``core_attention`` as ``self.layer_index``, which
        clamps 0 to 1 and therefore makes global layers 0 and 1 indistinguishable. A
        subclass whose core attention needs the *true* layer index must pass it through
        this hook instead; ``self.layer_number`` is already set when this runs.
        """
        return {}

    def construct(self, x: Tensor, attention_mask=None, rotary_pos_emb=None,
                  prefix_keys_values=None, pad_zeros=None, actual_seq_len=None, mscale=1.0,
                  rotary_cos_sin=None):
        """
        Forward pass of the Multi-head Latent Attention mechanism.

        Args:
            x: Input tensor with shape (seq_length, batch_size, hidden_size).
            attention_mask: Attention mask tensor (optional).
            rotary_pos_emb: Rotary position embedding frequencies tensor (optional).
            pad_zeros: Padding zeros tensor (not used).
            actual_seq_len: Actual sequence length for EOD mask compression (optional).
            mscale: Rotary magnitude scaling (yarn interface). Default: 1.0

        Returns:
            Tensor: Output tensor with shape (seq_length, batch_size, hidden_size).
        """
        if prefix_keys_values:
            raise NotImplementedError("prefix_keys_values is not supported for now.")
        if pad_zeros:
            raise NotImplementedError("pad_zeros is not supported for now.")
        seq_len, bs, _ = self.shape(x)
        query, key, value = self.get_query_key_value_tensors(
            x, rotary_pos_emb=rotary_pos_emb, mscale=mscale, rotary_cos_sin=rotary_cos_sin)
        # The attention output carries q's sequence length -- ``seq_len`` when the input
        # is full, or the (gathered) full sequence when q was sequence-sharded upstream.
        attn_seq_len = self.shape(query)[0]
        if self.use_flash_attention:
            if self.input_layout == "TND":
                if actual_seq_len is None:
                    raise ValueError("TND attention requires actual_seq_len.")
                context_layer = self.core_attention(
                    query, key, value, attention_mask,
                    actual_seq_qlen=actual_seq_len, actual_seq_kvlen=actual_seq_len,
                )
                full_seq_len = attn_seq_len // bs
                attn_out = self.reshape(context_layer, (bs, full_seq_len, -1))
                attn_out = self.transpose(attn_out, 0, 1)
                if hasattr(attn_out, "contiguous"):
                    attn_out = attn_out.contiguous()
            elif self.use_eod_attn_mask_compression:
                context_layer = self.core_attention(
                    query, key, value, attention_mask,
                    actual_seq_qlen=actual_seq_len, actual_seq_kvlen=actual_seq_len
                )
                attn_out = self.reshape(context_layer, (bs, seq_len, -1))
                attn_out = self.transpose(attn_out, 0, 1)
            else:
                context_layer = self.core_attention(
                    query, key, value, attention_mask,
                )
                attn_out = self.reshape(context_layer, (attn_seq_len, bs, -1))
                if hasattr(attn_out, "contiguous"):
                    attn_out = attn_out.contiguous()
        else:
            attn_out = self.core_attention(query, key, value, attention_mask)

        output = self.linear_proj(attn_out)
        return output

    def sbh2tnd(self, x):
        """
        Convert a tensor from SBH/SBND layout to TND layout.

        Args:
            x: Input tensor with SBH/SBND layout.

        Returns:
            Tensor: Output tensor with TND layout.
        """
        if x.ndim != 4:
            raise ValueError(f"TND conversion expects SBND input, but got {x.ndim} dimensions")
        seq_len, bs, num_heads = x.shape[:3]
        x = self.transpose(x, 0, 1)
        x = self.reshape(x, (bs * seq_len, num_heads, -1))
        return x

class MLASelfAttention(MultiLatentAttention):
    """
    MLA Self-attention layer implementation.

    This class implements the MLA self-attention layer following the same structure as Mindspeed A2.
    It inherits from MultiLatentAttention and provides self-attention specific functionality.

    Args:
        config: Configuration object with MLA parameters.
        submodules: Submodules configuration for building the self-attention layer.
        layer_number: Layer index in the transformer stack.

    Inputs:
        x: Input tensor with shape [seq_length, batch_size, hidden_size].

    Outputs:
        Tensor: Output tensor with shape [seq_length, batch_size, hidden_size].
    """

    def __init__(
            self,
            config: MLATransformerConfig,
            submodules: MLASelfAttentionSubmodules,
            layer_number: int,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type="self",
        )
        self.use_tnd = config.input_layout == "TND"
        self.split = mint.split
        self.tile_kv = mint.tile
        self.cat = mint.cat
        self.apply_rotary_emb_q = ApplyRotaryPosEmb(config)
        self.apply_rotary_emb_k = ApplyRotaryPosEmb(config)
        self.reshape = mint.reshape

        if self.config.q_lora_rank is None:
            self.q_rank = self.config.num_attention_heads * self.q_head_dim
            self.q_layernorm = None
        else:
            self.q_rank = self.config.q_lora_rank
            if submodules.q_layernorm is not None:
                self.q_layernorm = build_module(
                    submodules.q_layernorm,
                    dim=self.config.q_lora_rank,
                    eps=self.config.layernorm_epsilon,
                    compute_dtype=config.layernorm_compute_dtype
                )
            else:
                self.q_layernorm = None

            self.linear_qb = build_module(
                submodules.linear_qb,
                input_size=self.config.q_lora_rank,
                output_size=self.config.num_attention_heads * self.q_head_dim,
                compute_dtype=config.compute_dtype,
                init_method=self.config.init_method,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            input_size=self.config.hidden_size,
            output_size=self.q_rank + self.kv_lora_rank + self.qk_pos_emb_head_dim,
            compute_dtype=config.compute_dtype,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
        )

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                dim=self.kv_lora_rank,
                eps=self.config.layernorm_epsilon,
                compute_dtype=config.layernorm_compute_dtype
            )
        else:
            self.k_layernorm = None

        self.linear_kvb = build_module(
            submodules.linear_kvb,
            input_size=self.kv_lora_rank,
            output_size=self.config.num_attention_heads * (
                    self.q_head_dim - self.qk_pos_emb_head_dim + self.v_head_dim),
            compute_dtype=config.compute_dtype,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
        )

        self.linear_proj = build_module(
            submodules.linear_proj,
            input_size=self.query_projection_size,
            output_size=self.config.hidden_size,
            compute_dtype=config.compute_dtype,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
        )

    def get_query_key_value_tensors(self, hidden_states, rotary_pos_emb=None, mscale=1.0, rotary_cos_sin=None):
        """
        Derive query, key, and value tensors from hidden states.

        The handoff modules are parameter-free boundaries by default. Parallel
        styles may replace them to start asynchronous transfers at the exact
        points where Q/K/V become the tensors consumed by ``core_attention``.

        Args:
            hidden_states: Input hidden states tensor with shape [seq_length, batch_size, hidden_size].
            rotary_pos_emb: Rotary position embedding frequencies tensor (optional).
            mscale: Rotary magnitude scaling (yarn interface). Default: 1.0

        Returns:
            tuple: A tuple containing query, key, and value tensors.
                - query: Query tensor with shape [seq_length, batch_size, num_heads, head_dim].
                - key: Key tensor with shape [seq_length, batch_size, num_heads, head_dim].
                - value: Value tensor with shape [seq_length, batch_size, num_heads, v_head_dim].
        """
        # linear_qkv and the latent norms run on the local sequence shard. The
        # head-sharded up-projections gather their latent inputs immediately before
        # matmul; k_pe is gathered separately at the rotary-k boundary.
        seq_len, bs, _ = self.shape(hidden_states)
        qkv_combo = self.linear_qkv(hidden_states)

        q_a, compressed_kv, k_pe = self.split(
            qkv_combo,
            [
                self.q_rank,
                self.kv_lora_rank,
                self.qk_pos_emb_head_dim,
            ],
            dim=-1,
        )

        if self.q_layernorm is not None:
            q_a = self.q_layernorm(q_a)
            q = self.linear_qb(q_a)
            q = self.reshape(q, (self.shape(q)[0], bs, -1, self.q_head_dim))

            q_nope, q_pe = self.split(
                q, [self.qk_head_dim, self.qk_pos_emb_head_dim], dim=-1
            )

        else:
            q = self.reshape(q_a, (self.shape(q_a)[0], bs, -1, self.q_head_dim))
            q_nope, q_pe = self.split(
                q, [self.qk_head_dim, self.qk_pos_emb_head_dim], dim=-1
            )

        if rotary_pos_emb is not None:
            q_pe = self.apply_rotary_emb_q(
                q_pe,
                rotary_pos_emb,
                mscale=mscale,
                rotary_interleaved=self.config.rotary_interleaved,
                multi_latent_attention=self.config.multi_latent_attention,
                cos_sin=rotary_cos_sin
            )

        query = self.cat([q_nope, q_pe], 3)
        query = self.cast(query, self.compute_dtype)
        if self.input_layout == "TND":
            query = self.sbh2tnd(query)
        query = self.q_handoff(query)

        # k_pe stays sequence-sharded here (local seq); gathered to the full
        # sequence below so it lines up with the full-seq, head-sharded k_nope.
        k_pe = self.reshape(k_pe, (seq_len, bs, 1, self.qk_pos_emb_head_dim))
        compressed_kv_norm = self.k_layernorm(compressed_kv)
        kv = self.linear_kvb(compressed_kv_norm)  # colwise up-proj -> full seq
        kv = self.reshape(kv, (
            self.shape(kv)[0],
            bs,
            -1,
            self.qk_head_dim + self.v_head_dim,
        ))

        k_nope, value = self.split(kv, [self.qk_head_dim, self.v_head_dim], dim=-1)
        value = self.cast(value, self.compute_dtype)
        if self.input_layout == "TND":
            value = self.sbh2tnd(value)
        value = self.v_handoff(value)

        # k_pe is sequence-gathered to the full sequence by the parallelize pre-hook on
        # apply_rotary_emb_k (a no-op single-card), so after rotary it lines up with the
        # full-seq k_nope; broadcast its single rope head to k_nope's head count.
        if rotary_pos_emb is not None:
            k_pe = self.apply_rotary_emb_k(
                k_pe,
                rotary_pos_emb,
                mscale=mscale,
                rotary_interleaved=self.config.rotary_interleaved,
                multi_latent_attention=self.config.multi_latent_attention,
                cos_sin=rotary_cos_sin
            )

        k_pe = self.tile_kv(k_pe, (1, 1, self.shape(k_nope)[2], 1))
        key = self.cat([k_nope, k_pe], 3)
        key = self.cast(key, self.compute_dtype)
        if self.input_layout == "TND":
            key = self.sbh2tnd(key)
        key = self.k_handoff(key)

        return query, key, value
