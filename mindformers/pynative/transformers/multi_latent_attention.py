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
for KV projections and rotary position encoding support. Includes DSA (DeepSeek Sparse
Attention) support for pynative mode.
"""
# MindSpore ``Cell``/``_Function`` subclasses intentionally use operator-specific signatures.
# pylint: disable=arguments-differ,abstract-method
from dataclasses import dataclass
from typing import Union
import math

from mindspore import nn, Tensor, mint, ops
from mindspore.common._grad_function import _Function
from hyper_parallel import DTensor, SkipDTensorDispatch
from hyper_parallel.core.dtensor.dtensor import distribute_tensor

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.pynative.base_models.common.embeddings.rope_utils import ApplyRotaryPosEmb
from mindformers.pynative.base_models.common.embeddings.yarn_rotary_pos_embedding import _yarn_get_mscale
from mindformers.pynative.layers.identity_op import IdentityOp
from mindformers.pynative.dtensor_compat import inplace_copy
from mindformers.pynative.transformers.experimental_attention_variant.indexer import _IndexerLossAutoScaler
from mindformers.pynative.transformers.experimental_attention_variant.utils import save_to_indexer_losses_tracker


class _DSADetachFunction(_Function):
    """Identity in forward, zero gradient in backward for DSA detach boundaries."""

    @staticmethod
    def forward(ctx, tensor):
        """Return the input unchanged while recording no backward state."""
        del ctx
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Stop the gradient at the detach boundary."""
        del ctx
        return mint.zeros_like(grad_output)


class _AbsorbMatmul(_Function):
    """Memory-frugal MLA weight-absorb matmul: ``out[..., h, r] = sum_d x[..., h, d] * w[h, d, r]``.

    The naive ``mint.matmul(mint.unsqueeze(x, -2), w)`` broadcasts the per-head
    weight ``w`` across every token, so autograd materialises a per-token outer
    product of shape ``[tokens, heads, d, r]`` in backward (e.g. 32 GB at
    seq_local=8192 for the q/v absorb). Here forward/backward are explicit
    batched-matmuls over the head axis, and the ``d_w`` reduction contracts the
    token axis *inside* the bmm, so nothing larger than the activation is ever
    held. This mirrors PR#8229's graph-mode ``ops.Morph`` absorb, ported to
    pynative. Mathematically identical to the unsqueeze+matmul it replaces.
    """

    @staticmethod
    def forward(ctx, x, w):
        """Apply the head-wise absorbed matmul without token-wise weight expansion."""
        # x: [*lead, H, D], w: [H, D, R] -> [*lead, H, R]
        lead = tuple(x.shape[:-2])
        h, d = x.shape[-2], x.shape[-1]
        r = w.shape[-1]
        t = 1
        for s in lead:
            t *= s
        ctx.x = x
        ctx.w = w
        ctx.lead = lead
        ctx.hdr = (h, d, r)
        xh = mint.permute(mint.reshape(x, (t, h, d)), (1, 0, 2))  # [H, T, D]
        out = mint.bmm(xh, w)                                     # [H, T, R]
        out = mint.permute(out, (1, 0, 2))                        # [T, H, R]
        return mint.reshape(out, lead + (h, r))

    @staticmethod
    def backward(ctx, grad_output):
        """Compute activation and weight gradients with head-wise batched matmuls."""
        h, d, r = ctx.hdr
        t = 1
        for s in ctx.lead:
            t *= s
        w = ctx.w
        gh = mint.permute(mint.reshape(grad_output, (t, h, r)), (1, 0, 2))  # [H, T, R]
        xh = mint.permute(mint.reshape(ctx.x, (t, h, d)), (1, 0, 2))        # [H, T, D]
        # d_x = g @ w^T : [H, T, R] @ [H, R, D] -> [H, T, D]
        d_x = mint.bmm(gh, mint.permute(w, (0, 2, 1)))
        d_x = mint.reshape(mint.permute(d_x, (1, 0, 2)), ctx.lead + (h, d))
        # d_w = x^T @ g : [H, D, T] @ [H, T, R] -> [H, D, R]  (token axis contracted in-kernel)
        d_w = mint.bmm(mint.permute(xh, (0, 2, 1)), gh)
        return d_x, d_w


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
        self.use_dsa = getattr(config, 'experimental_attention_variant', None) == "dsa"
        self.sparse_loss = getattr(config, 'dsa_indexer_use_sparse_loss', False) if self.use_dsa else False

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
        self.dsa_indexer_key_handoff = IdentityOp()
        self.dsa_value_handoff = IdentityOp()
        self.dsa_loss_key_indexer_handoff = IdentityOp()

    @staticmethod
    def _to_full_tensor_with_layout(tensor):
        """Return full tensor and original DTensor layout metadata when needed."""
        if not isinstance(tensor, DTensor):
            return tensor, None
        return tensor.full_tensor(), (tensor.device_mesh, tensor.placements)

    @staticmethod
    def _to_local_tensor(tensor):
        """Return the local tensor for a DTensor, otherwise pass through."""
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    @staticmethod
    def _restore_tensor_layout(tensor, layout):
        """Restore a tensor back to the original DTensor local layout."""
        if layout is None:
            return tensor
        device_mesh, placements = layout
        return distribute_tensor(tensor, device_mesh, placements).to_local()

    @staticmethod
    def _get_scale_broadcast(scales, head_dim):
        """Broadcast per-head QK-clip scales to projection weight rows."""
        scale_broadcast = mint.tile(
            mint.unsqueeze(scales, 1), (1, head_dim)
        ).reshape(-1)
        return mint.unsqueeze(scale_broadcast, 1)

    @staticmethod
    def _local_head_count(tensor, head_dims):
        """Return the number of complete local heads represented by tensor rows."""
        rows = int(tensor.shape[0])
        rows_per_head = sum(int(dim) for dim in head_dims)
        if rows_per_head <= 0 or rows % rows_per_head != 0:
            return None
        return rows // rows_per_head

    def _get_qk_clip_weight(self, param_name, fp32_param_map=None):
        """Return the MLA projection weight addressed by an optimizer parameter name.

        When ``fp32_param_map`` is provided (Muon mixed-precision path), the fp32
        master copy is returned so QK-clip scales the value the optimizer reads,
        not the bf16/fp16 model parameter that gets overwritten on the next
        copy-back.
        """
        for attr in ("linear_qb", "linear_kvb", "linear_qkv"):
            if f"self_attention.{attr}.weight" in param_name:
                param = getattr(getattr(self, attr, None), "weight", None)
                if param is None:
                    return None
                if fp32_param_map is not None:
                    fp32_param = fp32_param_map.get(param.name)
                    if fp32_param is not None:
                        return fp32_param
                return param
        return None

    def can_apply_qk_clip_to_local_weights(self, scales):
        """Check whether local MLA projection shards align with local head scales."""
        if self.config.q_lora_rank is None:
            return False

        scale_heads = int(scales.shape[0])
        q_weight = getattr(getattr(self, "linear_qb", None), "weight", None)
        kv_weight = getattr(getattr(self, "linear_kvb", None), "weight", None)
        if q_weight is None or kv_weight is None:
            return False

        q_local = self._to_local_tensor(q_weight)
        kv_local = self._to_local_tensor(kv_weight)
        q_heads = self._local_head_count(q_local, (self.qk_head_dim, self.qk_pos_emb_head_dim))
        kv_heads = self._local_head_count(kv_local, (self.qk_head_dim, self.v_head_dim))
        return q_heads == scale_heads and kv_heads == scale_heads

    def try_apply_qk_clip_to_local_weights(self, param_prefix, scales, split_fn, merge_fn,
                                           fp32_param_map=None):
        """Apply QK-clip on local TP shards without gathering full projection weights."""
        if not self.can_apply_qk_clip_to_local_weights(scales):
            return False

        self._apply_qk_clip_to_local_weight(
            f"{param_prefix}.linear_qb.weight", scales, split_fn, merge_fn,
            fp32_param_map=fp32_param_map)
        self._apply_qk_clip_to_local_weight(
            f"{param_prefix}.linear_kvb.weight", scales, split_fn, merge_fn,
            fp32_param_map=fp32_param_map)
        return True

    def _apply_qk_clip_to_local_weight(self, param_name, scales, split_fn, merge_fn,
                                       fp32_param_map=None):
        """Apply QK-clip to one local MLA projection shard."""
        param = self._get_qk_clip_weight(param_name, fp32_param_map=fp32_param_map)
        if param is None:
            return
        local_param = self._to_local_tensor(param)

        if "self_attention.linear_qb.weight" in param_name:
            with SkipDTensorDispatch():
                nope, pe = split_fn(param_name, local_param)
                nope = nope * self._get_scale_broadcast(mint.sqrt(scales), self.qk_head_dim)
                pe = pe * self._get_scale_broadcast(scales, self.qk_pos_emb_head_dim)
                weights = merge_fn(param_name, [nope, pe])
        elif "self_attention.linear_kvb.weight" in param_name:
            with SkipDTensorDispatch():
                k_nope, v = split_fn(param_name, local_param)
                k_nope = k_nope * self._get_scale_broadcast(mint.sqrt(scales), self.qk_head_dim)
                weights = merge_fn(param_name, [k_nope, v])
        else:
            return

        with SkipDTensorDispatch():
            inplace_copy(param, weights)

    def apply_qk_clip_to_weights(self, param_prefix, scales, split_fn, merge_fn,
                                 fp32_param_map=None):
        """Apply QK-clip scaling to all MLA projection weights owned by this layer."""
        for weight_name in ("linear_qb.weight", "linear_kvb.weight", "linear_qkv.weight"):
            self.apply_qk_clip_to_weight(f"{param_prefix}.{weight_name}", scales, split_fn, merge_fn,
                                         fp32_param_map=fp32_param_map)

    def apply_qk_clip_to_weight(self, param_name, scales, split_fn, merge_fn,
                                fp32_param_map=None):
        """Apply QK-clip scaling to an MLA projection weight in-place."""
        if "self_attention.linear_qkv.weight" in param_name and self.config.q_lora_rank is not None:
            return

        param = self._get_qk_clip_weight(param_name, fp32_param_map=fp32_param_map)
        if param is None:
            return

        full_param, param_layout = self._to_full_tensor_with_layout(param)

        if "self_attention.linear_qb.weight" in param_name:
            with SkipDTensorDispatch():
                nope, pe = split_fn(param_name, full_param)
                nope = nope * self._get_scale_broadcast(mint.sqrt(scales), self.qk_head_dim)
                pe = pe * self._get_scale_broadcast(scales, self.qk_pos_emb_head_dim)
                weights = merge_fn(param_name, [nope, pe])
        elif "self_attention.linear_kvb.weight" in param_name:
            with SkipDTensorDispatch():
                k_nope, v = split_fn(param_name, full_param)
                k_nope = k_nope * self._get_scale_broadcast(mint.sqrt(scales), self.qk_head_dim)
                weights = merge_fn(param_name, [k_nope, v])
        elif "self_attention.linear_qkv.weight" in param_name:
            # Concat MLA. With LoRA: [q_down, kv_lora, k_pe] has no per-head structure to scale.
            parts = split_fn(param_name, full_param)
            if len(parts) != 4:
                return
            q_nope, q_pe, kv_lora, k_pe = parts
            with SkipDTensorDispatch():
                q_nope = q_nope * self._get_scale_broadcast(mint.sqrt(scales), self.qk_head_dim)
                q_pe = q_pe * self._get_scale_broadcast(scales, self.qk_pos_emb_head_dim)
                weights = merge_fn(param_name, [q_nope, q_pe, kv_lora, k_pe])
        else:
            return

        weights = self._restore_tensor_layout(weights, param_layout)
        with SkipDTensorDispatch():
            inplace_copy(param, weights)

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
    def dsa_forward(self, x, attention_mask=None, rotary_pos_emb=None,
                    actual_seq_len=None, attention_loss=0.):
        """
        Forward pass with DSA (DeepSeek Sparse Attention).
        Two-stage training:
        - Dense stage (sparse_loss=False): Full attention with indexer warm-up.
        - Sparse stage (sparse_loss=True): Top-k sparse attention with MQA absorb.

        Args:
            x: Input tensor with shape (seq_length, batch_size, hidden_size).
            attention_mask: Attention mask tensor (optional).
            rotary_pos_emb: Rotary position embedding tensor (optional).
            actual_seq_len: Actual sequence length (optional).
            attention_loss: Accumulated attention loss from previous layers.

        Returns:
            Tuple[Tensor, Tensor]: (output, attention_loss).
        """
        ori_dtype = x.dtype
        seq_len, bs, _ = self.shape(x)
        qkv_combo = self.linear_qkv(x)

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
        q_compress = q_a
        x_detached = _DSADetachFunction.apply(x)
        q_compress_detached = _DSADetachFunction.apply(q_compress)
        q_index, k_index, idx_weights = self.core_attention.indexer.get_qk_index(
            x_detached, q_compress_detached, rotary_pos_emb
        )
        k_index = self.dsa_indexer_key_handoff(k_index)
        if self.sparse_loss:
            topk_indices, _, softmax_max_index, softmax_sum_index = self.core_attention.indexer(
                q_index, k_index, idx_weights, actual_seq_len, actual_seq_len
            )
        else:
            softmax_max_index, softmax_sum_index = self.core_attention.indexer(
                q_index, k_index, idx_weights, actual_seq_len, actual_seq_len
            )
            topk_indices = None

        v_absorb = None
        if self.sparse_loss:
            query, key, value, v_absorb = self._dsa_sparse_qkv(
                q_a, compressed_kv, k_pe, seq_len, bs, rotary_pos_emb
            )
        else:
            query, key, value = self._dsa_dense_qkv(
                q_a, compressed_kv, k_pe, seq_len, bs, rotary_pos_emb
            )
        if self.use_tnd:
            query = self.sbh2tnd(query)
            key = self.sbh2tnd(key)
            value = self.sbh2tnd(value)
        else:
            query = mint.permute(query, (1, 0, 2, 3))
            key = mint.permute(key, (1, 0, 2, 3))
            value = mint.permute(value, (1, 0, 2, 3))

        query = self.cast(query, self.compute_dtype)
        key = self.cast(key, self.compute_dtype)
        value = self.cast(value, self.compute_dtype)
        value = self.dsa_value_handoff(value)
        attn_out, softmax_max, softmax_sum = self.core_attention(
            query, key, value, topk_indices=topk_indices, attention_mask=attention_mask,
            actual_seq_qlen=actual_seq_len, actual_seq_kvlen=actual_seq_len
        )
        if self.sparse_loss and v_absorb is not None:
            v_absorb_t = mint.permute(v_absorb, (0, 2, 1))
            # Weight absorb (v): memory-frugal batched matmul (see _AbsorbMatmul).
            attn_out = _AbsorbMatmul.apply(attn_out, v_absorb_t)
            attn_out = mint.reshape(attn_out, (bs, seq_len, self.num_attention_heads, self.v_head_dim))
            attn_out = mint.permute(attn_out, (1, 0, 2, 3))
        elif not self.sparse_loss:
            # FlashAttention returns BSND for the dense BSND path. Restore the
            # model's SBH layout explicitly; reshaping BSND directly to SBH
            # interleaves samples whenever batch size is greater than one.
            attn_out = mint.reshape(attn_out, (bs, seq_len, -1))
            attn_out = mint.permute(attn_out, (1, 0, 2))

        attn_out = mint.reshape(attn_out, (seq_len, bs, -1))
        output = self.linear_proj(attn_out)
        output = self.cast(output, ori_dtype)

        # Compute indexer loss. DSAIndexerLoss handles stop_gradient and
        # gradient routing internally via _DSAIndexerLossFunction.
        indexer_loss = self.core_attention.indexer_loss(
            query, key, q_index, k_index, idx_weights,
            topk_indices, softmax_max, softmax_sum,
            softmax_max_index, softmax_sum_index,
            actual_seq_len, actual_seq_len
        )
        attention_loss = attention_loss + indexer_loss

        # Save indexer loss to global tracker for logging in callback.
        save_to_indexer_losses_tracker(
            indexer_loss,
            self.layer_number + 1,
            self.config.num_layers + (self.config.mtp_num_layers or 0),
        )
        output = _IndexerLossAutoScaler.apply(output, indexer_loss)
        return output, attention_loss

    def _dsa_dense_qkv(self, q_a, compressed_kv, k_pe, seq_len, bs, rotary_pos_emb):
        """Generate QKV for the DSA dense warm-up stage."""
        q = self.linear_qb(q_a)
        q = self.reshape(q, (seq_len, bs, self.num_attention_heads, -1))
        q_nope, q_pe = self.split(q, [self.qk_head_dim, self.qk_pos_emb_head_dim], dim=-1)

        k_pe = self.reshape(k_pe, (seq_len, bs, 1, self.qk_pos_emb_head_dim))
        compressed_kv_norm = self.k_layernorm(compressed_kv)
        kv = self.linear_kvb(compressed_kv_norm)
        kv = self.reshape(kv, (seq_len, bs, self.num_attention_heads, self.qk_head_dim + self.v_head_dim))
        k_nope, value = self.split(kv, [self.qk_head_dim, self.v_head_dim], dim=-1)

        if rotary_pos_emb is not None:
            q_pe = self.apply_rotary_emb_q(
                q_pe, rotary_pos_emb,
                rotary_interleaved=self.config.rotary_interleaved,
                multi_latent_attention=self.config.multi_latent_attention
            )
            k_pe = self.apply_rotary_emb_k(
                k_pe, rotary_pos_emb,
                rotary_interleaved=self.config.rotary_interleaved,
                multi_latent_attention=self.config.multi_latent_attention
            )

        query = self.cat([q_nope, q_pe], 3)
        k_pe = self.tile_kv(k_pe, (1, 1, self.num_attention_heads, 1))
        key = self.cat([k_nope, k_pe], 3)

        return query, key, value

    def _dsa_sparse_qkv(self, q_a, compressed_kv, k_pe, seq_len, bs, rotary_pos_emb):
        """Generate QKV for the DSA sparse stage with MQA weight absorb."""
        q = self.linear_qb(q_a)
        q = self.reshape(q, (seq_len, bs, self.num_attention_heads, -1))
        q_nope, q_pe = self.split(q, [self.qk_head_dim, self.qk_pos_emb_head_dim], dim=-1)
        k_pe = self.reshape(k_pe, (seq_len, bs, 1, self.qk_pos_emb_head_dim))
        compressed_kv_norm = self.k_layernorm(compressed_kv)
        k_nope = self.reshape(compressed_kv_norm, (seq_len, bs, 1, self.kv_lora_rank))
        value = self.reshape(compressed_kv_norm, (seq_len, bs, 1, self.kv_lora_rank))

        if rotary_pos_emb is not None:
            q_pe = self.apply_rotary_emb_q(
                q_pe, rotary_pos_emb,
                rotary_interleaved=self.config.rotary_interleaved,
                multi_latent_attention=self.config.multi_latent_attention
            )
            k_pe = self.apply_rotary_emb_k(
                k_pe, rotary_pos_emb,
                rotary_interleaved=self.config.rotary_interleaved,
                multi_latent_attention=self.config.multi_latent_attention
            )

        w_kvb = self.linear_kvb.weight
        if w_kvb.has_init:
            w_kvb.init_data()
        w_kvb = self.cast(w_kvb, self.compute_dtype)
        w_kvb = mint.reshape(w_kvb, (self.num_attention_heads, self.qk_head_dim + self.v_head_dim, self.kv_lora_rank))
        q_absorb, v_absorb = mint.split(w_kvb, [self.qk_head_dim, self.v_head_dim], dim=1)

        q_nope = self.cast(q_nope, self.compute_dtype)
        # Weight absorb (q): memory-frugal batched matmul instead of unsqueeze+broadcast-matmul,
        # whose backward materialises a [seq*bs, heads, qk_head_dim, kv_lora_rank] outer product.
        q_nope = _AbsorbMatmul.apply(q_nope, q_absorb)
        q_nope = mint.reshape(q_nope, (seq_len, bs, self.num_attention_heads, self.kv_lora_rank))

        query = self.cat([q_nope, q_pe], 3)
        key = self.cat([k_nope, k_pe], 3)
        return query, key, value, v_absorb



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

    def construct(self, x: Tensor, attention_mask=None, rotary_pos_emb=None,
                  prefix_keys_values=None, pad_zeros=None, actual_seq_len=None,
                  attention_loss=0., mscale=1.0, rotary_cos_sin=None):
        """
        Forward pass of MLA self-attention with optional DSA support.

        Args:
            x: Input tensor with shape (seq_length, batch_size, hidden_size).
            attention_mask: Attention mask tensor (optional).
            rotary_pos_emb: Rotary position embedding tensor (optional).
            prefix_keys_values: Prefix key-value pairs (optional).
            pad_zeros: Padding zeros tensor (not used).
            actual_seq_len: Actual sequence length for EOD mask compression (optional).
            attention_loss: Accumulated attention loss from previous layers.

        Returns:
            Tuple[Tensor, float]: (output, attention_loss).
        """
        if self.use_dsa:
            return self.dsa_forward(x, attention_mask, rotary_pos_emb, actual_seq_len, attention_loss)

        return super().construct(x, attention_mask, rotary_pos_emb,
                                 prefix_keys_values, pad_zeros, actual_seq_len, mscale,
                                 rotary_cos_sin)
