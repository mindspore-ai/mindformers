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
"""Top-level DeepSeek Sparse Attention module for pynative mode."""
# MindSpore ``Cell``/``_Function`` subclasses intentionally use operator-specific signatures.
# pylint: disable=arguments-differ,abstract-method
from mindspore import Tensor, mint
from mindspore.common._grad_function import _Function

from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.pynative.layers.identity_op import IdentityOp
from mindformers.pynative.transformers.experimental_attention_variant.indexer import _IndexerLossAutoScaler
from mindformers.pynative.transformers.experimental_attention_variant.utils import save_to_indexer_losses_tracker
from mindformers.pynative.transformers.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)


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
    product of shape ``[tokens, heads, d, r]`` in backward. Here
    forward/backward are explicit batched-matmuls over the head axis, and the
    ``d_w`` reduction contracts the token axis inside the bmm.
    """

    @staticmethod
    def forward(ctx, x, w):
        """Apply the head-wise absorbed matmul without token-wise weight expansion."""
        lead = tuple(x.shape[:-2])
        h, d = x.shape[-2], x.shape[-1]
        r = w.shape[-1]
        t = 1
        for s in lead:
            t *= s
        ctx.save_for_backward(x, w)
        ctx.lead = lead
        ctx.hdr = (h, d, r)
        xh = mint.permute(mint.reshape(x, (t, h, d)), (1, 0, 2))
        out = mint.bmm(xh, w)
        out = mint.permute(out, (1, 0, 2))
        return mint.reshape(out, lead + (h, r))

    @staticmethod
    def backward(ctx, grad_output):
        """Compute activation and weight gradients with head-wise batched matmuls."""
        h, d, r = ctx.hdr
        t = 1
        for s in ctx.lead:
            t *= s
        x, w = ctx.saved_tensors
        gh = mint.permute(mint.reshape(grad_output, (t, h, r)), (1, 0, 2))
        xh = mint.permute(mint.reshape(x, (t, h, d)), (1, 0, 2))
        d_x = mint.bmm(gh, mint.permute(w, (0, 2, 1)))
        d_x = mint.reshape(mint.permute(d_x, (1, 0, 2)), ctx.lead + (h, d))
        d_w = mint.bmm(mint.permute(xh, (0, 2, 1)), gh)
        return d_x, d_w


class DSASelfAttention(MLASelfAttention):
    """MLA-based DeepSeek Sparse Attention top-level implementation."""

    attention_variant = "dsa"

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
        )
        self.sparse_loss = config.dsa_indexer_use_sparse_loss
        self.dsa_value_handoff = IdentityOp()
        self.dsa_loss_key_indexer_handoff = IdentityOp()

    def construct(self, x: Tensor, attention_mask=None, rotary_pos_emb=None,
                  prefix_keys_values=None, pad_zeros=None, actual_seq_len=None,
                  attention_loss=0., mscale=1.0, rotary_cos_sin=None):
        """Forward pass with DSA index selection, attention and indexer loss."""
        del prefix_keys_values, pad_zeros, mscale, rotary_cos_sin
        ori_dtype = x.dtype
        seq_len, bs, _ = self.shape(x)
        qkv_combo = self.linear_qkv(x)
        q_a, compressed_kv, k_pe = self.split(
            qkv_combo,
            [self.q_rank, self.kv_lora_rank, self.qk_pos_emb_head_dim],
            dim=-1,
        )

        if self.q_layernorm is not None:
            q_a = self.q_layernorm(q_a)
        q_compress = q_a
        x_detached = _DSADetachFunction.apply(x)
        q_compress_detached = _DSADetachFunction.apply(q_compress)
        (
            q_index, k_index, idx_weights,
            topk_indices, _, softmax_max_index, softmax_sum_index,
        ) = self.core_attention.indexer(
            x_detached, q_compress_detached, rotary_pos_emb,
            actual_seq_len, actual_seq_len,
        )

        v_absorb = None
        if self.sparse_loss:
            query, key, value, v_absorb = self._dsa_sparse_qkv(
                q_a, compressed_kv, k_pe, seq_len, bs, rotary_pos_emb
            )
        else:
            query, key, value = self._dsa_dense_qkv(
                q_a, compressed_kv, k_pe, seq_len, bs, rotary_pos_emb
            )
        attention_seq_len = self.shape(query)[0]
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
            local_heads = v_absorb.shape[0]
            v_absorb_t = mint.permute(v_absorb, (0, 2, 1))
            attn_out = _AbsorbMatmul.apply(attn_out, v_absorb_t)
            attn_out = mint.reshape(
                attn_out,
                (bs, attention_seq_len, local_heads, self.v_head_dim),
            )
            attn_out = mint.permute(attn_out, (1, 0, 2, 3))
        elif not self.sparse_loss:
            attn_out = mint.reshape(attn_out, (bs, attention_seq_len, -1))
            attn_out = mint.permute(attn_out, (1, 0, 2))

        attn_out = mint.reshape(attn_out, (attention_seq_len, bs, -1))
        output = self.linear_proj(attn_out)
        output = self.cast(output, ori_dtype)
        indexer_loss = self.core_attention.indexer_loss(
            query, key, q_index, k_index, idx_weights,
            topk_indices, softmax_max, softmax_sum,
            softmax_max_index, softmax_sum_index,
            actual_seq_len, actual_seq_len
        )
        attention_loss = attention_loss + indexer_loss
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
        query_head_dim = self.qk_head_dim + self.qk_pos_emb_head_dim
        local_heads = self.shape(q)[-1] // query_head_dim
        q = self.reshape(q, (self.shape(q)[0], bs, local_heads, query_head_dim))
        q_nope, q_pe = self.split(q, [self.qk_head_dim, self.qk_pos_emb_head_dim], dim=-1)

        k_pe = self.reshape(k_pe, (seq_len, bs, 1, self.qk_pos_emb_head_dim))
        compressed_kv_norm = self.k_layernorm(compressed_kv)
        kv = self.linear_kvb(compressed_kv_norm)
        kv_head_dim = self.qk_head_dim + self.v_head_dim
        if self.shape(kv)[-1] != local_heads * kv_head_dim:
            raise ValueError(
                f"linear_kvb local output must contain {local_heads} heads, "
                f"but got width {self.shape(kv)[-1]} with head width {kv_head_dim}."
            )
        kv = self.reshape(
            kv,
            (self.shape(kv)[0], bs, local_heads, kv_head_dim),
        )
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
        k_pe = self.tile_kv(k_pe, (1, 1, local_heads, 1))
        key = self.cat([k_nope, k_pe], 3)
        return query, key, value

    def _dsa_sparse_qkv(self, q_a, compressed_kv, k_pe, seq_len, bs, rotary_pos_emb):
        """Generate QKV for the DSA sparse stage with MQA weight absorb."""
        q = self.linear_qb(q_a)
        query_head_dim = self.qk_head_dim + self.qk_pos_emb_head_dim
        local_heads = self.shape(q)[-1] // query_head_dim
        q = self.reshape(q, (self.shape(q)[0], bs, local_heads, query_head_dim))
        q_nope, q_pe = self.split(q, [self.qk_head_dim, self.qk_pos_emb_head_dim], dim=-1)
        k_pe = self.reshape(k_pe, (seq_len, bs, 1, self.qk_pos_emb_head_dim))
        compressed_kv_norm = self.k_layernorm(compressed_kv)
        sparse_kv_seq_len = self.shape(compressed_kv_norm)[0]
        k_nope = self.reshape(compressed_kv_norm, (sparse_kv_seq_len, bs, 1, self.kv_lora_rank))
        value = self.reshape(compressed_kv_norm, (sparse_kv_seq_len, bs, 1, self.kv_lora_rank))

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
        if getattr(w_kvb, "has_init", False):
            w_kvb.init_data()
        w_kvb = self._to_local_tensor(w_kvb)
        w_kvb = self.cast(w_kvb, self.compute_dtype)
        expected_rows = local_heads * (self.qk_head_dim + self.v_head_dim)
        if int(w_kvb.shape[0]) != expected_rows:
            raise ValueError(
                f"linear_kvb local rows must equal local_heads * (qk_head_dim + v_head_dim), "
                f"but got {w_kvb.shape[0]} and expected {expected_rows}."
            )
        w_kvb = mint.reshape(w_kvb, (local_heads,
                                     self.qk_head_dim + self.v_head_dim, self.kv_lora_rank))
        q_absorb, v_absorb = mint.split(w_kvb, [self.qk_head_dim, self.v_head_dim], dim=1)

        q_nope = self.cast(q_nope, self.compute_dtype)
        q_nope = _AbsorbMatmul.apply(q_nope, q_absorb)
        q_nope = mint.reshape(
            q_nope,
            (self.shape(q_nope)[0], bs, local_heads, self.kv_lora_rank),
        )

        query = self.cat([q_nope, q_pe], 3)
        key = self.cat([k_nope, k_pe], 3)
        return query, key, value, v_absorb
