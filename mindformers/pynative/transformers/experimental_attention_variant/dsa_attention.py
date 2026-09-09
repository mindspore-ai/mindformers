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
from mindformers.pynative.transformers.experimental_attention_variant.dsa import (
    dsa_layer_index_share_group_size,
    dsa_layer_is_index_leader,
)
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


class IndexShareGroup:
    """Per-invocation state for one DSA indexer-sharing group.

    Created fresh by ``TransformerBlock.construct`` at every group leader, so concurrently
    running micro-batches (dualpipe / ``overlap_b_f`` interleave forward and backward across
    threads) never share one. It is also passed to each layer as a plain argument, so an
    activation-recompute replay of a Shared layer still finds the leader's Top-K alive.
    """

    __slots__ = ("topk_indices", "leader_indexer_loss", "leader_projections")

    def __init__(self):
        self.topk_indices = None
        # Set by the leader: its ``indexer_loss`` module plus the indexer projections and
        # index softmax stats the student side needs. Each served layer calls the leader's
        # module with its own q/k/Top-K/softmax stats, so its term is evaluated inside its
        # own activation-checkpoint region and scaled by the group size; the terms then sum
        # to the group mean without any accumulator crossing a region boundary.
        self.leader_indexer_loss = None
        self.leader_projections = None


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
        self.index_share_group_size = dsa_layer_index_share_group_size(config, layer_number)
        self.dsa_value_handoff = IdentityOp()
        self.dsa_loss_key_indexer_handoff = IdentityOp()

    def core_attention_extra_kwargs(self):
        """Decide Full/Shared from the *true* global layer index.

        ``MLASelfAttention`` builds ``core_attention`` with ``self.layer_index``, which is
        ``max(1, layer_number)``. Letting ``DSAttention`` re-derive the sharing layout from
        that clamped value reads the layout one slot too far for global layer 0, so with an
        ``FSSS`` layout every layer builds as Shared, nobody owns an indexer, and the first
        Shared layer attends with a ``None`` Top-K.
        """
        return {
            "is_index_leader": dsa_layer_is_index_leader(self.config, self.layer_number),
        }

    def construct(self, x: Tensor, attention_mask=None, rotary_pos_emb=None,
                  prefix_keys_values=None, pad_zeros=None, actual_seq_len=None,
                  attention_loss=0., mscale=1.0, rotary_cos_sin=None,
                  index_share_group=None):
        """Forward pass with DSA index selection, attention and indexer loss.

        ``index_share_group`` is this layer's :class:`IndexShareGroup`, or None when sharing
        is off. A Full layer fills in its ``topk_indices``; a Shared layer reads them and runs
        no indexer at all -- it owns no indexer parameters, computes no indexer projections
        and no indexer loss.
        """
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
        is_leader = self.core_attention.is_index_leader
        if is_leader:
            (
                q_index, k_index, idx_weights,
                topk_indices, _, softmax_max_index, softmax_sum_index,
            ) = self.core_attention.indexer(
                x_detached, q_compress_detached, rotary_pos_emb,
                actual_seq_len, actual_seq_len,
            )
            # Dense warm-up produces no Top-K (``dsa_indexer.py`` returns None when not
            # ``sparse_loss``); there the group shares the leader's *indexer*, not its
            # selection, so leave ``topk_indices`` unset rather than publishing a None.
            if index_share_group is not None and topk_indices is not None:
                index_share_group.topk_indices = topk_indices.detach()
        else:
            # Shared layer: no indexer module exists on this layer at all.
            q_index = k_index = idx_weights = None
            softmax_max_index = softmax_sum_index = None
            topk_indices = index_share_group.topk_indices if index_share_group else None
            # Dense warm-up runs ``dense_flash_attention``, which takes no Top-K, and its
            # leader produces none (``dsa_indexer.py``: ``topk_indices = None`` when not
            # ``sparse_loss``). A missing Top-K is only a layout error in the sparse stage.
            if topk_indices is None and self.sparse_loss:
                # Without this the None flows into ops.sparse_flash_attention and surfaces as
                # "missing 1 required positional argument: sparse_indices", which says nothing
                # about the layout. Name the layer and the disagreement instead.
                raise RuntimeError(
                    f"DSA layer {self.layer_number} is a Shared layer but no group leader "
                    "supplied Top-K indices. Its own view of the sharing layout disagrees "
                    "with the one TransformerBlock used to form the groups; both must come "
                    "from resolve_dsa_index_share_leaders() with the same global layer index."
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
        indexer_loss, reported_loss = self._indexer_loss(
            is_leader, index_share_group,
            query, key, q_index, k_index, idx_weights,
            topk_indices, softmax_max, softmax_sum,
            softmax_max_index, softmax_sum_index, actual_seq_len,
        )
        if indexer_loss is not None:
            attention_loss = attention_loss + indexer_loss
            # Report the unscaled term: the tracker averages over the layers that record
            # one, so what it prints is the group's mean, on the same scale as an
            # unshared run's per-layer indexer loss.
            save_to_indexer_losses_tracker(
                reported_loss,
                self.layer_number + 1,
                self.config.num_layers + (self.config.mtp_num_layers or 0),
            )
            output = _IndexerLossAutoScaler.apply(output, indexer_loss)
        return output, attention_loss

    def _indexer_loss(self, is_leader, group,
                      query, key, q_index, k_index, idx_weights,
                      topk_indices, softmax_max, softmax_sum,
                      softmax_max_index, softmax_sum_index, actual_seq_len):
        """Return this layer's indexer loss, or None when this layer contributes none.

        Returns ``(contribution, reported)``: what enters the graph, and what the tracker
        records. Under sharing each layer contributes its term scaled by the group size but
        reports the term itself, so the tracker stays on the unshared scale.

        Sharing leaves Shared layers without any indexer: the Full layer's indexer is
        distilled against every layer it serves and the losses are averaged, which is
        gradient-equivalent to distilling against the averaged attention distribution
        (IndexCache's L_multi, used by GLM-5.2). Because the fused kernel rebuilds the
        teacher from one layer's own q/k/softmax stats and has no slot for a pre-averaged
        teacher, this costs one kernel call per served layer: the indexer-loss cost is NOT
        reduced, what sharing saves is the Shared layers' indexer forward.
        """
        if group is None:
            if not is_leader:
                return None, None
            loss = self.core_attention.indexer_loss(
                query, key, q_index, k_index, idx_weights,
                topk_indices, softmax_max, softmax_sum,
                softmax_max_index, softmax_sum_index,
                actual_seq_len, actual_seq_len
            )
            return loss, loss

        if is_leader:
            # Publish this layer's indexer teacher so every layer its Top-K serves can call
            # it with its own stats, inside that layer's own checkpoint region.
            group.leader_indexer_loss = self.core_attention.indexer_loss
            group.leader_projections = (
                q_index, k_index, idx_weights, softmax_max_index, softmax_sum_index,
            )

        if group.leader_indexer_loss is None:
            return None, None
        l_q_index, l_k_index, l_weights, l_max_index, l_sum_index = group.leader_projections
        term = group.leader_indexer_loss(
            query, key, l_q_index, l_k_index, l_weights,
            topk_indices, softmax_max, softmax_sum,
            l_max_index, l_sum_index,
            actual_seq_len, actual_seq_len,
        )
        # Scale here rather than summing the terms and dividing at the group end. A running
        # sum would be produced in one layer's checkpoint region and consumed in the next,
        # and the pipeline scheduler's backward then unpacks the same saved tensor twice:
        # "Unpack is being triggered for a tensor, make sure to do this only once!". Scaling
        # per layer keeps every term inside its own region and still totals the group mean,
        # because the terms are only ever added together through the loss.
        return term / self.index_share_group_size, term

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
