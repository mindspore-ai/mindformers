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
"""DSA-specific transformer layers with a warm-up-only backward boundary."""
# MindSpore ``_Function`` subclasses intentionally use model-specific signatures.
# pylint: disable=arguments-differ,abstract-method
from hyper_parallel.core.dtensor.dtensor import DTensor
from mindspore import Tensor, ops
from mindspore.common._grad_function import _Function

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.transformers.experimental_attention_variant.indexer import (
    DSA_WARMUP_LOSS_SINK,
    _IndexerLossAutoScaler,
)
from mindformers.pynative.transformers.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
    TransformerLayerSubmodules,
)


class _DSAWarmupLayerBoundary(_Function):
    """Cut frozen trunk backward while keeping every local Indexer loss trainable.

    The numeric layer output is preserved. During backward, the incoming language-
    model gradient is deliberately not propagated into the frozen layer trunk.
    Instead, a zero-valued trigger is routed directly to the previous layer input,
    and the local Indexer loss receives the same loss scale as the main loss.
    """

    @staticmethod
    def forward(ctx, output, layer_input, indexer_loss, propagate_to_previous_layer):
        """Return the detached trunk output and retain only lightweight metadata."""
        del layer_input
        # A layer that contributes no Indexer term leaves ``attention_loss`` at the Python
        # ``0.`` the block passes in. That happens under ``dsa_index_share_loss: leader``,
        # where a Shared layer owns no indexer and publishes no loss -- unlike ``served``,
        # where every layer in the group distils against the leader's teacher and so always
        # has a term. Record it, because a float has no gradient to seed.
        ctx.has_indexer_loss = isinstance(indexer_loss, Tensor)
        ctx.indexer_loss = ops.stop_gradient(indexer_loss) if ctx.has_indexer_loss else None
        ctx.propagate_to_previous_layer = propagate_to_previous_layer
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Return no trunk grad, a layer trigger, and the scaled Indexer-loss grad."""
        indexer_grad = None
        if ctx.has_indexer_loss:
            indexer_loss = ctx.indexer_loss
            if isinstance(indexer_loss, DTensor):
                indexer_loss = indexer_loss.to_local()
            indexer_grad = (
                ops.ones_like(indexer_loss) * _IndexerLossAutoScaler.main_loss_backward_scale
            )
        layer_input_grad = ops.zeros_like(grad_output) if ctx.propagate_to_previous_layer else None
        return None, layer_input_grad, indexer_grad, None


class _DSAWarmupTrunkCut(_Function):
    """Cut the frozen trunk's gradient while keeping the layer on the autograd graph.

    ``ops.stop_gradient`` would do the cutting, but it also takes the layer *off* the graph:
    with every layer cut that way the language-model loss ends up with no ``grad_fn`` at all
    and the trainer's ``loss.backward(sense)`` fails with "The output tensor you provided
    doesn't requires grad and not have a grad_fn". A ``_Function`` returning ``None`` cuts
    the gradient just as effectively while leaving the graph connected, which is also why
    ``_DSAWarmupLayerBoundary`` is one.

    Used only with per-layer warm-up backward, where the indexer gradient is seeded by the
    block loop instead of by the boundary.
    """

    @staticmethod
    def forward(ctx, output):
        """Return the trunk output unchanged."""
        del ctx
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Drop the incoming gradient: the trunk is frozen, so nothing propagates through.

        Falling off the end is the ``_Function`` way of saying "no gradient for this input";
        spelling it as ``return None`` is what pylint's useless-return flags.
        """
        del ctx, grad_output


class _DSAWarmupLayerMixin:
    """Shared DSA warm-up boundary behavior for Transformer layer variants."""

    def _init_dsa_warmup(self, layer_number):
        """Initialize DSA stage state after the parent layer is constructed."""
        self.layer_number = layer_number
        self.dsa_warmup = not self.config.dsa_indexer_use_sparse_loss
        self.layerwise_warmup_backward = (
            self.dsa_warmup and getattr(self.config, "dsa_warmup_layerwise_backward", False)
        )
        # The frozen trunk is never worth replaying, so the full-layer wrapper is always
        # skipped in warm-up (!8692). Whether the *indexer* inside still gets wrapped is
        # decided in ``activation_checkpoint`` from ``layerwise_warmup_backward``: with
        # per-layer backward its activations are already short-lived, without it the wrapper
        # is what stops them piling up.
        self.disable_activation_recompute = self.dsa_warmup
        # The zero-gradient wake-up chain only exists to reach the previous layer's boundary
        # during the step-end backward; per-layer backward has already driven every layer by
        # then, so the chain is dead weight.
        self.propagate_to_previous_layer = layer_number > 0 and not self.layerwise_warmup_backward

    @staticmethod
    def _split_attention_result(attention_result):
        """Split the DSA attention output and its local Indexer loss."""
        if not isinstance(attention_result, tuple) or len(attention_result) != 2:
            raise RuntimeError(
                "DSATransformerLayer expects DSASelfAttention to return "
                "(attention_output, indexer_loss)."
            )
        return attention_result

    def _finish_layer(self, output, layer_input, indexer_loss, context):
        """Install the DSA1 graph boundary without changing forward numerics.

        With ``dsa_warmup_layerwise_backward`` the boundary is not needed at all: it exists to
        seed the indexer gradient and wake the previous layer during the step-end backward,
        and the block loop does both explicitly instead. All that remains of it is cutting the
        frozen trunk, which is what ``_DSAWarmupTrunkCut`` is for. The layer's indexer loss
        goes into the block loop's per-call list, which back-propagates it and lets this
        layer's activations go before the next layer runs. The list is picked up from a
        thread-local rather than received as an argument, because the layer call marshals its
        keyword arguments and the layer would append into a copy -- see ``_DSAWarmupLossSink``.
        """
        warmup_loss_sink = (
            DSA_WARMUP_LOSS_SINK.sink if self.layerwise_warmup_backward else None
        )
        if self.training and self.dsa_warmup:
            if self.layerwise_warmup_backward and warmup_loss_sink is None:
                # Falling back to the boundary path here would be silent data loss: nothing
                # back-propagates it in this mode, so the indexer would stay trainable and
                # never receive a gradient. Whoever invokes this layer has to open a
                # ``warmup_indexer_loss_sink`` -- ``TransformerBlock`` and the MTP layer do.
                raise RuntimeError(
                    "DSA warm-up per-layer backward is on but no indexer-loss sink is "
                    "published on this thread. The caller of this layer must wrap the call "
                    "in `warmup_indexer_loss_sink(...)`; otherwise this layer's indexer "
                    "would train on no gradient at all."
                )
            if warmup_loss_sink is not None:
                # ``detach`` first, exactly as the boundary path below does: a
                # ``_Function`` alone keeps the layer's whole intra-layer graph reachable
                # from ``output``, so every layer's attention/MLP activations survive until
                # the step-end backward -- measured at 16k, peak went 24 GiB -> 44 GiB.
                # Detaching frees them as the forward advances; ``_DSAWarmupTrunkCut`` on top
                # puts a ``grad_fn`` back so the language-model loss still has a landing
                # point (see the class docstring).
                output = _DSAWarmupTrunkCut.apply(output.detach())
                # Same predicate as ``_DSAWarmupLayerBoundary``: a layer that contributes no
                # Indexer term leaves this at the Python ``0.`` the attention starts from --
                # under ``dsa_index_share_loss: leader`` every Shared layer does, because it
                # owns no indexer. A float carries no graph, so buffering it would only put a
                # term into the group sum that ``backward`` cannot start from.
                if isinstance(indexer_loss, Tensor):
                    warmup_loss_sink.append(indexer_loss)
            else:
                detached_output = ops.stop_gradient(output)
                output = _DSAWarmupLayerBoundary.apply(
                    detached_output,
                    layer_input,
                    indexer_loss,
                    self.propagate_to_previous_layer,
                )
        # The sharing state travels in ``index_share_group`` rather than in the return
        # value, so the layer keeps the stock ``(output, context)`` contract for every
        # caller (MTP block, non-DSA models) whether or not sharing is on.
        return output, context


class DSATransformerLayer(_DSAWarmupLayerMixin, TransformerLayer):
    """Transformer layer that adds the DSA warm-up boundary to the standard path.

    DSA2 follows the regular full-backward path. In dense DSA1 warm-up, only the
    Indexer parameters are trainable. The complete TransformerLayer forward is
    still evaluated, while ``_DSAWarmupLayerBoundary`` prevents backward and
    recompute replay of the frozen attention/MLP trunk.
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: TransformerLayerSubmodules,
            layer_number: int = 0,
            hidden_dropout: float = None,
            is_mtp_layer: bool = False,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            is_mtp_layer=is_mtp_layer,
        )
        self._init_dsa_warmup(layer_number)

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            context=None,
            rotary_pos_emb=None,
            prefix_keys_values=None,
            actual_seq_len=None,
            input_ids=None,
            mscale=1.0,
            rotary_cos_sin=None,
            index_share_group=None,
    ):
        """Run a DSA layer and apply the warm-up boundary when required."""
        layer_input = hidden_states
        input_layernorm_output = self.input_layernorm(hidden_states)
        residual = (
            input_layernorm_output
            if self.apply_residual_connection_post_norm
            else hidden_states
        )

        attention_result = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            prefix_keys_values=prefix_keys_values,
            actual_seq_len=actual_seq_len,
            mscale=mscale,
            rotary_cos_sin=rotary_cos_sin,
            index_share_group=index_share_group,
        )
        attention_output, indexer_loss = self._split_attention_result(attention_result)
        dropout_output = self.hidden_states_dropout(attention_output)
        norm_input = self.add(residual, dropout_output)
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(norm_input)
        residual = (
            pre_mlp_layernorm_output
            if self.apply_residual_connection_post_norm
            else norm_input
        )

        mlp_output = self.mlp(pre_mlp_layernorm_output, input_ids=input_ids)
        dropout_output = self.hidden_states_dropout(mlp_output)
        output = self.add(residual, dropout_output)
        return self._finish_layer(output, layer_input, indexer_loss, context)


class DSAHyperConnectionTransformerLayer(_DSAWarmupLayerMixin, HyperConnectionTransformerLayer):
    """mHC Transformer layer that adds the DSA warm-up boundary."""

    def __init__(
            self,
            config: TransformerConfig,
            submodules: TransformerLayerSubmodules,
            layer_number: int = 0,
            hidden_dropout: float = None,
            is_mtp_layer: bool = False,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            is_mtp_layer=is_mtp_layer,
        )
        self._init_dsa_warmup(layer_number)

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            context=None,
            rotary_pos_emb=None,
            prefix_keys_values=None,
            actual_seq_len=None,
            input_ids=None,
            mscale=1.0,
            rotary_cos_sin=None,
            index_share_group=None,
    ):
        """Run the DSA mHC layer and apply the DSA1 warm-up boundary."""
        layer_input = hidden_states
        streams_before_attention = hidden_states
        aggregated_attention, residual_attention, post_attention = self.attn_hc(hidden_states)

        input_layernorm_output = self.input_layernorm(aggregated_attention)
        attention_result = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            prefix_keys_values=prefix_keys_values,
            actual_seq_len=actual_seq_len,
            mscale=mscale,
            rotary_cos_sin=rotary_cos_sin,
            index_share_group=index_share_group,
        )
        attention_output, indexer_loss = self._split_attention_result(attention_result)
        dropout_output = self.hidden_states_dropout(attention_output)
        hidden_states = self.attn_hc.output_cell(
            residual_attention,
            post_attention,
            streams_before_attention,
            dropout_output,
        )

        streams_before_ffn = hidden_states
        aggregated_ffn, residual_ffn, post_ffn = self.ffn_hc(hidden_states)
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(aggregated_ffn)
        mlp_output = self.mlp(pre_mlp_layernorm_output, input_ids=input_ids)
        dropout_output = self.hidden_states_dropout(mlp_output)
        output = self.ffn_hc.output_cell(
            residual_ffn,
            post_ffn,
            streams_before_ffn,
            dropout_output,
        )
        return self._finish_layer(output, layer_input, indexer_loss, context)
