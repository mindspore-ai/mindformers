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
from mindspore import ops
from mindspore.common._grad_function import _Function

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.transformers.experimental_attention_variant.indexer import (
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
        ctx.indexer_loss = ops.stop_gradient(indexer_loss)
        ctx.propagate_to_previous_layer = propagate_to_previous_layer
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Return no trunk grad, a layer trigger, and the scaled Indexer-loss grad."""
        indexer_loss = ctx.indexer_loss
        if isinstance(indexer_loss, DTensor):
            indexer_loss = indexer_loss.to_local()
        indexer_grad = ops.ones_like(indexer_loss) * _IndexerLossAutoScaler.main_loss_backward_scale
        layer_input_grad = ops.zeros_like(grad_output) if ctx.propagate_to_previous_layer else None
        return None, layer_input_grad, indexer_grad, None


class _DSAWarmupLayerMixin:
    """Shared DSA warm-up boundary behavior for Transformer layer variants."""

    def _init_dsa_warmup(self, layer_number):
        """Initialize DSA stage state after the parent layer is constructed."""
        self.layer_number = layer_number
        self.dsa_warmup = not self.config.dsa_indexer_use_sparse_loss
        self.disable_activation_recompute = self.dsa_warmup
        self.propagate_to_previous_layer = layer_number > 0

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
        """Install the DSA1 graph boundary without changing forward numerics."""
        if self.training and self.dsa_warmup:
            detached_output = ops.stop_gradient(output)
            output = _DSAWarmupLayerBoundary.apply(
                detached_output,
                layer_input,
                indexer_loss,
                self.propagate_to_previous_layer,
            )
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
