# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Modified to adapt to MindSpore pynative mode.
# Main changes: PyTorch->MindSpore, removed parallel/BDA/CUDA graph features, simplified forward pass.
"""Transformer Layer"""
import inspect
from dataclasses import dataclass
from typing import Union
from mindspore import nn, mint
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module, get_module
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.layers.dropout import Dropout
from mindformers.pynative.layers.identity_op import IdentityOp
from mindformers.pynative.transformers.hyper_connection import HyperConnectionModule, FusedHyperConnectionModule


def _accepts_kwarg(spec_or_module, name: str) -> bool:
    """Return whether the module a spec resolves to declares ``name`` in its ``__init__``.

    ``build_module`` forwards ``**kwargs`` straight into the target ``__init__`` with no
    filtering, so passing a kwarg to a class that neither names it nor accepts ``**kwargs``
    raises ``TypeError``. Only DSv4 hybrid attention / MoE declare ``is_mtp_layer``; the many
    other reachable self-attention and dense-MLP classes do not, so callers must gate the pass.
    """
    module = get_module(spec_or_module)
    if not isinstance(module, type):
        return False
    try:
        params = inspect.signature(module.__init__).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _maybe_kwarg(spec_or_module, name: str, value) -> dict:
    """Build a ``{name: value}`` dict only when the target accepts ``name``, else ``{}``.

    Spread the result into a ``build_module`` call so non-accepting modules are never handed
    the extra kwarg.
    """
    if _accepts_kwarg(spec_or_module, name):
        return {name: value}
    return {}


@dataclass
class TransformerLayerSubmodules:
    """
    Configuration class for specifying the submodules of a transformer layer.

    This class defines the structure and default implementations for various
    components of a transformer layer, allowing for flexible customization
    of the layer's architecture.

    Args:
        input_layernorm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        self_attention (Union[ModuleSpec, type]): Specification for the self-attention mechanism.
        pre_cross_attn_layernorm (Union[ModuleSpec, type]): Specification for the layer
            normalization before cross-attention.
        cross_attention (Union[ModuleSpec, type]): Specification for the cross-attention mechanism.
        pre_mlp_layernorm (Union[ModuleSpec, type]): Specification for the layer normalization
            before the MLP.
        mlp (Union[ModuleSpec, type]): Specification for the MLP in Dense layer.
    """

    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp


class BaseTransformerLayer:
    """A common parent class for `TransformerLayer` like implementations.

    A dummy class that is subclassed by similar `TransformerLayer`s e.g. the
    `TransformerLayer` in this file and possibly other `TransformerLayer`
    implementations that aim to use `TransformerBlock` as the base module.
    The main purpose is to check if any layer (or module) provided in the spec
    is a subclass of this class to allow fanning-out of that spec for all the
    layers in the `TransformerBlock`. See `_get_block_submodules` method
    implementation in `transformer_block.py` file for more details.
    """

    def __init__(self):
        pass


class TransformerLayer(nn.Cell, BaseTransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size. This layer implements the standard transformer
    architecture including self-attention, cross-attention (optional), and
    feed-forward MLP components.

    Args:
        config (TransformerConfig): Configuration for the transformer model.
        submodules (TransformerLayerSubmodules): The submodules used to construct
            the transformer layer, including input_layernorm, self_attention,
            pre_cross_attn_layernorm, cross_attention, pre_mlp_layernorm, and mlp.
        layer_number (int, optional): The layer number in the transformer stack.
            Default: 1.
        hidden_dropout (float, optional): Dropout probability for hidden states.
            If None, will use config.hidden_dropout. Default: None.
    """

    def __init__(self,
                 config: TransformerConfig,
                 submodules: TransformerLayerSubmodules,
                 layer_number: int = 0,
                 hidden_dropout: float = None,
                 is_mtp_layer: bool = False,
                 ):
        super().__init__()
        self.config = config
        self.is_mtp_layer = is_mtp_layer
        self.apply_residual_connection_post_norm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        self.input_layernorm = build_module(
            submodules.input_layernorm,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon,
            compute_dtype=config.layernorm_compute_dtype
        )

        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=layer_number,
            **_maybe_kwarg(submodules.self_attention, "is_mtp_layer", is_mtp_layer),
        )

        # self_attn_bda(BiasDropoutFusion) is not supported.

        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon,
            compute_dtype=config.layernorm_compute_dtype
        )

        # NOTE: cross_attention remains disabled here,
        # with GPTModel implementing it as Identity(X) initialization.
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=self.config,
            layer_number=layer_number,
        )

        # cross_attn_bda(BiasDropoutFusion) is not supported.

        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon,
            compute_dtype=config.layernorm_compute_dtype
        )

        self.mlp = build_module(
            submodules.mlp,
            config=self.config,
            layer_number=layer_number,
            **_maybe_kwarg(submodules.mlp, "is_mtp_layer", is_mtp_layer),
        )

        # mlp_bda(BiasDropoutFusion) is not supported.

        self.hidden_states_dropout = Dropout(drop_prob=self.hidden_dropout)
        self.add = mint.add

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            context=None,
            rotary_pos_emb=None,
            prefix_keys_values=None,
            actual_seq_len=None,
            input_ids=None
    ):
        """
        Perform a forward pass through the transformer layer.

        This method implements the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor, optional): Mask tensor for self-attention. Default: None.
            context (Tensor, optional): Context tensor for cross-attention. Default: None.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings. Default: None.
            prefix_keys_values (Tensor, optional): Prefix key-value cache for attention. Default: None.
            actual_seq_len (int, optional): Actual sequence length for variable-length sequences. Default: None.
            input_ids (Tensor, optional): Token IDs of shape [b, s], forwarded to the mlp. Consumed
                only when the mlp is a hash-routed MoELayer; ignored by dense MLP. Default: None.

        Returns:
            Tuple[Tensor, Tensor, float]: A tuple containing:
                - output (Tensor): Transformed hidden states of shape [s, b, h].
                - context (Tensor): Updated context tensor if cross-attention is used,
                  otherwise the same as input context.
        """
        # Note: context parameter is currently unused but kept for API compatibility.
        # It may be used in future cross-attention implementations.

        # Layer norm at the beginning
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Residual connection
        if self.apply_residual_connection_post_norm:
            residual = input_layernorm_output
        else:
            residual = hidden_states

        # Self-Attention
        attention_output = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            prefix_keys_values=prefix_keys_values,
            actual_seq_len=actual_seq_len
        )

        # Dropout
        dropout_output = self.hidden_states_dropout(attention_output)

        norm_input = self.add(residual, dropout_output)

        # Layer norm post the self attention
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(norm_input)

        # Residual connection
        if self.apply_residual_connection_post_norm:
            residual = pre_mlp_layernorm_output
        else:
            residual = norm_input

        mlp_output = self.mlp(pre_mlp_layernorm_output, input_ids=input_ids)

        # Dropout
        dropout_output = self.hidden_states_dropout(mlp_output)

        output = self.add(residual, dropout_output)
        # Note: context parameter is returned for API compatibility but currently unused.
        # It may be deprecated in future versions.
        return output, context


class HyperConnectionTransformerLayer(TransformerLayer):
    """Transformer layer with mHC path."""

    def __init__(self,
                 config: TransformerConfig,
                 submodules: TransformerLayerSubmodules,
                 layer_number: int = 1,
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
        self.n = config.num_residual_streams
        hc_cls = FusedHyperConnectionModule if config.use_fused_mhc else HyperConnectionModule
        self.attn_hc = hc_cls(
            config=config,
            layer_number=layer_number,
        )
        self.ffn_hc = hc_cls(
            config=config,
            layer_number=layer_number,
        )

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            context=None,
            rotary_pos_emb=None,
            prefix_keys_values=None,
            actual_seq_len=None,
            input_ids=None
    ):
        """mHC forward path.

        Args:
            input_ids (Tensor, optional): Token IDs of shape [b, s], forwarded to the mlp on the
                mHC path. Consumed only when the mlp is a hash-routed MoELayer. Default: None.
        """

        streams_before_attn = hidden_states
        aggregated_attn, h_res_attn, h_post_attn = self.attn_hc(hidden_states)

        input_layernorm_output = self.input_layernorm(aggregated_attn)
        attention_output = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            prefix_keys_values=prefix_keys_values,
            actual_seq_len=actual_seq_len
        )

        dropout_output = self.hidden_states_dropout(attention_output)
        hidden_states = self.attn_hc.output_cell(
            h_res_attn, h_post_attn, streams_before_attn, dropout_output)

        streams_before_ffn = hidden_states
        aggregated_ffn, h_res_ffn, h_post_ffn = self.ffn_hc(hidden_states)

        pre_mlp_layernorm_output = self.pre_mlp_layernorm(aggregated_ffn)
        mlp_output = self.mlp(pre_mlp_layernorm_output, input_ids=input_ids)

        dropout_output = self.hidden_states_dropout(mlp_output)
        output = self.ffn_hc.output_cell(
            h_res_ffn, h_post_ffn, streams_before_ffn, dropout_output)

        return output, context
