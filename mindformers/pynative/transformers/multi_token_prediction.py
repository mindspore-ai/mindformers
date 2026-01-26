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
"""Multi-Token Prediction (MTP) module for parallel token prediction during training.

This module enables the model to predict multiple future tokens in a single forward pass,
enhancing training efficiency and context awareness. It can be integrated with
speculative decoding for faster inference by generating draft tokens in parallel.

Note: Typically used only during training (disabled at inference in DeepSeek-V3).
"""

__all__ = ['MultiTokenPredictionBlock', 'MultiTokenPredictionBlock',
           'MultiTokenPredictionLayer', 'MultiTokenPredictionLayerSubmodules',
           'MultiTokenPredictionBlockSubmodules', 'get_mtp_layer_spec']

from dataclasses import dataclass
from typing import Union, List, Optional

from mindspore import nn, Tensor
from mindspore import mint
from mindspore import ops

from mindformers.pynative.loss import CrossEntropyLoss
from mindformers.pynative.layers.layer_norm import get_norm_cls
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.layers.linear import Linear


@dataclass
class MultiTokenPredictionLayerSubmodules:
    """
    Dataclass for specifying the submodules of a MultiTokenPrediction module.

    Args:
        enorm (Union[ModuleSpec, type]): Specification or instance of the
            embedding normalization to be applied.
        hnorm (Union[ModuleSpec, type]): Specification or instance of the
             hidden states normalization to be applied.
        eh_proj (Union[ModuleSpec, type]): Specification or instance of the
            linear projection to be applied.
        transformer_layer (Union[ModuleSpec, type]): Specification
            or instance of the transformer block to be applied.
        layer_norm (Union[ModuleSpec, type]): Specification or instance of the
            final layer normalization to be applied.
    """
    enorm: Union[ModuleSpec, type] = None
    hnorm: Union[ModuleSpec, type] = None
    eh_proj: Union[ModuleSpec, type] = None
    transformer_layer: Union[ModuleSpec, type] = None
    layer_norm: Union[ModuleSpec, type] = None


def get_mtp_layer_spec(transformer_layer_spec: ModuleSpec, normalization, fused_norm=True) -> ModuleSpec:
    """Get the MTP layer spec.

    Args:
        transformer_layer_spec (ModuleSpec): Specification of the transformer layer to use.
        normalization (str): Type of normalization to use.
        fused_norm (bool): Whether to use fused-normalization. Defaults to True.

    Returns:
        ModuleSpec: Module specification of MultiTokenPredictionLayer.
    """
    mtp_layer_spec = ModuleSpec(
        module=MultiTokenPredictionLayer,
        submodules=MultiTokenPredictionLayerSubmodules(
            enorm=get_norm_cls(normalization, fused_norm),
            hnorm=get_norm_cls(normalization, fused_norm),
            eh_proj=Linear,
            transformer_layer=transformer_layer_spec,
            layer_norm=get_norm_cls(normalization, fused_norm),
        ),
    )

    return mtp_layer_spec


class MultiTokenPredictionLayer(nn.Cell):
    """The implementation for Multi-Token Prediction (MTP) which extends
    the prediction scope to multiple future tokens at each position.

    This MTP implementation sequentially predict additional tokens and keep the complete
    causal chain at each prediction depth, by using D sequential modules to predict
    D additional tokens.

    The k-th MTP module consists of:
        - a shared embedding layer
        - a projection matrix
        - a Transformer block
        - a shared output head.

    For the i-th input token at the (k - 1)-th prediction depth, we first combine
    the representation of the i-th token and the embedding of the (i + K)-th token with
    the linear projection. The combined serves as the input of the Transformer block at
    the k-th depth to produce the output representation.

    for more information, please refer to DeepSeek-V3 Technical Report
    https://arxiv.org/abs/2412.19437
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: MultiTokenPredictionLayerSubmodules,
            layer_number: int = 1,
    ):
        super().__init__()
        self.config = config
        self.submodules = submodules
        self.dtype = config.compute_dtype
        self.layer_number = layer_number

        self.enorm = build_module(
            self.submodules.enorm,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon,
            params_dtype=config.params_dtype,
            compute_dtype=config.layernorm_compute_dtype
        )

        self.hnorm = build_module(
            self.submodules.hnorm,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon,
            params_dtype=config.params_dtype,
            compute_dtype=config.layernorm_compute_dtype
        )

        self.eh_proj = build_module(
            self.submodules.eh_proj,
            input_size=self.config.hidden_size * 2,
            output_size=self.config.hidden_size,
            compute_dtype=config.compute_dtype,
            params_dtype=config.params_dtype,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False
        )

        self.transformer_layer = build_module(
            self.submodules.transformer_layer,
            config=self.config,
        )

        self.final_layernorm = build_module(
            self.submodules.layer_norm,
            dim=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            params_dtype=config.params_dtype,
            compute_dtype=config.layernorm_compute_dtype
        )

        self.cat = mint.cat
        self.cast = ops.cast
        self.reshape = mint.reshape

    def construct(self,
                  decoder_input: Tensor,
                  hidden_states: Tensor,
                  attention_mask: Tensor,
                  rotary_pos_emb: Tensor = None,
                  actual_seq_len: Tensor = None):
        """
        Perform the forward pass through the MTP layer.

        Args:
            decoder_input (Tensor): Input tensor of shape [s, b, h] where s is the sequence length, b is the batch size,
                and h is the hidden size. At the (k - 1)-th MTP module, the i-th element of decoder input is the
                embedding of (i + K)-th token.
            hidden_states (Tensor): hidden states tensor of shape [s, b, h] where s is the sequence length, b is the
                batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking self-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            actual_seq_len (Tensor, optional): Actual sequence length tensor.

        Returns:
            Tuple[Tensor, Tensor]: The output hidden states tensor of shape [s, b, h].
        """
        decoder_input = self.enorm(decoder_input)
        hidden_states = self.hnorm(hidden_states)

        # At the (k - 1)-th MTP module, concatenates the i-th token's hidden_states
        # and the (i + K)-th token's embedding, and combine them with linear projection.
        hidden_states = self.cat((decoder_input, hidden_states), -1)
        hidden_states, _ = self.eh_proj(hidden_states)
        hidden_states, _ = self.transformer_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            actual_seq_len=actual_seq_len
        )

        # Layer norm before shared head layer.
        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


@dataclass
class MultiTokenPredictionBlockSubmodules:
    """
    Dataclass for specifying the submodules of a multi token prediction block.

    This class defines the structure for configuring the layers, allowing for
    flexible and customizable architecture designs.

    Args:
        layer_specs (List[ModuleSpec], optional): A list of module specifications for
            the layers within the multi token prediction block. Each specification typically
            defines a complete multi token prediction layer (e.g., shared embedding,
            projection matrix, transformer block, shared output head).
    """

    layer_specs: List[ModuleSpec] = None


def _get_mtp_block_submodules(
        spec: Union[MultiTokenPredictionBlockSubmodules, ModuleSpec]) -> MultiTokenPredictionBlockSubmodules:
    """
    Retrieve or construct MultiTokenPredictionBlockSubmodules based on the provided specification.

    Args:
        spec (Union[MultiTokenPredictionBlockSubmodules, ModuleSpec]): Specification for the
            multi token prediction block submodules.
            Can be either a MultiTokenPredictionBlockSubmodules instance or a ModuleSpec.

    Returns:
        MultiTokenPredictionBlockSubmodules: The submodules for the multi token prediction block.
    """

    # Transformer block submodules.
    if isinstance(spec, MultiTokenPredictionBlockSubmodules):
        return spec
    if isinstance(spec, ModuleSpec):
        if issubclass(spec.module, MultiTokenPredictionBlock):
            return spec.submodules
        raise Exception(f"specialize for {spec.module.__name__}.")
    raise Exception(f"specialize for {type(spec).__name__}.")


class MultiTokenPredictionBlock(nn.Cell):
    """The implementation for Multi-Token Prediction (MTP) which extends
    the prediction scope to multiple future tokens at each position.

    This MTP implementation sequentially predict additional tokens and keep the complete
    causal chain at each prediction depth, by using D sequential modules to predict
    D additional tokens.

    The k-th MTP module consists of a shared embedding layer, a projection matrix,
    a Transformer block, and a shared output head.

    For the i-th input token at the (k - 1)-th prediction depth, we first combine
    the representation of the i-th token and the embedding of the (i + K)-th token with
    the linear projection. The combined serves as the input of the Transformer block at
    the k-th depth to produce the output representation.

    for more information, please refer to DeepSeek-V3 Technical Report
    https://arxiv.org/abs/2412.19437
    """

    def __init__(self, config: TransformerConfig, spec: Union[ModuleSpec]):
        super().__init__()
        self.config = config
        self.submodules = _get_mtp_block_submodules(spec)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        self.calculate_per_token_loss = config.calculate_per_token_loss
        self._build_layers()
        if not self.layers:
            raise ValueError("MultiTokenPredictionBlock must have at least one layer.")

        self.compute_language_model_loss = CrossEntropyLoss(config.calculate_per_token_loss)

        self.cast = ops.cast
        self.cat = mint.cat
        self.shape = ops.shape
        self.slice = ops.strided_slice
        self.zeros = mint.zeros
        self.transpose = mint.transpose
        self.reshape = mint.reshape
        self.ones_like = mint.ones_like



    def _build_layers(self):
        """Building MTP layers."""
        self.layers = nn.CellList()
        for layer_spec in self.submodules.layer_specs:
            mtp_layer = build_module(layer_spec, config=self.config)
            self.layers.append(mtp_layer)

    def roll_tensor(self, tensor):
        """Implement tensor rolling with slice and pad operation.

        Args:
            tensor (Tensor): Input tensor to roll, shape [b, s].

        Returns:
            Tensor: Rolled tensor, shape [b, s].
        """
        bs, seq_len = self.shape(tensor)
        pad_zeros = self.zeros((bs, 1))
        tensor = self.slice(tensor, (0, 1), (bs, seq_len), (1, 1))
        tensor = self.cat((tensor, self.cast(pad_zeros, tensor.dtype)), -1)

        return tensor

    def construct(
            self,
            input_ids: Tensor,
            position_ids: Tensor,
            hidden_states: Tensor,
            attention_mask: Tensor,
            labels: Tensor = None,
            rotary_pos_emb: Tensor = None,
            extra_block_kwargs: dict = None,
            loss_mask: Optional[Tensor] = None,
            embedding: nn.Cell = None,
            output_layer: nn.Cell = None,
            output_weight: Tensor = None,
    ):
        """
        Perform the forward pass through all of the MTP modules.

        Args:
            input_ids (Tensor): Input token IDs with shape [b, s], where b is batch size,
                s is sequence length.
            position_ids (Tensor): Position IDs with shape [b, s].
            hidden_states (Tensor): Hidden states for input token with shape [s, b, h],
                where h is hidden size.
            attention_mask (Tensor): Boolean tensor of shape [b, 1, s, s] for masking self-attention.
            labels (Tensor, optional): Labels tensor with shape [b, s].
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            extra_block_kwargs (dict, optional): Additional keyword arguments for blocks.
            loss_mask (Tensor, optional): Loss mask tensor with shape [b, s].
            embedding (nn.Cell, optional): Embedding layer for token embedding. Default: ``None``.
            output_layer (nn.Cell, optional): Output layer for logits generation. Default: ``None``.
            output_weight (Tensor, optional): Output weight tensor. Default: ``None``.

        Returns:
            Union[Tuple[Tensor, Tensor], Tensor]: The MTP loss tensor.
            - If calculate_per_token_loss is True: Returns tuple of (numerator, denominator)
            - Otherwise, returns scalar loss tensor.
        """
        if labels is None:
            raise ValueError("labels should not be None for calculating multi token prediction loss.")
        if loss_mask is None:
            # if loss_mask is not provided, use all ones as loss_mask
            loss_mask = self.ones_like(labels)

        mtp_loss = 0
        numerator, denominator = 0, 0
        for layer in self.layers:
            # Calc logits for the current Multi-Token Prediction (MTP) layers.
            input_ids = self.roll_tensor(input_ids)
            # embedding
            decoder_input = embedding(
                input_ids=input_ids,
                position_ids=position_ids
            )
            # norm, linear projection and transformer
            hidden_states = layer(
                decoder_input=decoder_input,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                **(extra_block_kwargs or {}),
            )
            # output
            mtp_logits, _ = output_layer(hidden_states, weight=output_weight)
            mtp_logits = self.transpose(mtp_logits, 0, 1)
            mtp_logits = self.reshape(mtp_logits, (-1, mtp_logits.shape[-1]))

            # Calc loss for the current Multi-Token Prediction (MTP) layers.
            labels = self.roll_tensor(labels)
            loss_mask = self.roll_tensor(loss_mask)

            # If the compute_language_model_loss is actually unwrapped CrossEntropyLoss, the inputs should
            # be reshaped manually.
            labels_t = self.reshape(labels, (-1,))
            loss_mask_t = self.reshape(loss_mask, (-1,))

            # config.calculate_per_token_loss is supported in CrossEntropyLoss
            mtp_layer_loss = self.compute_language_model_loss(mtp_logits, labels_t, loss_mask_t)

            mtp_layer_loss_scale = self.mtp_loss_scaling_factor / self.config.mtp_num_layers
            if self.calculate_per_token_loss:
                numerator = numerator + mtp_layer_loss_scale * mtp_layer_loss[0]
                denominator = denominator + mtp_layer_loss[1]
            else:
                mtp_layer_loss = mtp_layer_loss_scale * mtp_layer_loss
                # MTPLossAutoScaler is not supported for now, forward is not effective, backward grad scale=1.0 by default.
                mtp_loss = mtp_loss + mtp_layer_loss

        if self.calculate_per_token_loss:
            return (numerator, denominator)
        return mtp_loss
