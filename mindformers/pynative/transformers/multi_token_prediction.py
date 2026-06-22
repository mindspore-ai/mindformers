# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
from typing import Union, List

from hyper_parallel.core.dtensor.dtensor import DTensor
from mindspore import nn, Tensor, mint, ops
from mindspore.common._grad_function import _Function
from mindspore.common import dtype as mstype
from mindspore.mint.distributed import get_world_size, all_reduce

from mindformers.pynative.loss import CrossEntropyLoss
from mindformers.pynative.layers.layer_norm import get_norm_cls
from mindformers.pynative.transformers.transformer_block import (
    expand_hyper_connection_streams,
    collapse_hyper_connection_streams,
)
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.layers.linear import Linear

# MTP logging
_MTP_LAYER_WISE_LOGGING_TRACKER: dict = {}


class _MTPLossAutoScaler(_Function):
    """An AutoScaler that triggers the backward pass and scales the grad for mtp loss."""

    main_loss_backward_scale: Tensor = Tensor(1.0)

    @staticmethod
    def forward(ctx, output: Tensor, mtp_loss: Tensor):
        """Preserve the mtp by storing it in the context to avoid garbage collection.

        Args:
            output (Tensor): The output tensor.
            mtp_loss (Tensor): The mtp loss tensor.

        Returns:
            Tensor: The output tensor.
        """
        ctx.mtp_loss = mtp_loss
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Compute and scale the gradient for mtp loss.

        Args:
            grad_output (Tensor): The gradient of the output.

        Returns:
            Tuple[Tensor, Tensor]: The gradient of the output, scaled mtp loss
                                               gradient.
        """
        mtp_loss = ctx.mtp_loss
        mtp_loss_backward_scale = _MTPLossAutoScaler.main_loss_backward_scale
        mtp_loss = mtp_loss.to_local() if isinstance(mtp_loss, DTensor) else mtp_loss
        scaled_mtp_loss_grad = mint.ones_like(mtp_loss) * mtp_loss_backward_scale
        return grad_output, scaled_mtp_loss_grad

    @staticmethod
    def set_loss_scale(scale: Tensor):
        """set the scale of the mtp loss.

        Args:
            scale (Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        _MTPLossAutoScaler.main_loss_backward_scale = scale


def get_mtp_layer_wise_logging_tracker() -> dict:
    """Return the mtp layer wise tracker."""
    # pylint: disable=W0602
    global _MTP_LAYER_WISE_LOGGING_TRACKER
    return _MTP_LAYER_WISE_LOGGING_TRACKER


def save_to_mtp_losses_tracker(
        loss: Tensor,
        layer_number: int,
        num_layers: int,
) -> None:
    """Save the mtp loss for logging.
    Args:
        name (str): The name of the loss.
        loss (Tensor): The loss tensor.
        layer_number (int): Layer index of the loss.
        num_layers (int): The number of total layers.
    """
    # Skip mtp loss logging if layer_number is None.
    if layer_number is None:
        return

    tracker = get_mtp_layer_wise_logging_tracker()
    if not tracker:
        tracker["values"] = mint.zeros(num_layers)
    if isinstance(loss, DTensor):
        loss = loss.to_local()
    # Accumulate (not overwrite) the loss for the layer so that all micro-batches
    # within a gradient-accumulation / pipeline step are summed. The callback then
    # divides by ``num_accumulation_steps`` to recover the per-step average. Without
    # this, only the final micro-batch survived and the logged value was off by a
    # factor of ``num_accumulation_steps`` under pipeline parallelism. Mirrors
    # ``save_to_aux_losses_tracker`` in moe_utils.
    if hasattr(loss, "detach"):
        tracker["values"][layer_number] = tracker["values"][layer_number] + loss.detach()
    else:
        tracker["values"][layer_number] = tracker["values"][layer_number] + loss


class MTPLossAutoScaler(nn.Cell):
    """
    Module wrapper for the custom LogSoftmax function.
    """

    @staticmethod
    def construct(output: Tensor, mtp_loss: Tensor):
        """
        Forward pass for LogSoftmax.
        """
        return _MTPLossAutoScaler.apply(output, mtp_loss)


def roll_tensor(tensor: Tensor, shifts: int = 1, dims: int = 0) -> Tensor:
    """
    Shift tensor with zero padding.

    Args:
        tensor (Tensor): Input tensor.
        shifts (int): Shift amount.
        dims (int): Dimension to shift along.

    Returns:
        Tensor: Shifted tensor with zero padding.
    """
    if shifts == 0:
        return tensor

    shape = tensor.shape
    dim_size = shape[dims]
    dtype = tensor.dtype

    if abs(shifts) >= dim_size:
        return ops.zeros(shape, dtype)

    pad_shape = list(shape)
    pad_shape[dims] = abs(shifts)
    pad = ops.zeros(tuple(pad_shape), dtype)
    slices = [slice(None)] * tensor.ndim
    if shifts > 0:
        slices[dims] = slice(0, dim_size - shifts)
        cropped = tensor[tuple(slices)]
        return ops.concat((pad, cropped), axis=dims)

    shifts = -shifts
    slices[dims] = slice(shifts, dim_size)
    cropped = tensor[tuple(slices)]
    return ops.concat((cropped, pad), axis=dims)


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


def get_mtp_layer_spec(transformer_layer_spec: ModuleSpec, normalization: str, fused_norm=True) -> ModuleSpec:
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
            is_mtp_layer: bool = False,
    ):
        super().__init__()
        # MTP + dsv4_hybrid + hash-MoE co-support is not yet validated end-to-end. The router
        # short-circuit makes MTP layers non-hash, but fail loudly here to document the boundary.
        if is_mtp_layer and getattr(config, "experimental_attention_variant", None) == "dsv4_hybrid" \
                and getattr(config, "moe_n_hash_layers", 0) > 0:
            raise NotImplementedError(
                "MTP + dsv4_hybrid + hash-MoE co-support is not yet validated; "
                "set moe_n_hash_layers=0 or disable mtp_num_layers."
            )
        self.config = config
        self.is_mtp_layer = is_mtp_layer
        self.submodules = submodules
        self.dtype = config.compute_dtype
        self.layer_number = layer_number

        self.enorm = build_module(
            self.submodules.enorm,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon,
            compute_dtype=config.layernorm_compute_dtype
        )

        self.hnorm = build_module(
            self.submodules.hnorm,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon,
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
        )

        self.transformer_layer = build_module(
            self.submodules.transformer_layer,
            config=self.config,
            layer_number=self.layer_number,
            is_mtp_layer=self.is_mtp_layer,
        )

        self.final_layernorm = build_module(
            self.submodules.layer_norm,
            dim=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            compute_dtype=config.layernorm_compute_dtype
        )

        self.cat = mint.cat
        self.cast = ops.cast
        self.reshape = mint.reshape

        # Mirror TransformerBlock's hyper-connection handling: with mHC enabled
        # the inner transformer_layer computes on packed residual streams
        # (s, b, n*h), so the MTP layer must expand its (s, b, h) input into n
        # streams before the layer and collapse them back afterwards.
        self.hc = config.enable_hyper_connections
        if self.hc:
            self.hc_num_streams = config.num_residual_streams
            self.hc_hidden_size = config.hidden_size

    def construct(
            self,
            input_ids: Tensor,
            position_ids: Tensor,
            hidden_states: Tensor,
            attention_mask: Tensor,
            rotary_pos_emb: Tensor = None,
            actual_seq_len: Tensor = None,
            embedding=None,
            mscale: float = 1.0,
    ):
        """
        Perform the forward pass through the MTP layer.

        Args:
            input_ids (Tensor): Input token IDs .
            position_ids (Tensor): Positional IDs of the input tokens.
            hidden_states (Tensor): Hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            actual_seq_len (Tensor, optional): Actual sequence length tensor.

        Returns:
            Tuple[Tensor, Tensor]: The output hidden states tensor of shape [s, b, h].
        """
        input_ids, position_ids, decoder_input, hidden_states = self._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            embedding=embedding,
            hidden_states=hidden_states,
        )

        decoder_input = self.enorm(decoder_input)
        hidden_states = self.hnorm(hidden_states)
        # At the (k - 1)-th MTP module, concatenates the i-th token's hidden_states
        # and the (i + K)-th token's embedding, and combine them with linear projection.
        hidden_states = mint.cat((decoder_input, hidden_states), -1)
        hidden_states = self.eh_proj(hidden_states)
        if self.hc:
            # Expand (s, b, h) -> (s, b, n*h) packed residual streams,
            # same as TransformerBlock does for the main decoder.
            hidden_states = expand_hyper_connection_streams(
                hidden_states, self.hc_num_streams, self.hc_hidden_size
            )
        hidden_states, _ = self.transformer_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            actual_seq_len=actual_seq_len,
            mscale=mscale
        )
        if self.hc:
            # Collapse the streams back to (s, b, h) by averaging,
            # same as TransformerBlock does before its final layernorm.
            hidden_states = collapse_hyper_connection_streams(
                hidden_states, self.hc_num_streams, self.hc_hidden_size
            )

        # Layer norm before shared head layer.
        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, input_ids, position_ids

    def _get_embeddings(
            self,
            input_ids: Tensor,
            position_ids: Tensor,
            embedding: nn.Cell,
            hidden_states: Tensor,
    ):
        """
        Preprocesses input data for the Multi-Token Prediction (MTP) layers.

        This function computes the decoder input and sends updated input_ids and position_ids to
        the next layer.

        Args:
            input_ids (Tensor): The input token IDs.
            position_ids (Tensor): The position IDs corresponding to the input tokens.
            embedding (Callable): The embedding module
                from gpt model to compute the decoder input.
            hidden_states (Tensor): hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            packed_seq_params (PackedSeqParams): Parameters for packed sequence processing.
        """
        # Calc logits for the current Multi-Token Prediction (MTP) layers.
        input_ids = roll_tensor(
            input_ids,
            shifts=-1,
            dims=-1,
        )
        if position_ids is not None:
            position_ids = roll_tensor(
                position_ids,
                shifts=-1,
                dims=-1,
            )
        # embedding
        decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)

        return input_ids, position_ids, decoder_input, hidden_states


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
        self.mtp_loss_auto_scaler = MTPLossAutoScaler()
        self.concat_hidden_states = _MTPConcat()

    def _build_layers(self):
        """Building MTP layers."""
        self.layers = nn.CellList()
        for i, layer_spec in enumerate(self.submodules.layer_specs):
            mtp_layer = build_module(
                layer_spec,
                config=self.config,
                layer_number=self.config.num_layers + i,
                is_mtp_layer=True,
            )
            self.layers.append(mtp_layer)

    def construct(
            self,
            input_ids: Tensor,
            position_ids: Tensor,
            hidden_states: Tensor,
            attention_mask: Tensor,
            rotary_pos_emb: Tensor = None,
            extra_block_kwargs: dict = None,
            embedding: nn.Cell = None,
            actual_seq_len: Tensor = None,
            mscale: float = 1.0,
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
            embedding (nn.Cell, optional): Embedding layer for token embedding. Default: ``None``.
            output_layer (nn.Cell, optional): Output layer for logits generation. Default: ``None``.

        Returns:
            Tensor: The MTP hidden states tensor of shape [s, b, h].
        """
        _ = extra_block_kwargs
        hidden_states_list = [hidden_states]
        for layer in self.layers:
            hidden_states, _, _ = layer(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                embedding=embedding,
                actual_seq_len=actual_seq_len,
                mscale=mscale,
            )
            hidden_states_list.append(hidden_states)
        hidden_states = self.concat_hidden_states(hidden_states_list)
        return hidden_states


class _MTPConcat(nn.Cell):
    # pylint: disable=W0246
    def __init__(self):
        super().__init__()

    def construct(self, hidden_states):
        if not isinstance(hidden_states, list):
            raise ValueError("hidden_states should be a list of tensors.")
        hidden_states = mint.cat(hidden_states, dim=0)
        return hidden_states


def process_mtp_loss(
        hidden_states_list,
        labels,
        loss_mask,
        output_layer,
        output_weight,
        compute_language_model_loss,
        config,
        mtp_loss_auto_scaler,
):
    """Process multi-token prediction loss and inject gradients for MTP layers."""
    if labels is None:
        raise ValueError("labels should not be None for calculating multi token prediction loss.")

    hidden_states_list = mint.chunk(hidden_states_list, 1 + config.mtp_num_layers, dim=0)
    hidden_states = hidden_states_list[0]

    # Calc loss for the current Multi-Token Prediction (MTP) layers.
    mtp_labels = labels.clone()
    if loss_mask is None:
        # if loss_mask is not provided, use all ones as loss_mask
        loss_mask = mint.ones_like(mtp_labels)
    mtp_loss_mask = loss_mask

    for mtp_layer_number in range(config.mtp_num_layers):
        mtp_logits = output_layer(hidden_states_list[mtp_layer_number + 1], weight=output_weight)
        mtp_labels = roll_tensor(mtp_labels, shifts=-1, dims=-1)
        mtp_loss_mask = roll_tensor(mtp_loss_mask, shifts=-1, dims=-1)
        mtp_loss_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers

        if getattr(config, "chunk_loss_num", 0) > 1:
            mtp_logits = mtp_logits.transpose(0, 1)
            mtp_loss = compute_language_model_loss(mtp_labels, mtp_logits, mtp_loss_mask)
            save_to_mtp_losses_tracker(
                mtp_loss_scale * mtp_loss,
                mtp_layer_number,
                config.mtp_num_layers,
            )
            hidden_states = mtp_loss_auto_scaler(hidden_states, mtp_loss_scale * mtp_loss)
        else:
            mtp_logits = mtp_logits.transpose(0, 1).reshape((-1, mtp_logits.shape[-1]))
            mtp_loss_mask = mint.reshape(mtp_loss_mask, (-1,))
            mtp_loss_mask = ops.cast(mtp_loss_mask, mstype.float32)
            mtp_loss = compute_language_model_loss(mtp_labels, mtp_logits)
            mtp_loss = mint.mul(mtp_loss, mtp_loss_mask)
            mtp_loss_sum = mtp_loss.sum()

            num_tokens = mtp_loss_mask.sum()
            mtp_loss_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers
            save_to_mtp_losses_tracker(
                mtp_loss_scale * mtp_loss_sum / num_tokens,
                mtp_layer_number,
                config.mtp_num_layers,
            )

            if config.calculate_per_token_loss:
                hidden_states = mtp_loss_auto_scaler(
                    hidden_states, mtp_loss_scale * mtp_loss
                )
            else:
                hidden_states = mtp_loss_auto_scaler(
                    hidden_states, mtp_loss_scale * mtp_loss / num_tokens
                )
    return hidden_states


def track_mtp_metrics(group=None, group_size=None):
    """Track and aggregate MTP loss metrics across distributed ranks."""
    tracker = get_mtp_layer_wise_logging_tracker()
    if not tracker:
        return None

    mtp_losses = tracker["values"].clone()
    if group_size is None:
        group_size = get_world_size()
    if group_size > 1:
        all_reduce(mtp_losses, op=ops.ReduceOp.SUM, group=group)
        mtp_losses /= group_size

    clear_mtp_losses_tracker()
    return mtp_losses


def clear_mtp_losses_tracker() -> None:
    """Clear the accumulated MTP losses after they have been logged for a step."""
    tracker = get_mtp_layer_wise_logging_tracker()
    if "values" not in tracker:
        return
    for value in tracker["values"]:
        value.zero_()
