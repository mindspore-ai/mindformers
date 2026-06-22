# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Modified to adapt to MindSpore pynative mode.
# Main changes: PyTorch->MindSpore, removed parallel/checkpointing/CUDA graph/CPU offloading features,
# removed pre_process/post_process parameters, simplified forward pass.
# Main changes: keeps pre_process/post_process stage flags for PP boundary semantics.
"""Transformer Block"""
from dataclasses import dataclass
from typing import Union, List, Optional
from mindspore import nn, Tensor, mint
from mindformers.pynative.layers.layer_norm import get_norm_cls
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.pynative.transformers.transformer_layer import BaseTransformerLayer
from mindformers.tools.logger import logger


def expand_hyper_connection_streams(hidden_states: Tensor, num_streams: int, hidden_size: int) -> Tensor:
    """Expand (s, b, h) hidden states into n packed mHC residual streams (s, b, n*h).

    Shared by TransformerBlock (main decoder) and the MTP layer so both feed the inner
    transformer layers the packed residual-stream layout their attn_hc/ffn_hc expect.
    """
    seq_len = hidden_states.shape[0]
    hidden_states = mint.unsqueeze(hidden_states, 2)
    hidden_states = mint.tile(hidden_states, (1, 1, num_streams, 1))
    return mint.reshape(hidden_states, (seq_len, -1, num_streams * hidden_size))


def collapse_hyper_connection_streams(hidden_states: Tensor, num_streams: int, hidden_size: int) -> Tensor:
    """Collapse n packed mHC residual streams (s, b, n*h) back to (s, b, h) by averaging.

    Inverse of :func:`expand_hyper_connection_streams`; shared by TransformerBlock and
    the MTP layer before their final layernorm.
    """
    seq_len = hidden_states.shape[0]
    hidden_states = mint.reshape(hidden_states, (seq_len, -1, num_streams, hidden_size))
    return mint.mean(hidden_states, dim=2)


@dataclass
class TransformerBlockSubmodules:
    """
    Class for specifying the submodules of a transformer block.

    This class defines the structure for configuring the layers and normalization
    within a transformer block, allowing for flexible and customizable architecture designs.

    Args:
        layer_specs (List[ModuleSpec], optional): Module specifications for
            the layers within the transformer block. Each specification typically
            defines a complete transformer layer (e.g., self-attention, feed-forward network).
        layer_norm (Optional[Union[ModuleSpec, mindspore.nn.Cell]], optional): Specification
            or instance of the layer normalization to be applied.
    """
    layer_specs: List[ModuleSpec] = None
    layer_norm: Optional[Union[ModuleSpec, nn.Cell]] = None


def _get_block_submodules(
        config: TransformerConfig, spec: Union[TransformerBlockSubmodules, ModuleSpec]
) -> TransformerBlockSubmodules:
    """
    Retrieve or construct TransformerBlockSubmodules based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[TransformerBlockSubmodules, ModuleSpec]): Specification for the
            transformer block submodules. Can be either a TransformerBlockSubmodules
            instance or a ModuleSpec.

    Returns:
        TransformerBlockSubmodules: The submodules for the transformer block.
    """

    # Transformer block submodules.
    if isinstance(spec, TransformerBlockSubmodules):
        return spec

    if isinstance(spec, ModuleSpec):
        if issubclass(spec.module, TransformerBlock):
            return spec.submodules
        if issubclass(spec.module, BaseTransformerLayer):
            num_layers = config.num_layers
            return TransformerBlockSubmodules(
                layer_specs=[spec] * num_layers, layer_norm=get_norm_cls(config.normalization)
            )
        raise Exception(f"specialize for {spec.module.__name__}.")
    raise Exception(f"specialize for {type(spec).__name__}.")

class CustomIndexCellList(nn.CellList):
    """
    Custom CellList that supports indexed access.
    """
    def __init__(self, indexed_cells=None):
        super().__init__()

        if indexed_cells is not None:
            for idx, cell in indexed_cells:
                if not isinstance(cell, nn.Cell):
                    raise TypeError(f"Item must be the subclass of the nn.Cell, but get {type(cell).__name__}")

                self._cells[str(idx)] = cell

    def __setitem__(self, idx, cell):
        if not isinstance(cell, nn.Cell):
            raise TypeError("Item must be the subclass of the nn.Cell")
        self._cells[str(idx)] = cell

    def __getitem__(self, idx):
        return self._cells[str(idx)]

    def get_first_cell(self):
        if not self._cells:
            return None
        return next(iter(self._cells.values()))

class TransformerBlock(nn.Cell):
    """
    Transformer class.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[TransformerBlockSubmodules, ModuleSpec]): Specification for the
            transformer block submodules. Can be either a TransformerBlockSubmodules
            instance or a ModuleSpec.
        post_layer_norm (bool): Insert normalization layer at the end of transformer block. Default: True.
        pre_process (bool): Whether this block is in PP pre-process stage. Default: True.
        post_process (bool): Whether this block is in PP post-process stage. Default: True.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(S, B, H)` where S is sequence length,
          B is batch size, and H is hidden size.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **rotary_pos_emb** (Tensor, optional) - Tensor of rotary position embedding. Default: None.
        - **prefix_keys_values** (optional) - List of prefix key-value tensors for each layer. Default: None.
        - **actual_seq_len** (optional) - Actual sequence length for variable-length sequences. Default: None.

    Outputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(S, B, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            spec: Union[TransformerBlockSubmodules, ModuleSpec],
            post_layer_norm: bool = True,
            pre_process: bool = True,
            post_process: bool = True,
            layer_start: int = 0,
            layer_end: int = None,
    ):
        super().__init__()

        self.config = config
        self.submodules = _get_block_submodules(config, spec)
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.num_layers = config.num_layers
        self.hc = config.enable_hyper_connections
        self.layer_start = layer_start
        self.layer_end = layer_end
        if config.enable_hyper_connections:
            self.n = config.num_residual_streams
            self.hidden_size = config.hidden_size
        cp = config.context_parallel_size if config.context_parallel_size is not None else 1
        if config.sequence_parallel and cp > 1:
            logger.warning("The context parallel way conflicts with sequence parallel way. "
                           "The sequence parallel way has no effect and ignored.")
        self.seq_length_in_cfg = config.seq_length

        self._build_layers(config)

    def _build_layers(self, config: TransformerConfig):
        """build transformer layers."""
        # Transformer layers.
        if self.hc:
            logger.info(
                "PyNative mHC is enabled: num_residual_streams=%s, mhc_sinkhorn_iterations=%s, "
                "mhc_init_gating_factor=%s",
                self.n,
                config.mhc_sinkhorn_iterations,
                config.mhc_init_gating_factor,
            )

        custom_layers = []
        for layer_id in range(self.layer_start, self.layer_end + 1):
            layer = build_module(self.submodules.layer_specs[layer_id], config=config, layer_number=layer_id)
            custom_layers.append((layer_id, layer))
        self.layers = CustomIndexCellList(custom_layers)

        if self.post_layer_norm and self.post_process:
            self.final_layernorm = build_module(self.submodules.layer_norm,
                                                dim=config.hidden_size,
                                                eps=config.layernorm_epsilon,
                                                compute_dtype=config.layernorm_compute_dtype)
        else:
            self.final_layernorm = None

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def has_final_layernorm_in_this_stage(self):
        """Whether this PP stage contains the final layernorm."""
        return self.post_layer_norm and self.post_process

    def construct(self,
                  hidden_states: Tensor,
                  attention_mask: Tensor,
                  rotary_pos_emb: Tensor = None,
                  prefix_keys_values=None,
                  actual_seq_len=None,
                  input_ids=None,
                  mscale=1.0):
        """
        Construct function of transformer block.

        Args:
            hidden_states (Tensor): Input tensor of shape (S, B, H) where S is sequence length,
                B is batch size, and H is hidden size.
            attention_mask (Tensor): Attention mask tensor.
            rotary_pos_emb (Tensor, optional): Rotary position embedding tensor. Default: None.
            prefix_keys_values (optional): List of prefix key-value tensors for each layer.
                Each element should be a tuple or list of (key, value) tensors. Default: None.
            actual_seq_len (optional): Actual sequence length for variable-length sequences. Default: None.
            input_ids (Tensor, optional): Token IDs of shape (B, S), forwarded to each layer for
                hash-based MoE routing. Consumed only by hash layers. Default: None.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - hidden_states (Tensor): Output tensor of shape (S, B, H).
        """
        if self.hc and self.pre_process:
            hidden_states = expand_hyper_connection_streams(hidden_states, self.n, self.hidden_size)

        for index in range(self.layer_start, self.layer_end + 1):
            layer = self._get_layer(index)
            prefix_kv = prefix_keys_values[index] if prefix_keys_values is not None else None
            hidden_states, _ = layer(
                hidden_states,
                attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                prefix_keys_values=prefix_kv,
                actual_seq_len=actual_seq_len,
                input_ids=input_ids,
                mscale=mscale
            )

        if self.hc and self.has_final_layernorm_in_this_stage():
            hidden_states = collapse_hyper_connection_streams(hidden_states, self.n, self.hidden_size)

        # final layernorm.
        if self.post_layer_norm and self.post_process:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states
