# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Modification points:
# 1. Replace all interfaces with MindSpore TransFormers'.
# 2. Add some input parameters for MindSpore TransFormers.
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
"""mindformers GPT model"""
__all__ = ['GPTModel']

from typing import Literal, Optional, Union

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate

from mindspore import Tensor, dtype, nn, mint, ops
from mindspore.mint.distributed import all_reduce, get_world_size

from mindformers.tools.logger import logger
from mindformers.pynative.loss.loss import CrossEntropyLoss, ChunkCrossEntropyLoss
from mindformers.pynative.distributed.utils import vocab_parallel_shard_dim
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.pynative.layers.mask_generate import CausalMaskGenerate
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.base_models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from mindformers.pynative.base_models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from mindformers.pynative.base_models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from mindformers.pynative.base_models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from mindformers.pynative.transformers.transformer_block import TransformerBlock, TransformerBlockSubmodules
from mindformers.pynative.transformers.multi_token_prediction import MultiTokenPredictionBlock, MTPLossAutoScaler
from mindformers.pynative.layers.linear import Linear
from mindformers.pynative.optimizer.muon_utils import make_muon_fns
from mindformers.pynative.transformers.multi_token_prediction import process_mtp_loss
from mindformers.pynative.dtensor_compat import inplace_copy


class GPTModel(nn.Cell):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig):
            Transformer config.
        transformer_layer_spec (ModuleSpec):
            Specifies module to use for transformer layers.
        vocab_size (int):
            Vocabulary size.
        max_sequence_length (int):
            maximum size of sequence. This is used for positional embedding.
        pre_process (bool, optional):
            Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional):
            Include an output layer (used with pipeline parallelism). Defaults to True.
        share_embeddings_and_output_weights (bool, optional):
            When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):
            Position embedding type. Defaults to 'learned_absolute'.
        rotary_percent (float, optional):
            Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional):
            Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'.
            Defaults to 10000.
        rope_scaling (bool, optional): Toggle RoPE scaling. Defaults to False.
        rope_scaling_factor (float): RoPE scaling factor. Defaults to 8.
        seq_len_interpolation_factor (Optional[float], optional):
            scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
            self,
            config: TransformerConfig,
            transformer_layer_spec: Union[TransformerBlockSubmodules, ModuleSpec],
            vocab_size: int,
            max_sequence_length: int,
            pre_process: bool = True,
            post_process: bool = True,
            share_embeddings_and_output_weights: bool = False,
            position_embedding_type: Literal['learned_absolute', 'rope', 'yarn', 'none'] = 'learned_absolute',
            rotary_percent: float = 1.0,
            rotary_base: int = 10000,
            rope_scaling: bool = False,
            rope_scaling_factor: float = 8.0,
            seq_len_interpolation_factor: Optional[float] = None,
            layer_start: int = 0,
            layer_end: int = None,
            vp_size: int = 1,
            stage_idx: int = 0,
    ):
        super().__init__()

        self.config = config
        self.transformer_layer_spec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.layer_start = layer_start
        self.layer_end = layer_end if layer_end is not None else config.num_layers - 1
        self.pre_process = pre_process and (self.layer_start == 0)
        self.post_process = post_process and (self.layer_end == config.num_layers - 1)
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.use_attn_mask_compression = config.use_attn_mask_compression or config.use_eod_attn_mask_compression
        self.return_logits = False

        if hasattr(self.config, 'position_embedding_type'):
            # By default, use the position_embedding_type configuration in TransformerConfig.
            self.position_embedding_type = self.config.position_embedding_type
        else:
            self.position_embedding_type = position_embedding_type

        self.rotary_percent = rotary_percent

        if hasattr(self.config, 'rotary_base'):
            # By default, use the rotary_base configuration in TransformerConfig.
            self.rotary_base = self.config.rotary_base
        else:
            self.rotary_base = rotary_base

        self.rotary_scaling = rope_scaling
        self.rope_scaling_factor = rope_scaling_factor
        if seq_len_interpolation_factor is not None:
            self.rotary_seq_len_interpolation_factor = seq_len_interpolation_factor
        else:
            self.rotary_seq_len_interpolation_factor = config.rotary_seq_len_interpolation_factor
        self.seq_length = config.seq_length
        mtp_block_spec = None
        if config.mtp_num_layers:
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec)
        self.mtp_process = mtp_block_spec is not None and self.post_process
        self.stage_idx = stage_idx
        self.vp_size = vp_size

        # get value from config
        self.use_eod_attn_mask_compression = config.use_eod_attn_mask_compression
        self.init_method = config.init_method
        self.compute_dtype = config.compute_dtype
        self.hidden_size = config.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.pad_token_id = config.pad_token_id
        self.ignore_token_id = config.ignore_token_id
        self.calculate_per_token_loss = config.calculate_per_token_loss
        self.chunk_loss_num = config.chunk_loss_num
        logger.info(
            "GPTModel chunk loss config: chunk_loss_num=%s, calculate_per_token_loss=%s",
            self.chunk_loss_num,
            self.calculate_per_token_loss
        )

        # Internally generates AttentionMask.
        self.casual_mask = CausalMaskGenerate(
            seq_length=config.seq_length,
            compute_type=config.compute_dtype,
            is_dynamic=config.is_dynamic,
            pad_token_id=config.pad_token_id,
            use_attn_mask_compression=self.use_attn_mask_compression,
        )

        # Embeddings
        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=self.position_embedding_type,
            )

        # rope
        # The MTP implementation pre-computes RotaryEmbedding
        # (unlike Megatron v0.12.0's real-time generation) to minimize dynamic memory usage.
        self.use_rotary_position_embeddings = self.position_embedding_type in ['rope', 'yarn']
        # Non-CP uses self-generated positions (use_position_ids=False); context parallel
        # flips this to True at setup time (see apply_context_parallel_attention).
        if self.position_embedding_type == 'rope':
            if config.multi_latent_attention:
                self.rotary_pos_emb = RotaryEmbedding(
                    kv_channels=config.qk_pos_emb_head_dim,
                    rotary_percent=config.rotary_percent,
                    rotary_interleaved=self.config.rotary_interleaved,
                    seq_len_interpolation_factor=seq_len_interpolation_factor,
                    rotary_base=config.rotary_base,
                    rope_scaling=rope_scaling,
                    rope_scaling_factor=rope_scaling_factor,
                    use_position_ids=False
                )
            else:
                self.rotary_pos_emb = RotaryEmbedding(
                    kv_channels=self.config.kv_channels,
                    rotary_percent=rotary_percent,
                    rotary_interleaved=self.config.rotary_interleaved,
                    seq_len_interpolation_factor=seq_len_interpolation_factor,
                    rotary_base=rotary_base,
                    rope_scaling=rope_scaling,
                    rope_scaling_factor=rope_scaling_factor,
                    use_position_ids=False
                )
        elif self.position_embedding_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                kv_channels=config.qk_pos_emb_head_dim,
                rotary_percent=config.rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=config.rotary_base,
                scaling_factor=config.rotary_scaling_factor,
                original_max_position_embeddings=config.max_position_embeddings,
                beta_fast=config.beta_fast,
                beta_slow=config.beta_slow,
                mscale=config.mscale,
                mscale_all_dim=config.mscale_all_dim,
                use_position_ids=False
            )
        elif self.position_embedding_type == 'mrope':
            raise NotImplementedError("position_embedding_type = mrope is not supported now.")

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            # The corresponding Megatron v0.12.0 module's forward pass has this logic disabled by default,
            # so it won't cause significant impact.
            pre_process=self.pre_process,
            post_process=self.post_process,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
        )

        # Output
        if self.post_process or self.mtp_process:
            skip_weight_param_allocation = self.pre_process and self.share_embeddings_and_output_weights
            self.output_layer = Linear(input_size=self.hidden_size,
                                       output_size=self.vocab_size,
                                       init_method=self.init_method,
                                       bias=False,
                                       skip_weight_param_allocation=skip_weight_param_allocation,
                                       compute_dtype=self.config.compute_dtype,
                                       params_dtype=self.config.params_dtype)
            config.model_parallel = config.tensor_model_parallel_size
            loss_cls = ChunkCrossEntropyLoss if self.chunk_loss_num > 1 else CrossEntropyLoss
            self.loss = loss_cls(
                calculate_per_token_loss=self.calculate_per_token_loss,
                chunk_loss_num=self.chunk_loss_num,
                config=config
            )

        self.mtp = None
        if self.mtp_process:
            self.mtp = MultiTokenPredictionBlock(config=self.config, spec=mtp_block_spec)
            self.mtp_loss_auto_scaler = MTPLossAutoScaler()

        # operations
        self.cast = ops.cast
        self.concat_prefix = mint.concat
        self.zeros = mint.zeros
        self.not_equal = mint.not_equal
        self.reshape = mint.reshape
        self.mul = mint.mul
        self.add = mint.add
        self.sub = mint.sub
        self.sign = mint.sign
        self.transpose = mint.permute
        self.assign = mint.clone

    def _should_use_chunked_loss(self):
        """Whether current runtime state should use chunked loss."""
        return self.training and self.chunk_loss_num > 1 and not self.return_logits

    # pylint: disable=arguments-differ
    def construct(
            self,
            input_ids: Tensor,
            position_ids: Tensor = None,
            attention_mask: Tensor = None,
            decoder_input: Tensor = None,
            labels: Tensor = None,
            loss_mask=None,
            actual_seq_len=None,
    ):
        """GPTModel construct.

        Args:
            input_ids (Tensor): The input tensor of token IDs.
            position_ids (Tensor, optional): Position ID tensor, used to specify the position
            of each token. Default is None.
            attention_mask (Tensor, optional): Attention mask tensor, used to mask padding
            tokens. Default is None.
            decoder_input (Tensor, optional): Decoder input tensor. Default is None.
            labels (Tensor, optional): The label tensor, used for calculating the loss.
            Default is None.
            loss_mask (Tensor, optional): Loss mask tensor, used to specify which positions
            are included in the loss calculation. Default is None.
            actual_seq_len (Tensor, optional): Actual sequence length tensor. Default is None.
        """
        if actual_seq_len is not None:
            actual_seq_len = self.reshape(actual_seq_len, (-1,))

        # Mindspore support TND layout by using actual_seq_len,
        # which indicates the partial seq_lens of eod sequences for compression mask.
        # Check mindformers.dataset.blended_datasets.gpt_dataset._get_eod_attention_mask() for implement details.

        labels, attention_mask, loss_mask = self._preprocess_input_labels_and_masks(
            input_ids, labels, attention_mask, loss_mask)

        hidden_states, rotary_pos_emb, mscale = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            actual_seq_len=actual_seq_len
        )

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if self.mtp_process:
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                embedding=self.embedding,
                actual_seq_len=actual_seq_len,
                mscale=mscale
            )

        if not self.post_process:
            return hidden_states

        if self.mtp_process:
            hidden_states = process_mtp_loss(
                hidden_states_list=hidden_states,
                labels=labels,
                loss_mask=loss_mask,
                output_layer=self.output_layer,
                output_weight=output_weight,
                compute_language_model_loss=self.compute_language_model_loss,
                config=self.config,
                mtp_loss_auto_scaler=self.mtp_loss_auto_scaler,
            )

        # logits origin shape is [s b h], transform it to [b*s h].
        hidden_states = self.cast(hidden_states, dtype.bfloat16)
        logits = self.output_layer(hidden_states, output_weight)
        need_chunked_loss = self._should_use_chunked_loss()
        if logits.ndim > 2 and not need_chunked_loss:
            logits = self.transpose(logits, (1, 0, 2))
            logits = self.reshape(logits, (-1, logits.shape[-1]))

        if not self.training or self.return_logits:
            # With enable_loss_parallel the logits stay vocab-sharded (Shard(-1)); when we
            # return them instead of computing the loss, gather to full vocab so callers
            # (generation, metrics) see the same replicated logits as without loss parallel.
            if vocab_parallel_shard_dim(logits) is not None:
                logits = logits.redistribute(
                    logits.device_mesh, (Replicate(),) * len(logits.placements))
            return self.cast(logits, dtype.float32).contiguous()

        # labels origin shape is [b s], Transpose is not required.
        if need_chunked_loss:
            if logits.ndim != 3:
                raise ValueError(f"Chunked loss expects 3D logits, but got shape {logits.shape}.")
            logits = self.transpose(logits, (1, 0, 2))
            loss = self.compute_language_model_loss(labels, logits, loss_mask)
        else:
            loss = self.compute_language_model_loss(labels, logits, loss_mask)
        if self.calculate_per_token_loss:
            numerator0, denominator0 = loss
            return numerator0, denominator0
        # Only the loss is consumed downstream (compute_loss / compute_pp_loss take
        # output[0]); logits and hidden_states were detached and never read. Keep the
        # tuple shape so the [0]-indexing consumers stay unchanged.
        return (loss,)

    def language_model(
            self,
            input_ids,
            position_ids,
            attn_mask,
            decoder_input,
            tokentype_ids=None,
            prefix_keys_values=None,
            actual_seq_len=None
    ):
        """decoder output.

        Args:
            input_ids (Tensor): The input tensor of token IDs.
            position_ids (Tensor, optional): Position ID tensor, used to specify the position
            of each token. Default is None.
            attn_mask (Tensor, optional): Attention mask tensor, used to mask padding
            tokens. Default is None.
            decoder_input (Tensor, optional): Decoder input tensor. Default is None.
            tokentype_ids (Tensor, optional): Token's type ID. Default is None.
            prefix_keys_values (Tensor, optional): Prefix key-value pairs, used for
            scenarios such as prefix tuning. The default value is None.
            actual_seq_len (Tensor, optional): Actual sequence length tensor. Default is None.
        """
        bs, seq_len = input_ids.shape
        # Encoder embedding
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids, position_ids, tokentype_ids=tokentype_ids)
        else:
            decoder_input = None

        # rope
        rotary_pos_emb = None
        mscale = 1.0
        if self.use_rotary_position_embeddings:
            rotary_pos_emb, mscale = self.rotary_pos_emb(seq_len, position_ids=position_ids)

        if prefix_keys_values is not None:
            if attn_mask is None:
                raise ValueError("attn_mask should not be None when prefix_keys_values is not None!")
            if self.config.use_attn_mask_compression or attn_mask.ndim != 4:
                raise ValueError("use_attn_mask_compression should be False when prefix_keys_values is not None! "
                                 f"And attn_mask.ndim should be 4, but got {attn_mask.ndim}")

            # prefix_key_values shape num_layers*(2, B, prefix_len, kv_num*kv_channel)
            bs, seq_len = input_ids.shape
            prefix_length = prefix_keys_values[0].shape[2]
            prefix_mask = self.zeros((bs, 1, seq_len, prefix_length), attn_mask.dtype)
            # (B, 1, S, S) -> (B, 1, S, S+prefix_len)
            attn_mask = self.concat_prefix((prefix_mask, attn_mask))

        # Run decoder.
        # input_ids is forwarded so hash-based MoE layers can look up tid2eid;
        # default None keeps non-hash models bit-for-bit unchanged.
        hidden_states = self.decoder(
            decoder_input,
            attn_mask,
            rotary_pos_emb,
            prefix_keys_values,
            actual_seq_len,
            input_ids=input_ids,
            mscale=mscale
        )

        return hidden_states, rotary_pos_emb, mscale

    def shared_embedding_or_output_weight(self):
        """Gets the embedding weight or output logit weights when share embedding and output weights set to True.

        Returns:
            Tensor: During pre-processing it returns the input embeddings weight while during post-processing
            it returns the final output layers weight
        """
        if self.pre_process:
            return self.embedding.word_embeddings.weight
        if self.post_process:
            return self.output_layer.weight
        return None

    def compute_language_model_loss(self,
                                    labels: Tensor,
                                    logits: Tensor,
                                    loss_mask: Tensor=None
                                    ):
        """Post-processing of language model output.

        Args:
            labels (Tensor): Labels.
            logits (Tensor): Logit.
            loss_mask (Tensor): Loss mask.

        Returns:
            output (Tensor): Output loss.
        """
        return self.loss(logits, labels, loss_mask)

    def _iter_self_attentions(self):
        """Yield (layer_idx, param_name_prefix, self_attention_module) for every transformer / MTP layer."""
        for i in range(self.layer_start, self.layer_end + 1):
            self_attn = self.decoder.layers[i].self_attention
            yield i, f"decoder.layers.{i}.self_attention", self_attn

        mtp_num_layers = getattr(self.config, "mtp_num_layers", 0) or 0
        if mtp_num_layers and getattr(self, "mtp", None) is not None:
            for i in range(mtp_num_layers):
                self_attn = self.mtp.layers[i].transformer_layer.self_attention
                yield -i - 1, f"mtp.layers.{i}.transformer_layer.self_attention", self_attn

    def _iter_core_attentions(self):
        """Yield (param_name_prefix, core_attention_module) for every transformer / MTP layer."""
        for _, prefix, self_attn in self._iter_self_attentions():
            yield f"{prefix}.core_attention", self_attn.core_attention

    def set_qk_clip_reduce_group(self, group):
        """Register the comm group for the qk_clip max-logit all-reduce.

        ``max_logits_val`` is a per-(layer, head) running max that only needs to be
        max-reduced over the domain where the SAME weight sees DIFFERENT data — the
        data + context-parallel domain (``dp x cp``, i.e. the ``loss_mesh`` group).
        tp / pp hold different heads / layers (handled by ``full_tensor()`` and by
        each stage owning only its layers' params), so they must NOT be folded into
        this reduce. Set by ``parallelize_gptmodel``; ``None`` falls back to a world
        all-reduce so unparallelized / legacy call paths keep working.
        """
        self._qk_clip_reduce_group = group

    def _all_reduce_max_logits(self, tensor):
        """AllReduce-Max ``tensor`` over the dp x cp (``loss_mesh``) group when one
        has been registered, else over the world group (legacy fallback)."""
        group = getattr(self, "_qk_clip_reduce_group", None)
        if group is not None:
            result = all_reduce(tensor, op=ops.ReduceOp.MAX, group=group)
            return result if result is not None else tensor
        if get_world_size() > 1:
            result = all_reduce(tensor, op=ops.ReduceOp.MAX)
            return result if result is not None else tensor
        return tensor

    def _stacked_synced_max_logits(self):
        """Allreduce every layer's ``max_logits_val`` in one batched op. Returns
        ``(params, mesh, placements, stacked)`` (stacked shape: ``(num_layers,
        num_heads)``), or all-``None`` when no layer has ``max_logits_val``."""
        params = []
        for _, core_attn in self._iter_core_attentions():
            if hasattr(core_attn, "max_logits_val"):
                params.append(core_attn.max_logits_val)
        if not params:
            return None, None, None, None

        first = params[0]
        if isinstance(first, DTensor):
            mesh = first.device_mesh
            placements = first.placements
            full_locals = [p.full_tensor() for p in params]
        else:
            mesh = None
            placements = None
            full_locals = list(params)

        stacked = mint.stack(full_locals, dim=0)
        stacked = self._all_reduce_max_logits(stacked)
        return params, mesh, placements, stacked

    def _writeback_synced_max_logits(self, params, mesh, placements, stacked):
        """Write the per-row synced max back into each layer's ``max_logits_val``."""
        with SkipDTensorDispatch():
            for i, p in enumerate(params):
                row = stacked[i]
                if mesh is not None:
                    row = distribute_tensor(row, mesh, placements).to_local()
                inplace_copy(p, row)

    def allreduce_max_attention_logit(self):
        """AllReduce-Max of every layer's ``max_logits_val`` in a single batched
        collective. Assumes every layer's parameter shares the same DTensor
        layout — true here, they're all per-layer counters constructed identically."""
        params, mesh, placements, stacked = self._stacked_synced_max_logits()
        if params is None:
            return
        self._writeback_synced_max_logits(params, mesh, placements, stacked)

    def synced_max_attention_logit_fires(self, logit_threshold):
        """Combined qk_clip threshold check + sync in one batched AllReduce-Max.
        Writes the synced values back only on fire so non-fire steps stay
        bit-identical with the pre-batching behavior."""
        params, mesh, placements, stacked = self._stacked_synced_max_logits()
        if params is None:
            return False
        global_max = mint.max(stacked).reshape((1,))
        fires = bool(mint.greater_equal(global_max, logit_threshold).asnumpy()[0])
        if fires:
            self._writeback_synced_max_logits(params, mesh, placements, stacked)
        return fires

    def has_qk_clip_candidates(self, logit_threshold):
        """Whether any local/global max attention logit reaches QK-clip threshold.
        Stacks every layer's ``max_logits_val`` and runs a single ``mint.max``
        instead of 24 chained per-layer ``mint.max`` / ``mint.maximum`` ops."""
        locals_list = []
        for _, core_attn in self._iter_core_attentions():
            if not hasattr(core_attn, "max_logits_val"):
                continue
            param = core_attn.max_logits_val
            local_param = param.to_local() if isinstance(param, DTensor) else param
            locals_list.append(local_param.reshape((-1,)))
        if not locals_list:
            return False

        stacked = mint.stack(locals_list, dim=0)
        local_max = mint.max(stacked).reshape((1,))
        local_max = self._all_reduce_max_logits(local_max)
        return bool(mint.greater_equal(local_max, logit_threshold).asnumpy()[0])

    def get_max_attention_logit(self):
        """Return {full_param_name: Tensor} for layers whose running max is non-zero."""
        max_logits = {}
        for prefix, core_attn in self._iter_core_attentions():
            if not hasattr(core_attn, "max_logits_val"):
                continue
            param = core_attn.max_logits_val
            local_param = param.to_local() if isinstance(param, DTensor) else param
            if mint.sum(mint.abs(local_param)) <= 0:
                continue
            max_logits[f"{prefix}.max_logits_val"] = param
        return max_logits

    def reset_max_attention_logit(self):
        """Reset every per-layer max_logits_val to zeros."""
        for _, core_attn in self._iter_core_attentions():
            if hasattr(core_attn, "max_logits_val"):
                core_attn.reset_parameter()

    def _preprocess_input_labels_and_masks(self,
                                           input_ids: Tensor,
                                           labels: Tensor = None,
                                           attention_mask: Tensor = None,
                                           loss_mask: Tensor = None):
        """Preprocess input_ids and generate labels and masks if they are None.
        """
        if loss_mask is None:
            loss_mask = self.cast(self.not_equal(input_ids, self.pad_token_id), dtype.float32)
        if labels is not None:
            label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), dtype.float32)
            loss_mask = self.mul(loss_mask, label_mask)
        if self.use_attn_mask_compression:
            attention_mask = self.casual_mask()
        elif attention_mask is None:
            attention_mask = self.casual_mask(input_ids)
        return labels, attention_mask, loss_mask

    def get_gpt_transformer_config(self):
        """Get the transformer config for GPT model.

        Returns:
            TransformerConfig: The transformer configuration.
        """
        return self.config

    def make_model_muon_fns(self):
        """Read values from TransformersConfig and generate schema."""
        num_moe_experts = self.config.num_moe_experts
        hidden_size = self.config.hidden_size
        moe_ffn_hidden_size = self.config.moe_ffn_hidden_size
        qk_head_dim = self.config.qk_head_dim
        qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim
        num_attention_heads = self.config.num_attention_heads
        kv_lora_rank = self.config.kv_lora_rank
        value_head_dim = self.config.v_head_dim

        q_lora_rank = self.config.q_lora_rank

        # linear_qkv (concat MLA) is [q_rank | kv_lora_rank | qk_pos_emb_head_dim] along dim=0.
        # With LoRA    : q_rank = q_lora_rank → 3 plain slabs.
        # Without LoRA : q_rank = n_heads * (qk_head_dim + qk_pos_emb_head_dim) → 1 periodic + 2 slabs.
        if q_lora_rank is not None:
            qkv_blocks = (q_lora_rank, kv_lora_rank, qk_pos_emb_head_dim)
        else:
            qkv_blocks = (
                (qk_head_dim, qk_pos_emb_head_dim, num_attention_heads),
                kv_lora_rank,
                qk_pos_emb_head_dim,
            )

        schema = [
            # experts.weight1: reshape → split into two [num_moe_experts, hidden_size, moe_ffn_hidden_size]
            {
                "patterns": ["*mlp.experts.weight1*"],
                "kind": "reshape_concat",
                "reshape": (num_moe_experts, hidden_size, 2 * moe_ffn_hidden_size),
            },
            # experts.weight2: reshape → [num_moe_experts, moe_ffn_hidden_size, hidden_size]
            {
                "patterns": ["*mlp.experts.weight2*"],
                "kind": "reshape_only",
                "reshape": (num_moe_experts, moe_ffn_hidden_size, hidden_size),
            },
            # linear_qb (concat MLA = q_up_proj): periodic split across heads
            {
                "patterns": ["*self_attention.linear_qb.weight*"],
                "kind": "periodic",
                "parts": (qk_head_dim, qk_pos_emb_head_dim, num_attention_heads),
            },
            # linear_kvb (concat MLA = kv_up_proj): periodic split across heads
            {
                "patterns": ["*self_attention.linear_kvb.weight*"],
                "kind": "periodic",
                "parts": (qk_head_dim, value_head_dim, num_attention_heads),
            },
            # linear_qkv (concat MLA): hybrid slabs along dim=0.
            # LoRA case    : 3 plain slabs  → [q_down, kv_lora, k_pe]
            # no-LoRA case : 1 periodic + 2 slabs → [q_nope, q_pe, kv_lora, k_pe]
            {
                "patterns": ["*self_attention.linear_qkv.weight*"],
                "kind": "block_split",
                "blocks": qkv_blocks,
            },
            # fc1 and shared_fc1: alternating 1,1 split along rows
            {
                "patterns": [
                    "*mlp.shared_experts.linear_fc1.weight*",
                    "*mlp.linear_fc1.weight*",
                ],
                "kind": "periodic",
                "parts": (1, 1),
            },
        ]

        return make_muon_fns(schema)

    def get_param_layer_indices(self, params):
        """Return layer indices for each parameter (used for QK-clip).

        Args:
            params: List of parameters from the optimizer.

        Returns:
            Tuple of layer indices for each parameter, where:
                - layer_idx >= 0 stands for the layer_idx-th decoder layer
                - layer_idx < 0 stands for the -(layer_idx+1)-th MTP layer
        """
        param_layer = []
        for param in params:
            name = param.name
            try:
                layer_idx = int(name.split(".")[2])
            except (ValueError, IndexError):
                layer_idx = 0
            if name.startswith('mtp'):
                layer_idx = -layer_idx - 1
            param_layer.append(layer_idx)
        return tuple(param_layer)

    def apply_qk_clip_scaling(self, logit_threshold, muon_split_fn, muon_merge_fn, fp32_param_map=None):
        """Apply QK-clip scaling to attention weight parameters.

        We collect every layer's mask/scale ops on-device first, then do a
        *single* ``asnumpy()`` to read back per-layer ``num_clipped`` counts.
        Previously each layer issued its own ``int(...asnumpy())`` which forces
        a host↔device sync — 24 syncs per fire-step in 24L models.  Stacking
        them collapses those into one sync and lets the per-layer ``mint.sum``
        ops pipeline on the device.

        The else-branch (``can_clip_local`` False) still falls back to per-layer
        full_tensor / asnumpy because the global path is much rarer (TP-induced
        sharding-of-head-dim) and has a debug logger that depends on per-layer
        values.
        """
        if not self.config.multi_latent_attention:
            return

        ones = mint.ones((1,), dtype=dtype.float32)

        # Phase A: per-layer device work (mask + scales for the local-clip case),
        # collect everything but defer the host sync.
        local_pending = []  # list of (param_prefix, self_attention, mask_local, scales_local)
        global_pending = []  # list of (layer_idx, param_prefix, self_attention, logits_row)

        for layer_idx, param_prefix, self_attention in self._iter_self_attentions():
            logits_row = self_attention.core_attention.max_logits_val.value()
            logits_row = logits_row.reshape((-1,))
            logits_local = logits_row.to_local() if isinstance(logits_row, DTensor) else logits_row
            can_clip_local = self_attention.can_apply_qk_clip_to_local_weights(logits_local)
            if can_clip_local:
                mask_local = mint.greater_equal(logits_local, logit_threshold)
                safe_den_local = mint.where(mask_local, logits_local, ones)
                scales_local = mint.where(mask_local, logit_threshold / safe_den_local, ones)
                local_pending.append((param_prefix, self_attention, mask_local, scales_local))
            else:
                global_pending.append((layer_idx, param_prefix, self_attention, logits_row))

        # Phase B: batched ``num_clipped`` for the local-clip layers.  One stack
        # + one sum + one asnumpy gives an array of K per-layer counts instead
        # of K separate host syncs.
        if local_pending:
            stacked_masks = mint.stack(
                [mask for _, _, mask, _ in local_pending], dim=0)
            counts = mint.sum(stacked_masks.astype(dtype.int32), dim=1).asnumpy()
            for i, (param_prefix, self_attention, _, scales_local) in enumerate(local_pending):
                if int(counts[i]) == 0:
                    continue
                self_attention.try_apply_qk_clip_to_local_weights(
                    param_prefix, scales_local, muon_split_fn, muon_merge_fn,
                    fp32_param_map=fp32_param_map)

        # Phase C: global-clip fallback for layers where local weights can't
        # apply the clip (e.g. TP/FSDP shards the head dim — the common case
        # under context/data parallelism). These need a per-layer full_tensor().
        for layer_idx, param_prefix, self_attention, logits_row in global_pending:
            logits_full = logits_row.full_tensor() if isinstance(logits_row, DTensor) else logits_row
            mask = mint.greater_equal(logits_full, logit_threshold)
            num_clipped = int(mint.sum(mask.astype(dtype.int32)).asnumpy())
            if num_clipped == 0:
                continue
            safe_den = mint.where(mask, logits_full, ones)
            scales = mint.where(mask, logit_threshold / safe_den, ones)
            max_logit = float(mint.max(logits_full).asnumpy())
            threshold_val = float(logit_threshold.asnumpy())
            logger.debug(
                f"[QK-Clip] layer_idx={layer_idx} {param_prefix}: "
                f"clipping {num_clipped}/{logits_full.shape[0]} heads, "
                f"max_logit={max_logit:.4f}, threshold={threshold_val:.4f}"
            )
            self_attention.apply_qk_clip_to_weights(
                param_prefix, scales, muon_split_fn, muon_merge_fn,
                fp32_param_map=fp32_param_map)
