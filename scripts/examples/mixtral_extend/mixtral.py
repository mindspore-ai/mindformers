# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Mixtral models' APIs."""
import copy

import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, nn
from mindspore import mint
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.utils import LayerSetting, check_fine_grain_interleave_valid
from mindformers.modules.layers import Linear, FreqsMgr
from mindformers.modules.transformer import LowerTriangularMaskWithDynamic
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.utils import get_predict_run_mode

from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.models.llama.llama_layer import LlamaEmbedding, LlamaRMSNorm
from mindformers.models.llama.llama_interleave import LLamaDecodeLayerInterleave
from mindformers.models.llama.llama import LlamaPreTrainedModel
from mindformers.models.utils import lazy_inline
from mindformers.tools.logger import logger

from .mixtral_transformer import MixtralDecodeLayer

__all__ = ['MixtralModel', 'MixtralForCausalLM']


class MixtralModel(LlamaPreTrainedModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MixtralDecoderLayer`]
    Args:
        config(LlamaConfig): the config of network

    Returns:
            output: Tensor, the output of llama decoderlayer

    Examples:
        >>> from mindformers import MixtralModel
        >>> network = MixtralModel.from_pretrained('llama_7b')
        >>> type(network)
        <class 'research.mixtral_extend.mixtral.MixtralModel'>
    """
    _support_list = MindFormerBook.get_model_support_list()['llama']

    def __init__(self,
                 config: LlamaConfig = None):
        super().__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.dtype = config.compute_dtype
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.n_head = config.num_heads
        self.head_dim = self.hidden_size // self.n_head
        self.pad_token_id = config.pad_token_id
        self.is_first_iteration = True
        self.use_past = config.use_past
        self.use_flash_attention = config.use_flash_attention
        self.use_ring_attention = config.use_ring_attention
        self.parallel_decoding = config.parallel_decoding_params is not None
        self.concat = P.Concat(-1)
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()

        if config.moe_config.expert_num > 1:
            logger.info("MoE config is provided, use MoE FFN")
        else:
            logger.info("MoE config is None, use normal FFN")
        if not self.use_flash_attention and self.use_ring_attention:
            raise ValueError(f"When the ring_attention = True, the flash_attention must be True ")
        self.freqs_mgr = FreqsMgr(head_dim=self.head_dim,
                                  seq_length=config.seq_length,
                                  max_position_embedding=config.max_position_embedding,
                                  rotary_dtype=config.rotary_dtype,
                                  theta=config.theta,
                                  scaling_factor=config.scaling_factor,
                                  extend_method=config.extend_method,
                                  parallel_config=config.parallel_config,
                                  is_dynamic=config.is_dynamic)
        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=config.use_flash_attention,
                                                          use_attn_mask_compression=config.use_attn_mask_compression,
                                                          use_past=config.use_past)
        self.tok_embeddings = LlamaEmbedding(vocab_table_size=config.vocab_size,
                                             embedding_size=config.hidden_size,
                                             param_init_type=config.embedding_init_type,
                                             parallel_optimizer=config.parallel_optimizer)
        self.fine_grain_interleave = check_fine_grain_interleave_valid(config.fine_grain_interleave,
                                                                       config.parallel_config)
        self.layers = nn.CellList()
        self.layer_setting = LayerSetting(config.num_layers,
                                          config.offset,
                                          config.parallel_config,
                                          config.pp_interleave_num)
        for layer_id in range(config.num_layers):
            if self.fine_grain_interleave:
                layer = LLamaDecodeLayerInterleave(config.batch_size,
                                                   config.seq_length,
                                                   layer_id,
                                                   dim=config.hidden_size,
                                                   n_heads=config.num_heads,
                                                   num_layers=config.num_layers,
                                                   multiple_of=config.multiple_of,
                                                   n_kv_heads=config.n_kv_heads,
                                                   intermediate_size=config.intermediate_size,
                                                   ffn_dim_multiplier=config.ffn_dim_multiplier,
                                                   norm_eps=config.rms_norm_eps,
                                                   qkv_has_bias=config.qkv_has_bias,
                                                   qkv_concat=config.qkv_concat,
                                                   compute_dtype=config.compute_dtype,
                                                   layernorm_compute_dtype=config.layernorm_compute_type,
                                                   softmax_compute_dtype=config.softmax_compute_type,
                                                   rotary_dtype=config.rotary_dtype,
                                                   param_init_type=config.param_init_type,
                                                   use_flash_attention=config.use_flash_attention,
                                                   use_ring_attention=config.use_ring_attention,
                                                   use_attn_mask_compression=config.use_attn_mask_compression,
                                                   fine_grain_interleave=config.fine_grain_interleave,
                                                   parallel_config=config.parallel_config)
            else:
                layer = MixtralDecodeLayer(layer_id,
                                           dim=config.hidden_size,
                                           n_heads=config.num_heads,
                                           n_kv_heads=config.n_kv_heads,
                                           intermediate_size=config.intermediate_size,
                                           multiple_of=config.multiple_of,
                                           ffn_dim_multiplier=config.ffn_dim_multiplier,
                                           norm_eps=config.rms_norm_eps,
                                           qkv_has_bias=config.qkv_has_bias,
                                           qkv_concat=config.qkv_concat,
                                           compute_dtype=config.compute_dtype,
                                           layernorm_compute_dtype=config.layernorm_compute_type,
                                           softmax_compute_dtype=config.softmax_compute_type,
                                           rotary_dtype=config.rotary_dtype,
                                           param_init_type=config.param_init_type,
                                           use_past=config.use_past,
                                           use_flash_attention=config.use_flash_attention,
                                           use_ring_attention=config.use_ring_attention,
                                           use_attn_mask_compression=config.use_attn_mask_compression,
                                           block_size=config.block_size,
                                           num_blocks=config.num_blocks,
                                           use_rope_slice=config.use_rope_slice,
                                           moe_config=config.moe_config,
                                           parallel_config=config.parallel_config,
                                           parallel_decoding=self.parallel_decoding)
            self.layer_setting(layer, layer_id)
            self.layers.append(layer)
        self.norm_out = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps,
                                     compute_type=config.layernorm_compute_type)
        dp = config.parallel_config.data_parallel
        cp = config.parallel_config.context_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.tok_embeddings.pipeline_stage = 0
            if config.parallel_config.pipeline_stage > 1:
                self.norm_out.pipeline_stage = config.parallel_config.pipeline_stage - 1
                self.tok_embeddings.set_comm_fusion(2)
                self.norm_out.set_comm_fusion(2)
            else:
                self.tok_embeddings.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
                self.norm_out.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

            self.tok_embeddings.shard(config.parallel_config)
            self.casual_mask.shard(config.parallel_config)
            self.concat.shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            if self.fine_grain_interleave:
                self.norm_out.shard((dp * cp, 1))
            else:
                self.norm_out.shard((dp, cp, 1))

    # pylint: disable=W0613
    def construct(self, tokens: Tensor, input_embeds=None, batch_valid_length=None, batch_index=None,
                  zactivate_len=None, block_tables=None, slot_mapping=None, prefix_keys_values=None,
                  attention_mask=None, position_ids=None, q_seq_lens=None):
        """
        Forward of mixtral model.

        Args:
            tokens: the tokenized inputs with datatype int32
            input_embeds: the embedding Tensor of tokens, Tensor of shape:math:`(batch_size, seq/_length, hidden_size)`.
                Default None.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            block_tables (Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping (Tensor[int32]): Store token cache physical slot index.
        Returns:
            output: Tensor, the output of mixtral decoderlayer
        """
        # preprocess
        bs, seq_len = self.shape(tokens)
        if self.parallel_decoding:
            mask = attention_mask
            if self.is_first_iteration:
                shape = mask.shape
                mask = mask.reshape(1, 1, shape[0], shape[1])
            freqs_cis = self.freqs_mgr.increment_multi_ids(position_ids)
        else:
            mask = None
            if self.use_past:
                if self.is_first_iteration:
                    freqs_cis = self.freqs_mgr.prefill(bs, seq_len)
                    mask = self.casual_mask.prefill()
                    if prefix_keys_values is not None:
                        if mask is None:
                            mask = self.casual_mask(tokens)
                        prefix_length = prefix_keys_values[0].shape[2]
                        prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                        mask = self.concat((prefix_mask, mask))
                else:
                    freqs_cis = self.freqs_mgr.increment(batch_valid_length)
            else:
                if not self.use_ring_attention:
                    mask = self.casual_mask(tokens)
                freqs_cis = self.freqs_mgr(seq_len)
                if prefix_keys_values is not None:
                    prefix_length = prefix_keys_values[0].shape[2]
                    prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                    mask = self.concat((prefix_mask, mask))

        # tokens: [bs, seq/1]
        if input_embeds is not None:
            h = self.cast(input_embeds, self.dtype)
        else:
            h = self.cast(self.tok_embeddings(tokens), self.dtype)
        h = self.reshape(h, (bs, seq_len, self.hidden_size))
        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            prefix_kv = prefix_keys_values[i] if prefix_keys_values is not None else None
            h = self.layers[i](h, freqs_cis, mask, batch_valid_length=batch_valid_length, block_tables=block_tables,
                               slot_mapping=slot_mapping, prefix_keys_values=prefix_kv, q_seq_lens=q_seq_lens)
        output = self.norm_out(h)
        return output


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class MixtralForCausalLM(LlamaPreTrainedModel):
    r"""
    Provide mixtral training loss or logits through network.

    Args:
        config (LlamaConfig): The config of llama model. Default: `None` .

    Inputs:
        - **input_ids** (Tensor) - the indices of input sequence tokens in the vocabulary with data type Int64/Int32,
          Tensor of shape :math:`(batch, seq\_length)`.
        - **labels** (Tensor, optional) - the labels of inputs with data type Int64/Int32, Tensor of
          shape :math:`(batch, seq\_length)` . Default: ``None`` .
        - **input_position** (Tensor, optional) - the position ids of inputs (at incremental reasoning mode) which is
          an increasing sequence with data type Int64/Int32, Tensor :math:`(batch, seq\_length)`.
          Default: ``None`` .
        - **position_ids** (Tensor, optional) - the position ids of inputs which is
          an increasing sequence with data type
          Int64/Int32, Tensor :math:`(batch, seq\_length)`. Default: ``None`` .
        - **attention_mask** (Tensor, optional) - input sentences padding mask, where 0 indicates padding position with
          data type Int64/Int32, Tensor of shape :math:`(batch, seq\_length)`. Default: ``None`` .
        - **input_embeds** (Tensor, optional) - the embedding of inputs with data type Float32/Float16, Tensor of
          shape :math:`(batch, seq\_length, hidden\_size)`. Default: ``None`` .
        - **init_reset** (Tensor, optional) - A Bool tensor with shape [1], used to clear the past key parameter and
          past value parameter used in the incremental prediction. Only valid when use_past is True.
          Tensor of shape :math:`(1)`. Default: ``Tensor([True])`` .
        - **batch_valid_length** (Tensor, optional) - Int32 tensor with shape [batch_size]
          the past calculated the index.
          Used for incremental prediction when the use_past is True. Default: ``None`` .
        - **block_tables** (Tensor, optional) - Int64 type Tensor, Store mapping tables for each sequence.
          Default: ``None`` .
        - **slot_mapping** (Tensor, optional) - Int32 type Tensor, token cache physical slot index. Default:``None`` .

    Outputs:
        Tensor. If it is in training mode, the output Tensor contains loss;
        If it is in prediction mode, the output Tensor contains logits;
        If it is in evaluation mode, the output Tensor contains logits, tokens, and input masks.
    """
    _support_list = MindFormerBook.get_model_support_list()['llama']

    @lazy_inline
    def __init__(self, config: LlamaConfig = None):
        super(MixtralForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.config = config
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ones = P.Ones()
        self.gather = P.Gather()
        self.sub_batch_valid_len = P.Sub()
        self.model = MixtralModel(config=config)
        self.lm_head = Linear(in_channels=config.hidden_size,
                              out_channels=config.vocab_size,
                              has_bias=False,
                              compute_dtype=config.compute_dtype,
                              param_init_type=config.param_init_type,
                              weight_init="normal")  # meta default: xavier_normal
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.tok_embeddings.embedding_weight

        mp = config.parallel_config.model_parallel
        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(config.parallel_config)
        if vocab_size % mp != 0:
            logger.warning("The vocab size of Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1
        loss_parallel_config.data_parallel *= loss_parallel_config.context_parallel
        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad)

        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        cp = config.parallel_config.context_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.slice.shard(((dp, 1),))
            self.not_equal.shard(((dp, 1), ()))
            self.mul.shard(((dp, 1), (dp, 1)))
            self.add.shard(((dp, 1), ()))
            self.gather.shard(((dp, 1, 1), (dp,)))
            self.sub_batch_valid_len.shard(((1,), ()))
            if config.parallel_config.vocab_emb_dp or (vocab_size % mp != 0):
                self.lm_head.shard(strategy_matmul=((dp * cp, 1), (1, 1)))
            else:
                self.lm_head.shard(strategy_matmul=((dp * cp, 1), (mp, 1)))
            if config.parallel_config.pipeline_stage > 1:
                self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.load_checkpoint(config)
        self.predict_run_mode = get_predict_run_mode()

        logger.info("Predict run mode:{}".format(self.predict_run_mode))

        llm_boost_kwargs = {"config": config}
        if config.llm_backend:
            from mindspore.experimental.llm_boost.register import LlmBoostRegister
            self.llm_boost = LlmBoostRegister.get_instance(config.llm_backend, "Llama", **llm_boost_kwargs)
            self.llm_boost.init()
            self.is_set_kvcache = False
        self.parallel_decoding = config.parallel_decoding_params is not None

    def to_embeddings(self, tokens):
        """return embedding tokens"""
        return self.model.tok_embeddings(tokens)

    # pylint: disable=W0613
    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get Llama model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs, seq = input_ids.shape[0], input_ids.shape[1]
        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        prefix_keys_values = Tensor(kwargs["prefix_keys_values"]) if "prefix_keys_values" in kwargs else None
        return input_ids, labels, None, None, None, None, None, None, None, None, None, slot_mapping, prefix_keys_values

    def set_dynamic_inputs(self, **kwargs):
        """set dynamic inputs"""
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        have_prefix_keys_values = getattr(kwargs, "have_prefix_keys_values", False)
        dynamic_position_ids = Tensor(shape=[None, None], dtype=mstype.int32) if self.parallel_decoding else None
        dynamic_mask = Tensor(shape=[None, None], dtype=mstype.float16) if self.parallel_decoding else None
        dynamic_q_seq_lens = Tensor(shape=[None], dtype=mstype.int32) if self.parallel_decoding else None
        if have_prefix_keys_values:
            dynamic_prefix_keys_values = Tensor(shape=[2, None, None, None, None], dtype=mstype.float16)
            self.set_inputs(dynamic_input_ids, None, None, dynamic_position_ids, dynamic_mask, None, None,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, dynamic_prefix_keys_values, None, dynamic_q_seq_lens)
        elif self.use_past:
            self.set_inputs(dynamic_input_ids, None, None, dynamic_position_ids, dynamic_mask, None, None,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, None, None, dynamic_q_seq_lens)
        elif kwargs.get("pre_gather", False):
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                            dynamic_batch_valid_length, None, None, None, None, None)
        else:
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                            None, None, None, None, None, None)
        logger.info("Set dynamic input for mixtral.")

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.infer_attention.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=None, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None, llm_boost_inputs=None,
                  q_seq_lens=None):
        r"""
        MixtralForCausalLM forward.

        Args:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            labels(Tensor): the tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            input_position(Tensor): current position, used by model.predict.
            position_ids(Tensor): Reserved param, not used.
            attention_mask(Tensor): Reserved param, not used.
            input_embeds(Tensor): the input embedding Tensor of shape :math:`(batch, seq\_length, hidden_size)`.
                Default None.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            block_tables (Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping (Tensor[int32]): Store token cache physical slot index.
            q_seq_lens (Tensor[int32]): In parallel decoding, the query may be flattened. The Paged Attention operator
                need `q_seq_lens` to obtain the length information.

        Returns:
            Tensor, The loss or (logits, tokens, input_mask) of the network.
        """
        if hasattr(self, 'llm_boost'):
            if not self.is_set_kvcache:
                self.llm_boost.set_kvcache()
                self.is_set_kvcache = True
            self.llm_boost.add_flags(is_first_iteration=self.is_first_iteration)
            llm_boost_inputs["cos_embed"] = self.model.freqs_mgr.freqs_cos
            llm_boost_inputs["sin_embed"] = self.model.freqs_mgr.freqs_sin
            return self.llm_boost.forward(llm_boost_inputs)

        bsz, seqlen = self.shape(input_ids)
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = input_ids
        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))

        output = self.model(tokens, input_embeds, batch_valid_length, batch_index, zactivate_len, block_tables, \
                            slot_mapping, prefix_keys_values, attention_mask, position_ids, q_seq_lens)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        pre_gather = pre_gather and not self.parallel_decoding
        if pre_gather:
            batch_valid_length = mint.cumsum(batch_valid_length, 0)
            output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        logits = self.lm_head(output)

        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
                input_mask = self.mul(input_mask, label_mask)

        if not self.training:
            logits = self.cast(logits, mstype.float32)
            if self.predict_run_mode:
                logits = self.reshape(logits, (-1, logits.shape[-1]))
                return logits
            return logits, tokens, input_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        logits = self.cast(logits, mstype.float32)
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss

    def kvcache(self, layer_idx):
        key_cache = self.model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.key_cache
        value_cache = self.model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.value_cache
        return key_cache, value_cache
