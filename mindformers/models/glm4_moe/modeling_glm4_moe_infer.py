# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Glm4Moe models' APIs."""
__all__ = ['InferenceGlm4MoeForCausalLM']

from typing import Dict

from mindformers.models.utils import jit
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.transformer_config_utils import convert_to_transformer_config
from mindformers.models.glm4_moe.utils import Glm4MoePreTrainedModel
from mindformers.parallel_core.inference.utils import update_comm_config
from mindformers.parallel_core.inference.base_models.gpt.gpt_model import GPTModel
from mindformers.parallel_core.inference.base_models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from mindformers.parallel_core.inference.model_utils import InferModelMixin

from .configuration_glm4_moe import Glm4MoeConfig


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class InferenceGlm4MoeForCausalLM(Glm4MoePreTrainedModel, InferModelMixin):
    r"""
    Provide Glm4_moe model infer through network.

    Args:
        config (Glm4MoeConfig): The config of Glm4_moe model.

    Returns:
        output: Tensor, the output of Glm4_moe decoder layer

    """

    def __init__(self, config: Glm4MoeConfig):
        super().__init__(config, auto_prefix=False)
        self.config = config
        config: TransformerConfig = convert_to_transformer_config(self.config)

        # update communication-related configuration in TransformerConfig
        config = update_comm_config(config)
        self.pad_token_id = self.config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.compute_dtype = config.compute_dtype
        self.is_prefill = True
        if isinstance(self.config.parallel_decoding_params, Dict):
            self.plugin_type = self.config.parallel_decoding_params.get("plugin_type")
        else:
            self.plugin_type = None
        self.model = GPTModel(config=config,
                              transformer_layer_spec=get_gpt_decoder_block_spec(
                                  config=config,
                                  normalization=config.normalization,
                                  qk_l2_norm=False,
                              ),
                              vocab_size=self.vocab_size,
                              max_sequence_length=self.max_position_embeddings,
                              position_embedding_type=config.position_embedding_type,
                              rotary_percent=self.config.partial_rotary_factor,
                              rotary_base=self.config.rope_theta,
                              share_embeddings_and_output_weights=self.config.tie_word_embeddings,
                              pre_process=self.config.pre_process,
                              post_process=self.config.post_process,)

    @jit
    def construct(self, input_ids, hidden_states=None, positions=None, batch_valid_length=None,
                  context_lens_tensor=None, q_seq_lens=None, block_tables=None, slot_mapping=None, attention_mask=None,
                  attn_metadata=None, attn_padding_idx=None, attn_unpadding_idx=None, ffn_padding_idx=None,
                  ffn_unpadding_idx=None, key_cache=None, value_cache=None):
        r"""
        model forward.

        Args:
            input_ids: input ids.
            positions: position ids.
            hidden_states: hidden states.
            batch_valid_length: actual seq length.
            context_lens_tensor: computed key value length.
            q_seq_lens: query sequence lengths.
            block_tables: Store mapping tables for each sequence.
            slot_mapping : Token cache physical slot index.
            attention_mask: attention mask used for fa or pa.
            attn_metadata: attention metadata.
            attn_padding_idx: Indices mapping positions in attention output sequence to original token positions,
                used for padding attention output to fixed size.
            attn_unpadding_idx: Indices mapping valid tokens in padded attention output sequence to
                their original positions, used for removing padding in attention output.
            ffn_padding_idx: Indices mapping positions in MoE output sequence to flattened valid token positions,
                used for padding MoE output to fixed size.
            ffn_unpadding_idx: Indices mapping valid tokens in padded MoE output sequence to their original positions,
                used for removing padding in MoE output.
            key_cache: key cache for incremental inference.
            value_cache: value cache for incremental inference.

        Returns:
            logits: the output logits.

        """
        logits = self.model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            positions=positions,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            q_seq_lens=q_seq_lens,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            attention_mask=attention_mask,
            attn_metadata=attn_metadata,
            attn_padding_idx=attn_padding_idx,
            attn_unpadding_idx=attn_unpadding_idx,
            ffn_padding_idx=ffn_padding_idx,
            ffn_unpadding_idx=ffn_unpadding_idx,
            key_cache=key_cache,
            value_cache=value_cache
        )
        return logits
