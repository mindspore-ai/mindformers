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
"""Qwen3 models' utils."""
from mindformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.parallel_core.utils.model_mixin import ModelMixin


class Qwen3MoePreTrainedModel(PreTrainedModel, ModelMixin):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Qwen3MoeConfig
    base_model_prefix = "Qwen3Moe"

    weight_mapping = [
        ('model.embed_tokens.', 'embedding.word_embeddings.'),
        ('.self_attn.q_proj.', '.self_attention.linear_q.'),
        ('.self_attn.k_proj.', '.self_attention.linear_k.'),
        ('.self_attn.v_proj.', '.self_attention.linear_v.'),
        ('.self_attn.o_proj.', '.self_attention.linear_proj.'),
        ('.self_attn.q_norm.', '.self_attention.q_layernorm.'),
        ('.self_attn.k_norm.', '.self_attention.k_layernorm.'),
        ('.gate.weight', '.router.weight'),
        ('.gate_proj.', '.gating.'),
        ('.down_proj.', '.linear_fc2.'),
        ('.up_proj.', '.hidden.'),
        ('.post_attention_layernorm.', '.pre_mlp_layernorm.'),
        ('model.norm.', 'decoder.final_layernorm.'),
        ('lm_head.', 'output_layer.'),
        ('model.layers.', 'decoder.layers.')
    ]

    def get_model_parameters(self):
        pass
