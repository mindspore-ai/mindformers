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
from mindformers.models.qwen3.configuration_qwen3 import Qwen3Config
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.parallel_core.utils.model_mixin import ModelMixin
from mindformers.checkpoint.converter.convert_op import (
    RenameConvertOp,
    ConcatConvertOp,
    QKVConvertOp,
)

class Qwen3PreTrainedModel(PreTrainedModel, ModelMixin):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Qwen3Config
    base_model_prefix = "Qwen3"

    weight_mapping = [
        ('model.embed_tokens.', 'embedding.word_embeddings.'),
        ('.self_attn.q_proj.', '.self_attention.linear_q.'),
        ('.self_attn.k_proj.', '.self_attention.linear_k.'),
        ('.self_attn.v_proj.', '.self_attention.linear_v.'),
        ('.self_attn.o_proj.', '.self_attention.linear_proj.'),
        ('.self_attn.q_norm.', '.self_attention.q_layernorm.'),
        ('.self_attn.k_norm.', '.self_attention.k_layernorm.'),
        ('.mlp.gate_proj.', '.mlp.gating.'),
        ('.mlp.down_proj.', '.mlp.linear_fc2.'),
        ('.mlp.up_proj.', '.mlp.hidden.'),
        ('.post_attention_layernorm.', '.pre_mlp_layernorm.'),
        ('model.norm.', 'decoder.final_layernorm.'),
        ('lm_head.', 'output_layer.'),
        ('model.layers.', 'decoder.layers.')
    ]

    weight_converters = [
        # ========== Embedding and Output ==========
        RenameConvertOp(
            hf_names="model.embed_tokens.weight",
            mf_names="embedding.word_embeddings.weight"
        ),
        RenameConvertOp(
            hf_names="lm_head.weight",
            mf_names="output_layer.weight"
        ),
        RenameConvertOp(
            hf_names="model.norm.weight",
            mf_names="decoder.final_layernorm.weight"
        ),

        # ========== Attention ==========
        QKVConvertOp(
            hf_names=[
                "model.layers.{}.self_attn.q_proj.weight",
                "model.layers.{}.self_attn.k_proj.weight",
                "model.layers.{}.self_attn.v_proj.weight"
            ],
            mf_names=["decoder.layers.{}.self_attention.linear_qkv.weight"]
        ),
        RenameConvertOp(
            hf_names="model.layers.{}.self_attn.o_proj.weight",
            mf_names="decoder.layers.{}.self_attention.linear_proj.weight"
        ),
        RenameConvertOp(
            hf_names="model.layers.{}.input_layernorm.weight",
            mf_names="decoder.layers.{}.input_layernorm.weight"
        ),
        RenameConvertOp(
            hf_names="model.layers.{}.self_attn.k_norm.weight",
            mf_names="decoder.layers.{}.self_attention.k_layernorm.weight"
        ),
        RenameConvertOp(
            hf_names="model.layers.{}.self_attn.q_norm.weight",
            mf_names="decoder.layers.{}.self_attention.q_layernorm.weight"
        ),

        # ========== FFN ==========
        ConcatConvertOp(
            hf_names=[
                "model.layers.{}.mlp.gate_proj.weight",
                "model.layers.{}.mlp.up_proj.weight"
            ],
            mf_names=["decoder.layers.{}.mlp.linear_fc1.weight"],
            dim=0
        ),
        RenameConvertOp(
            hf_names="model.layers.{}.mlp.down_proj.weight",
            mf_names="decoder.layers.{}.mlp.linear_fc2.weight"
        ),
        RenameConvertOp(
            hf_names="model.layers.{}.post_attention_layernorm.weight",
            mf_names="decoder.layers.{}.pre_mlp_layernorm.weight"
        ),
    ]
