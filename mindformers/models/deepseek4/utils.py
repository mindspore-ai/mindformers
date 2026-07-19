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
"""DeepSeek4 models' utils."""
from mindformers.checkpoint.converter.convert_op import RenameConvertOp, ConcatConvertOp, ExpertsConvertOp, ScaleSplitConvertOp
from mindformers.models.deepseek3.utils import DeepseekV3PreTrainedModel
from mindformers.models.deepseek4.configuration_deepseek_v4 import DeepseekV4Config


class DeepseekV4PreTrainedModel(DeepseekV3PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DeepseekV4Config
    base_model_prefix = "Deepseekv4"

    # Override weight_mapping from DeepseekV3PreTrainedModel for V4-specific differences:
    # 1. V4 uses native checkpoint naming (layers.{}.attn.*) instead of HF naming (model.layers.{}.self_attn.*)
    # 2. V4: single linear_kv_proj (wkv) vs V3's linear_kv_down_proj + linear_kv_up_proj
    # 3. V4: wo_a/wo_b (low-rank O proj) vs V3's single o_proj
    # 4. V4: w1/w3/w2 naming for FFN experts vs V3's gate_proj/up_proj/down_proj
    # 5. V4: CSA compressor, DSA indexer, attn_sink (new components)
    weight_mapping = [
        # ========== Top-level ==========
        ('embed.', 'embedding.word_embeddings.'),
        ('norm.', 'decoder.final_layernorm.'),
        ('head.', 'output_layer.'),
        # ========== MLA Attention - Q path ==========
        ('.attn.wq_a.', '.self_attention.linear_q_down_proj.'),
        ('.attn.q_norm.', '.self_attention.q_layernorm.'),
        ('.attn.wq_b.', '.self_attention.linear_q_up_proj.'),
        # ========== MLA Attention - KV path (V4: single wkv projection) ==========
        ('.attn.wkv.', '.self_attention.linear_kv_proj.'),
        ('.attn.kv_norm.', '.self_attention.kv_layernorm.'),
        # ========== MLA Attention - O path (V4: wo_a/wo_b low-rank) ==========
        ('.attn.wo_a.', '.self_attention.linear_o_group_proj.'),
        ('.attn.wo_b.', '.self_attention.linear_proj.'),
        # ========== CSA Attn Sink ==========
        ('.attn.attn_sink', '.self_attention.core_attention.attn_sink'),
        # ========== CSA Compressor ==========
        ('.attn.compressor.wkv.', '.self_attention.core_attention.compressor.linear_wkv.'),
        ('.attn.compressor.wgate.', '.self_attention.core_attention.compressor.linear_wgate.'),
        ('.attn.compressor.norm.', '.self_attention.core_attention.compressor.norm.'),
        ('.attn.compressor.ape', '.self_attention.core_attention.compressor.ape'),
        # ========== DSA Indexer (even layers) ==========
        ('.attn.indexer.wq_b.', '.self_attention.core_attention.indexer.linear_wq_b.'),
        ('.attn.indexer.weights_proj.', '.self_attention.core_attention.indexer.linear_weights_proj.'),
        ('.attn.indexer.compressor.wkv.', '.self_attention.core_attention.indexer.compressor.linear_wkv.'),
        ('.attn.indexer.compressor.wgate.', '.self_attention.core_attention.indexer.compressor.linear_wgate.'),
        ('.attn.indexer.compressor.norm.', '.self_attention.core_attention.indexer.compressor.norm.'),
        ('.attn.indexer.compressor.ape', '.self_attention.core_attention.indexer.compressor.ape'),
        # ========== MoE Router ==========
        ('.ffn.gate.weight', '.mlp.router.weight'),
        ('.ffn.gate.bias', '.mlp.expert_bias'),
        ('.ffn.gate.tid2eid', '.mlp.router.tid2eid'),
        # ========== MoE Shared Experts (w1→gate, w3→up, w2→down) ==========
        ('.ffn.shared_experts.w1.', '.mlp.shared_experts.gating.'),
        ('.ffn.shared_experts.w3.', '.mlp.shared_experts.hidden.'),
        ('.ffn.shared_experts.w2.', '.mlp.shared_experts.linear_fc2.'),
        # ========== Experts generic mappings (w1→gate, w3→up, w2→down) ==========
        ('.experts.{}.w1.', '.experts.{}.gating.'),
        ('.experts.{}.w3.', '.experts.{}.hidden.'),
        ('.experts.{}.w2.', '.experts.{}.linear_fc2.'),
        # ========== Layer Norm ==========
        ('.attn_norm.', '.input_layernorm.'),
        ('.ffn_norm.', '.pre_mlp_layernorm.'),
        # ========== MTP layers (mtp.0) ==========
        ('mtp.0.enorm.', 'mtp.layers.0.enorm.'),
        ('mtp.0.hnorm.', 'mtp.layers.0.hnorm.'),
        ('mtp.0.norm.', 'mtp.layers.0.final_layernorm.'),
        ('mtp.0.e_proj.', 'mtp.layers.0.eh_proj.'),
        ('mtp.0.h_proj.', 'mtp.layers.0.eh_proj.'),
        # ========== HyperConnection Head (hc_head) - MTP (before mtp.0 catch-all) ==========
        ('mtp.0.hc_head_base', 'mtp.layers.0.hc_head.hc_base'),
        ('mtp.0.hc_head_fn', 'mtp.layers.0.hc_head.hc_fn.weight'),
        ('mtp.0.hc_head_scale', 'mtp.layers.0.hc_head.hc_scale'),
        ('mtp.0.', 'mtp.layers.0.transformer_layer.'),
        # ========== HyperConnection (HC) mappings ==========
        ('.hc_attn_base', '.attn_hc.bias'),
        ('.hc_attn_fn', '.attn_hc.mapping_proj.weight'),
        ('.hc_ffn_base', '.ffn_hc.bias'),
        ('.hc_ffn_fn', '.ffn_hc.mapping_proj.weight'),
        # ========== HyperConnection Head (hc_head) - Top-level decoder ==========
        ('hc_head_base', 'decoder.hc_head.hc_base'),
        ('hc_head_fn', 'decoder.hc_head.hc_fn.weight'),
        ('hc_head_scale', 'decoder.hc_head.hc_scale'),
        # ========== Transformer layers (keep last - lowest priority) ==========
        ('layers.', 'decoder.layers.')
    ]

    weight_converters = [
        # ========== Top-level Embedding and Output ==========
        RenameConvertOp(
            hf_names="embed.weight",
            mf_names="embedding.word_embeddings.weight"
        ),
        RenameConvertOp(
            hf_names="head.weight",
            mf_names="output_layer.weight"
        ),
        RenameConvertOp(
            hf_names="norm.weight",
            mf_names="decoder.final_layernorm.weight"
        ),
        # ========== MLA Attention - Q path ==========
        RenameConvertOp(
            hf_names="layers.{}.attn.wq_a.weight",
            mf_names="decoder.layers.{}.self_attention.linear_q_down_proj.weight"
        ),
        RenameConvertOp(
            hf_names="layers.{}.attn.q_norm.weight",
            mf_names="decoder.layers.{}.self_attention.q_layernorm.weight"
        ),
        RenameConvertOp(
            hf_names="layers.{}.attn.wq_b.weight",
            mf_names="decoder.layers.{}.self_attention.linear_q_up_proj.weight"
        ),
        # ========== MLA Attention - KV path (V4: single linear_kv_proj) ==========
        RenameConvertOp(
            hf_names="layers.{}.attn.wkv.weight",
            mf_names="decoder.layers.{}.self_attention.linear_kv_proj.weight"
        ),
        RenameConvertOp(
            hf_names="layers.{}.attn.kv_norm.weight",
            mf_names="decoder.layers.{}.self_attention.kv_layernorm.weight"
        ),
        # ========== MLA Attention - O path (V4: wo_a/wo_b low-rank) ==========
        RenameConvertOp(
            hf_names="layers.{}.attn.wo_a.weight",
            mf_names="decoder.layers.{}.self_attention.linear_o_group_proj"
        ),
        RenameConvertOp(
            hf_names="layers.{}.attn.wo_b.weight",
            mf_names="decoder.layers.{}.self_attention.linear_proj.weight"
        ),
        # ========== CSA Attn Sink ==========
        RenameConvertOp(
            hf_names="layers.{}.attn.attn_sink",
            mf_names="decoder.layers.{}.self_attention.core_attention.attn_sink"
        ),
        # ========== CSA Compressor ==========
        RenameConvertOp(
            hf_names="layers.{}.attn.compressor.ape",
            mf_names="decoder.layers.{}.self_attention.core_attention.compressor.ape"
        ),
        RenameConvertOp(
            hf_names="layers.{}.attn.compressor.wkv.weight",
            mf_names="decoder.layers.{}.self_attention.core_attention.compressor.linear_wkv.weight"
        ),
        RenameConvertOp(
            hf_names="layers.{}.attn.compressor.wgate.weight",
            mf_names="decoder.layers.{}.self_attention.core_attention.compressor.linear_wgate.weight"
        ),
        RenameConvertOp(
            hf_names="layers.{}.attn.compressor.norm.weight",
            mf_names="decoder.layers.{}.self_attention.core_attention.compressor.norm.weight"
        ),
        # ========== DSA Indexer (even layers) ==========
        RenameConvertOp(
            hf_names="layers.{}.attn.indexer.wq_b.weight",
            mf_names="decoder.layers.{}.self_attention.core_attention.indexer.linear_wq_b.weight"
        ),
        RenameConvertOp(
            hf_names="layers.{}.attn.indexer.weights_proj.weight",
            mf_names="decoder.layers.{}.self_attention.core_attention.indexer.linear_weights_proj.weight"
        ),
        RenameConvertOp(
            hf_names="layers.{}.attn.indexer.compressor.ape",
            mf_names="decoder.layers.{}.self_attention.core_attention.indexer.compressor.ape"
        ),
        RenameConvertOp(
            hf_names="layers.{}.attn.indexer.compressor.wkv.weight",
            mf_names="decoder.layers.{}.self_attention.core_attention.indexer.compressor.linear_wkv.weight"
        ),
        RenameConvertOp(
            hf_names="layers.{}.attn.indexer.compressor.wgate.weight",
            mf_names="decoder.layers.{}.self_attention.core_attention.indexer.compressor.linear_wgate.weight"
        ),
        RenameConvertOp(
            hf_names="layers.{}.attn.indexer.compressor.norm.weight",
            mf_names="decoder.layers.{}.self_attention.core_attention.indexer.compressor.norm.weight"
        ),
        # ========== Layer Norm ==========
        RenameConvertOp(
            hf_names="layers.{}.attn_norm.weight",
            mf_names="decoder.layers.{}.input_layernorm.weight"
        ),
        RenameConvertOp(
            hf_names="layers.{}.ffn_norm.weight",
            mf_names="decoder.layers.{}.pre_mlp_layernorm.weight"
        ),
        # ========== MoE Router ==========
        RenameConvertOp(
            hf_names="layers.{}.ffn.gate.weight",
            mf_names="decoder.layers.{}.mlp.router.weight"
        ),
        RenameConvertOp(
            hf_names="layers.{}.ffn.gate.bias",
            mf_names="decoder.layers.{}.mlp.expert_bias"
        ),
        RenameConvertOp(
            hf_names="layers.{}.ffn.gate.tid2eid",
            mf_names="decoder.layers.{}.mlp.router.tid2eid"
        ),
        # ========== MoE Shared Experts (w1/w3 concat → linear_fc1, w2 → linear_fc2) ==========
        ConcatConvertOp(
            hf_names=[
                "layers.{}.ffn.shared_experts.w1.weight",
                "layers.{}.ffn.shared_experts.w3.weight",
            ],
            mf_names=["decoder.layers.{}.mlp.shared_experts.linear_fc1.weight"],
            dim=0
        ),
        RenameConvertOp(
            hf_names="layers.{}.ffn.shared_experts.w2.weight",
            mf_names="decoder.layers.{}.mlp.shared_experts.linear_fc2.weight"
        ),
        # ========== MoE Routed Experts ==========
        # w1 (gate) + w3 (up) → experts.weight1 (stacked + interleaved)
        ExpertsConvertOp(
            hf_names=[
                "layers.{}.ffn.experts.{}.w1.weight",
                "layers.{}.ffn.experts.{}.w3.weight",
            ],
            mf_names=[
                "decoder.layers.{}.mlp.experts.weight1",
            ],
        ),
        # w2 (down) → experts.weight2 (stacked)
        ExpertsConvertOp(
            hf_names=[
                "layers.{}.ffn.experts.{}.w2.weight"
            ],
            mf_names=[
                "decoder.layers.{}.mlp.experts.weight2"
            ],
        ),
        # ========== MTP Layers ==========
        RenameConvertOp(
            hf_names="mtp.0.enorm.weight",
            mf_names="mtp.layers.0.enorm.weight"
        ),
        RenameConvertOp(
            hf_names="mtp.0.hnorm.weight",
            mf_names="mtp.layers.0.hnorm.weight"
        ),
        RenameConvertOp(
            hf_names="mtp.0.norm.weight",
            mf_names="mtp.layers.0.final_layernorm.weight"
        ),
        # e_proj + h_proj concat → eh_proj
        ConcatConvertOp(
            hf_names=[
                "mtp.0.e_proj.weight",
                "mtp.0.h_proj.weight",
            ],
            mf_names=["mtp.layers.0.eh_proj.weight"],
            dim=1,
            interleaved=False,
        ),
        # ========== MTP Attention layers (same structure as main layers, under mtp.0 prefix) ==========
        RenameConvertOp(
            hf_names="mtp.0.attn.wq_a.weight",
            mf_names="mtp.layers.0.transformer_layer.self_attention.linear_q_down_proj.weight"
        ),
        RenameConvertOp(
            hf_names="mtp.0.attn.q_norm.weight",
            mf_names="mtp.layers.0.transformer_layer.self_attention.q_layernorm.weight"
        ),
        RenameConvertOp(
            hf_names="mtp.0.attn.wq_b.weight",
            mf_names="mtp.layers.0.transformer_layer.self_attention.linear_q_up_proj.weight"
        ),
        RenameConvertOp(
            hf_names="mtp.0.attn.wkv.weight",
            mf_names="mtp.layers.0.transformer_layer.self_attention.linear_kv_proj.weight"
        ),
        RenameConvertOp(
            hf_names="mtp.0.attn.kv_norm.weight",
            mf_names="mtp.layers.0.transformer_layer.self_attention.kv_layernorm.weight"
        ),
        RenameConvertOp(
            hf_names="mtp.0.attn.wo_a.weight",
            mf_names="mtp.layers.0.transformer_layer.self_attention.linear_o_group_proj"
        ),
        RenameConvertOp(
            hf_names="mtp.0.attn.wo_b.weight",
            mf_names="mtp.layers.0.transformer_layer.self_attention.linear_proj.weight"
        ),
        RenameConvertOp(
            hf_names="mtp.0.attn.attn_sink",
            mf_names="mtp.layers.0.transformer_layer.self_attention.core_attention.attn_sink"
        ),
        # Note: MTP layer (mtp.0) does NOT have compressor or indexer in checkpoint
        # MTP Layer Norm
        RenameConvertOp(
            hf_names="mtp.0.attn_norm.weight",
            mf_names="mtp.layers.0.transformer_layer.input_layernorm.weight"
        ),
        RenameConvertOp(
            hf_names="mtp.0.ffn_norm.weight",
            mf_names="mtp.layers.0.transformer_layer.pre_mlp_layernorm.weight"
        ),
        # ========== MTP MoE Router ==========
        RenameConvertOp(
            hf_names="mtp.0.ffn.gate.weight",
            mf_names="mtp.layers.0.transformer_layer.mlp.router.weight"
        ),
        RenameConvertOp(
            hf_names="mtp.0.ffn.gate.bias",
            mf_names="mtp.layers.0.transformer_layer.mlp.expert_bias"
        ),
        # Note: mtp.0 has no ffn.gate.tid2eid in checkpoint
        # ========== MTP MoE Shared Experts ==========
        ConcatConvertOp(
            hf_names=[
                "mtp.0.ffn.shared_experts.w1.weight",
                "mtp.0.ffn.shared_experts.w3.weight",
            ],
            mf_names=["mtp.layers.0.transformer_layer.mlp.shared_experts.linear_fc1.weight"],
            dim=0
        ),
        RenameConvertOp(
            hf_names="mtp.0.ffn.shared_experts.w2.weight",
            mf_names="mtp.layers.0.transformer_layer.mlp.shared_experts.linear_fc2.weight"
        ),
        # ========== MTP MoE Routed Experts ==========
        ExpertsConvertOp(
            hf_names=[
                "mtp.{}.ffn.experts.{}.w1.weight",
                "mtp.{}.ffn.experts.{}.w3.weight",
            ],
            mf_names=["mtp.layers.{}.transformer_layer.mlp.experts.weight1"],
        ),
        ExpertsConvertOp(
            hf_names=[
                "mtp.{}.ffn.experts.{}.w2.weight"
            ],
            mf_names=["mtp.layers.{}.transformer_layer.mlp.experts.weight2"],
        ),
        # ========== HyperConnection (HC) for decoder layers ==========
        # HF: layers.{}.hc_attn_base [dim]  → MF: decoder.layers.{}.attn_hc.bias
        RenameConvertOp(
            hf_names="layers.{}.hc_attn_base",
            mf_names="decoder.layers.{}.attn_hc.bias"
        ),
        # HF: layers.{}.hc_attn_fn [dim, n*H]  → MF: decoder.layers.{}.attn_hc.mapping_proj.weight
        RenameConvertOp(
            hf_names="layers.{}.hc_attn_fn",
            mf_names="decoder.layers.{}.attn_hc.mapping_proj.weight"
        ),
        # HF: layers.{}.hc_attn_scale [3]  → MF: alpha_pre[1], alpha_post[1], alpha_res[1]
        ScaleSplitConvertOp(
            hf_names="layers.{}.hc_attn_scale",
            mf_names=[
                "decoder.layers.{}.attn_hc.alpha_pre",
                "decoder.layers.{}.attn_hc.alpha_post",
                "decoder.layers.{}.attn_hc.alpha_res",
            ],
        ),
        # HF: layers.{}.hc_ffn_base [dim]  → MF: decoder.layers.{}.ffn_hc.bias
        RenameConvertOp(
            hf_names="layers.{}.hc_ffn_base",
            mf_names="decoder.layers.{}.ffn_hc.bias"
        ),
        # HF: layers.{}.hc_ffn_fn [dim, n*H]  → MF: decoder.layers.{}.ffn_hc.mapping_proj.weight
        RenameConvertOp(
            hf_names="layers.{}.hc_ffn_fn",
            mf_names="decoder.layers.{}.ffn_hc.mapping_proj.weight"
        ),
        # HF: layers.{}.hc_ffn_scale [3]  → MF: alpha_pre[1], alpha_post[1], alpha_res[1]
        ScaleSplitConvertOp(
            hf_names="layers.{}.hc_ffn_scale",
            mf_names=[
                "decoder.layers.{}.ffn_hc.alpha_pre",
                "decoder.layers.{}.ffn_hc.alpha_post",
                "decoder.layers.{}.ffn_hc.alpha_res",
            ],
        ),
        # ========== HyperConnection (HC) for MTP layer ==========
        # HF: mtp.0.hc_attn_base  → MF: mtp.layers.0.transformer_layer.attn_hc.bias
        RenameConvertOp(
            hf_names="mtp.0.hc_attn_base",
            mf_names="mtp.layers.0.transformer_layer.attn_hc.bias"
        ),
        # HF: mtp.0.hc_attn_fn  → MF: mtp.layers.0.transformer_layer.attn_hc.mapping_proj.weight
        RenameConvertOp(
            hf_names="mtp.0.hc_attn_fn",
            mf_names="mtp.layers.0.transformer_layer.attn_hc.mapping_proj.weight"
        ),
        # HF: mtp.0.hc_attn_scale  → MF: alpha_pre, alpha_post, alpha_res
        ScaleSplitConvertOp(
            hf_names="mtp.0.hc_attn_scale",
            mf_names=[
                "mtp.layers.0.transformer_layer.attn_hc.alpha_pre",
                "mtp.layers.0.transformer_layer.attn_hc.alpha_post",
                "mtp.layers.0.transformer_layer.attn_hc.alpha_res",
            ],
        ),
        # HF: mtp.0.hc_ffn_base  → MF: mtp.layers.0.transformer_layer.ffn_hc.bias
        RenameConvertOp(
            hf_names="mtp.0.hc_ffn_base",
            mf_names="mtp.layers.0.transformer_layer.ffn_hc.bias"
        ),
        # HF: mtp.0.hc_ffn_fn  → MF: mtp.layers.0.transformer_layer.ffn_hc.mapping_proj.weight
        RenameConvertOp(
            hf_names="mtp.0.hc_ffn_fn",
            mf_names="mtp.layers.0.transformer_layer.ffn_hc.mapping_proj.weight"
        ),
        # HF: mtp.0.hc_ffn_scale  → MF: alpha_pre, alpha_post, alpha_res
        ScaleSplitConvertOp(
            hf_names="mtp.0.hc_ffn_scale",
            mf_names=[
                "mtp.layers.0.transformer_layer.ffn_hc.alpha_pre",
                "mtp.layers.0.transformer_layer.ffn_hc.alpha_post",
                "mtp.layers.0.transformer_layer.ffn_hc.alpha_res",
            ],
        ),
        # ========== HyperConnection Head (hc_head) for decoder block ==========
        # HF: hc_head_base [n]  → MF: decoder.hc_head.hc_base
        RenameConvertOp(
            hf_names="hc_head_base",
            mf_names="decoder.hc_head.hc_base"
        ),
        # HF: hc_head_fn [n, n*H]  → MF: decoder.hc_head.hc_fn.weight
        RenameConvertOp(
            hf_names="hc_head_fn",
            mf_names="decoder.hc_head.hc_fn.weight"
        ),
        # HF: hc_head_scale [1]  → MF: decoder.hc_head.hc_scale
        RenameConvertOp(
            hf_names="hc_head_scale",
            mf_names="decoder.hc_head.hc_scale"
        ),
        # ========== HyperConnection Head (hc_head) for MTP layer ==========
        # HF: mtp.0.hc_head_base [n]  → MF: mtp.layers.0.hc_head.hc_base
        RenameConvertOp(
            hf_names="mtp.0.hc_head_base",
            mf_names="mtp.layers.0.hc_head.hc_base"
        ),
        # HF: mtp.0.hc_head_fn [n, n*H]  → MF: mtp.layers.0.hc_head.hc_fn.weight
        RenameConvertOp(
            hf_names="mtp.0.hc_head_fn",
            mf_names="mtp.layers.0.hc_head.hc_fn.weight"
        ),
        # HF: mtp.0.hc_head_scale [1]  → MF: mtp.layers.0.hc_head.hc_scale
        RenameConvertOp(
            hf_names="mtp.0.hc_head_scale",
            mf_names="mtp.layers.0.hc_head.hc_scale"
        ),
    ]

    def get_model_parameters(self, only_trainable=True):
        pass
