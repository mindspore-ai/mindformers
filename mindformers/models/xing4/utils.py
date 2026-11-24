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
"""Shared MLA, DSA, MoE, mHC and MTP weight rules for Xing4 models."""
from copy import copy

import numpy as np
from mindspore import float32

from mindformers.checkpoint.converter.convert_op import ConcatConvertOp, ExpertsConvertOp, RenameConvertOp, ScaleSplitConvertOp
from mindformers.checkpoint.converter.template import WeightTemplate
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.parallel_core.utils.model_mixin import ModelMixin
from mindformers.tools.logger import logger
from .configuration_xing4 import Xing4Config


def _transformer_config(config):
    """Accept model configuration objects at the model/export boundary."""
    if isinstance(config, dict):
        config = Xing4Config.from_dict(config)
    if isinstance(config, Xing4Config):
        return config.convert_to_transformer_config()
    return config


def _layer_converters(config, source, target, moe, hyper_connections, dsa=False):
    """Use concrete layer prefixes so MTP never matches a decoder-layer rule."""
    renames = [
        ("self_attn.q_b_proj.weight", "self_attention.linear_qb.weight"),
        ("self_attn.kv_b_proj.weight", "self_attention.linear_kvb.weight"),
        ("self_attn.kv_a_layernorm.weight", "self_attention.k_layernorm.weight"),
        ("self_attn.o_proj.weight", "self_attention.linear_proj.weight"),
        ("input_layernorm.weight", "input_layernorm.weight"),
        ("post_attention_layernorm.weight", "pre_mlp_layernorm.weight"),
    ]
    if config.q_lora_rank is not None:
        renames.append(("self_attn.q_a_layernorm.weight", "self_attention.q_layernorm.weight"))
    else:
        renames = [(hf, mf) for hf, mf in renames if "q_b_proj" not in hf]
    if dsa:
        renames.extend(("self_attn.indexer." + hf, "self_attention.core_attention.indexer." + mf)
                       for hf, mf in (
                           ("wq_b.weight", "linear_wq_b.weight"),
                           ("wk.weight", "linear_wk.weight"),
                           ("weights_proj.weight", "linear_weights_proj.weight"),
                           ("k_norm.weight", "k_norm.gamma"),
                           ("k_norm.bias", "k_norm.beta")))
    ops = [RenameConvertOp(source + hf, target + mf) for hf, mf in renames]
    query = "q_a_proj" if config.q_lora_rank is not None else "q_proj"
    query_rows = (config.q_lora_rank if config.q_lora_rank is not None
                  else config.num_attention_heads * (config.qk_head_dim + config.qk_pos_emb_head_dim))
    ops.append(ConcatConvertOp(
        [source + f"self_attn.{query}.weight", source + "self_attn.kv_a_proj_with_mqa.weight"],
        target + "self_attention.linear_qkv.weight", dim=0, interleaved=False,
        split_sizes=[query_rows, config.kv_lora_rank + config.qk_pos_emb_head_dim]))
    ffn = "mlp.shared_experts." if moe else "mlp."
    if not moe or config.shared_expert_num:
        ops.extend([
            ConcatConvertOp([source + ffn + "gate_proj.weight", source + ffn + "up_proj.weight"],
                            target + ffn + "linear_fc1.weight", dim=0),
            RenameConvertOp(source + ffn + "down_proj.weight", target + ffn + "linear_fc2.weight"),
        ])
    if moe:
        ops.extend([
            RenameConvertOp(source + "mlp.gate.weight", target + "mlp.router.weight"),
            RenameConvertOp(source + "mlp.gate.e_score_correction_bias", target + "mlp.expert_bias"),
            ExpertsConvertOp([source + "mlp.experts.{}.gate_proj.weight",
                              source + "mlp.experts.{}.up_proj.weight"], target + "mlp.experts.weight1"),
            ExpertsConvertOp(source + "mlp.experts.{}.down_proj.weight", target + "mlp.experts.weight2"),
        ])
    if hyper_connections:
        for part in ("attn_hc", "ffn_hc"):
            ops.extend([
                RenameConvertOp(source + part + ".hc_fn", target + part + ".mapping_proj.weight"),
                RenameConvertOp(source + part + ".hc_base", target + part + ".bias"),
                ScaleSplitConvertOp(source + part + ".hc_scale",
                                    [target + part + ".alpha_" + name for name in ("pre", "post", "res")]),
            ])
    return ops


def _weight_converters(config):
    """Build decoder and appended-MTP rules from the public HF architecture."""
    ops = [
        RenameConvertOp("model.embed_tokens.weight", "embedding.word_embeddings.weight"),
        RenameConvertOp("lm_head.weight", "output_layer.weight"),
        RenameConvertOp("model.norm.weight", "decoder.final_layernorm.weight"),
    ]
    for layer, moe in enumerate(config.moe_layer_freq):
        ops.extend(_layer_converters(config, f"model.layers.{layer}.", f"decoder.layers.{layer}.",
                                     bool(moe), config.enable_hyper_connections,
                                     config.experimental_attention_variant == "dsa"))
    for layer in range(config.mtp_num_layers or 0):
        source = f"model.layers.{config.num_layers + layer}."
        target = f"mtp.layers.{layer}."
        ops.extend(RenameConvertOp(source + hf, target + mf) for hf, mf in (
            ("enorm.weight", "enorm.weight"), ("hnorm.weight", "hnorm.weight"),
            ("eh_proj.weight", "eh_proj.weight"), ("shared_head.norm.weight", "final_layernorm.weight")))
        ops.extend(_layer_converters(config, source, target + "transformer_layer.",
                                     bool(config.moe_layer_freq[-1]), config.mtp_enable_hyper_connections,
                                     config.experimental_attention_variant == "dsa" and config.mtp_enable_dsa))
    return ops


class Xing4WeightTemplate(WeightTemplate):
    """One HF schema for runtime loading and legacy NumPy conversion callbacks."""

    def __init__(self, config):
        config = _transformer_config(config)
        self.shared_weight_aliases = {
            f"model.layers.{config.num_layers + layer}.{name}": primary
            for layer in range(config.mtp_num_layers or 0)
            for name, primary in (("embed_tokens.weight", "model.embed_tokens.weight"),
                                  ("shared_head.head.weight", "lm_head.weight"))
        }
        super().__init__(weight_converters=_weight_converters(config))
        self.set_model_config(config)
        for op in self._expert_converters:
            op.optimizer_parallel_size = max(op.optimizer_parallel_size, 1)

    def get_mf_name(self, hf_name):
        # MTP consumes the main embedding/head; do not load their duplicate copies.
        if hf_name in self.shared_weight_aliases:
            return []
        return super().get_mf_name(hf_name)

    def _cast_weight_to_dtype(self, mf_name, weight):
        # Preserve learned router corrections and FP32 target normalization parameters.
        target = self.network_tensor_info.get(mf_name)
        if mf_name.endswith(".expert_bias") or (target is not None and target.dtype == float32):
            return weight
        return super()._cast_weight_to_dtype(mf_name, weight)

    def validate_loaded_weights(self, loaded_names, strict_load=False):
        """Check declared converter targets on this rank; auxiliary/adapter state is excluded."""
        expected = {name for op in self.weight_converters for name in op.mf_names}
        missing = (expected & self.network_tensor_info.keys()) - set(loaded_names)
        if missing:
            message = (f"Missing Xing4 checkpoint weights: {sorted(missing)[:10]}. "
                       "Check that the model type and DSA/MTP settings match the checkpoint; "
                       "missing parameters retain their initialized values.")
            if strict_load:
                raise ValueError(message)
            logger.warning(message)
        return missing

    def _find_expert_converter(self, dst_name):
        op = self.get_convert_op(dst_name, self.mf_name_to_converter)
        return op if isinstance(op, ExpertsConvertOp) else None

    def get_num_moe_experts(self, dst_name, src_names_num):
        op = self._find_expert_converter(dst_name)
        return src_names_num // len(op.hf_names) if op is not None else None

    def check_weights_for_qkv(self, weight_name):
        op = (self.get_convert_op(weight_name, self.hf_name_to_converter)
              or self.get_convert_op(weight_name, self.mf_name_to_converter))
        return isinstance(op, ConcatConvertOp) and op.mf_names[0].endswith(".self_attention.linear_qkv.weight")

    def preprocess_hf_weights(self, src_name, parameter, experts_weights, qkv_weights, num_moe_experts,
                              expert_offset=0, experts_per_card=None):
        op = self.get_convert_op(src_name, self.hf_name_to_converter)
        if isinstance(op, ExpertsConvertOp):
            count = experts_per_card if experts_per_card is not None else num_moe_experts
            expert = self.get_expert_id(src_name)
            if not expert_offset <= expert < expert_offset + count:
                return None, None
            target = op.mf_names[0]
            group = experts_weights.setdefault(target, {})
            group[src_name] = parameter
            if len(group) < count * len(op.hf_names):
                return None, None
            result = self._numpy_group(op, experts_weights.pop(target), range(expert_offset, expert_offset + count))
            return target, result[target]
        if self.check_weights_for_qkv(src_name):
            target = op.mf_names[0]
            group = qkv_weights.setdefault(target, {})
            group[src_name] = parameter
            if not all(name in group for name in op.hf_names):
                return None, None
            return target, op.hf_to_mf(qkv_weights.pop(target))[target]
        return super().preprocess_hf_weights(src_name, parameter, experts_weights, qkv_weights, num_moe_experts,
                                              expert_offset, experts_per_card)

    def _groups(self, source):
        """List present conversion groups using the same schema as runtime loading."""
        groups = []
        known = set(self.shared_weight_aliases)
        for op in self.weight_converters:
            required = ([name.format(expert) for name in op.hf_names for expert in range(op.num_moe_experts)]
                        if isinstance(op, ExpertsConvertOp) else op.hf_names)
            known.update(required)
            present = [name for name in required if name in source]
            if present:
                groups.append((op, required, present))
        unknown = source.keys() - known
        if unknown:
            raise ValueError(f"Unrecognized Xing4 HF weights: {sorted(unknown)[:5]}")
        return groups

    def numpy_name_map(self, source):
        result = {}
        for op, required, present in self._groups(source):
            missing = [name for name in required if name not in source]
            if missing:
                raise ValueError(f"HF weight group {present[0]!r} is missing {missing[:5]}")
            result.update({name: source[required[0]] for name in op.mf_names})
        return result

    def numpy_state_dict(self, source, shared=None, condition=None):
        """Convert one shard, publishing non-owner pieces before waiting."""
        groups = self._groups(source)
        owned = []
        pending = {}
        for op, required, present in groups:
            if required[0] in source:
                owned.append((op, required))
            else:
                pending.update({name: source[name] for name in present})
        if pending:
            if shared is None or condition is None:
                raise ValueError("Cross-shard HF conversion requires qkv_dict and condition.")
            with condition:
                shared.update(pending)
                condition.notify_all()
        result = {}
        for op, required in owned:
            values = {name: source[name] for name in required if name in source}
            missing = [name for name in required if name not in values]
            if missing:
                if shared is None or condition is None:
                    raise ValueError(f"HF weight group {required[0]!r} is missing {missing[:5]}")
                with condition:
                    if not condition.wait_for(lambda names=missing: all(name in shared for name in names), timeout=600):
                        raise ValueError(f"Missing cross-shard HF weights: {missing[:5]}")
                    values.update({name: shared.pop(name) for name in missing})
            result.update(self._numpy_group(op, values))
        return result

    @staticmethod
    def _numpy_group(op, values, expert_ids=None):
        """Apply a typed rule while retaining three-dimensional expert weights."""
        if not isinstance(op, ExpertsConvertOp):
            return op.hf_to_mf(values)
        expert_ids = range(op.num_moe_experts) if expert_ids is None else expert_ids
        local_op = copy(op)
        local_op.num_moe_experts = len(expert_ids)
        local_op.expert_parallel_size = local_op.optimizer_parallel_size = 1
        arrays = [np.stack([values[name.format(expert)] for expert in expert_ids]) for name in op.hf_names]
        value = local_op._hf_to_mf(arrays)[0]
        width = 2 * op.moe_ffn_hidden_size if len(arrays) == 2 else op.hidden_size
        first_dim = op.hidden_size if len(arrays) == 2 else op.moe_ffn_hidden_size
        return {op.mf_names[0]: value.reshape(len(expert_ids), first_dim, width)}


class Xing4PreTrainedModel(PreTrainedModel, ModelMixin):
    """Shared Xing4 model metadata and HF conversion schema."""

    config_class = Xing4Config
    base_model_prefix = "Xing4"

    @staticmethod
    def select_weight_converters(config):
        return _weight_converters(_transformer_config(config))

    def convert_name(self, weight_name):
        names = self.template.get_mf_name(weight_name)
        return names[0] if isinstance(names, list) and names else weight_name

    def get_model_parameters(self, only_trainable=True):
        pass
