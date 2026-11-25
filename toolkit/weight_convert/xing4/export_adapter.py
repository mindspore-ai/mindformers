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
"""Offline Xing4 adapter using the runtime weight conversion schema."""
from mindformers.models.xing4.configuration_xing4 import Xing4Config
from mindformers.models.xing4.utils import Xing4WeightTemplate


class Xing4ExportTemplate(Xing4WeightTemplate):
    """Restore native shared-weight aliases when writing HF checkpoints."""

    def add_mf_weight(self, name, weight):
        """Restore the duplicate embedding/head entries used by native HF MTP."""
        result = super().add_mf_weight(name, weight)
        if result is not None:
            result.update({alias: result[primary] for alias, primary in self.shared_weight_aliases.items()
                           if primary in result})
        return result


class Xing4ExportAdapter:
    """Model-specific configuration and offline template factory."""

    config_class = Xing4Config
    export_model_types = ("xing4_v4", "xing4_v41", "Xing4.1-Preview")

    @staticmethod
    def build_weight_template(config):
        """Use the runtime schema for an unsharded offline export."""
        config = (config.get("_raw_model_section") or config) | {"model_type": config["model_type"]}
        template = Xing4ExportTemplate(config)
        for op in template._expert_converters:
            op.expert_parallel_size = op.optimizer_parallel_size = 1
        return template

    @staticmethod
    def export_hf_config(config):
        """Export architecture fields under the corresponding native HF identity."""
        config = (config.get("_raw_model_section") or config) | {"model_type": config["model_type"]}
        config = Xing4Config.from_dict(config)
        result = {name: getattr(config, name) for name in config._HF_DEFAULTS}
        is_v41 = config.model_type == "xing4_v41"
        result.update(
            model_type="Xing4.1-Preview" if is_v41 else "xing4_0",
            architectures=["DeepseekV32ForCausalLM" if is_v41 else "Xing4_0ForCausalLM"],
            scoring_func=config.moe_router_score_function,
            moe_layer_freq=1, ep_size=1, use_cache=True,
            bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id,
            pad_token_id=config.pad_token_id,
        )
        if is_v41:
            for name in ("hc_eps", "hc_mult", "hc_sinkhorn_iters",
                         "mhc_h_res_clamp_min", "mhc_h_res_clamp_max"):
                result.pop(name, None)
        if config.experimental_attention_variant != "dsa":
            for name in ("index_n_heads", "index_head_dim", "index_topk"):
                result.pop(name, None)
        if result["hidden_act"] == "fusedswiglu":
            result["hidden_act"] = "silu"
        return result
