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
"""Deepseek-V4 Model for training."""
from mindspore import Tensor

from mindformers.checkpoint.converter.template import register_hf_weight_template
from mindformers.models.deepseek4.utils import DeepseekV4PreTrainedModel
from mindformers.pynative.base_models.gpt.gpt_model import GPTModel
from mindformers.pynative.base_models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from mindformers.pynative.base_models.gpt.parallelize import parallelize_gptmodel
from mindformers.pynative.distributed.parallelize import register_parallelize
from mindformers.parallel_core.utils.model_mixin import TrainModelMixin

from .configuration_deepseek_v4 import DeepseekV4Config


@register_parallelize(parallelize_gptmodel)
class PyNativeDeepseekV4ForCausalLM(TrainModelMixin, DeepseekV4PreTrainedModel):
    """DeepseekV4 model for training"""

    @register_hf_weight_template
    def __init__(self, config: DeepseekV4Config, **kwargs):
        super().__init__(config, auto_prefix=False)
        transformer_config = self.convert_to_transformer_config(config, is_mla_model=True)
        if transformer_config.num_moe_experts:
            transformer_layer_spec = get_gpt_decoder_block_spec(transformer_config)
        else:
            raise ValueError("Only MoE model is supported in PyNativeDeepseekV4ForCausalLM.")
        self.model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=transformer_config.vocab_size,
            max_sequence_length=transformer_config.max_position_embeddings,
            position_embedding_type=transformer_config.position_embedding_type,
            rotary_percent=1.0,
            rotary_base=transformer_config.rotary_base,
            rope_scaling=False,
            layer_start=kwargs.get("layer_start", 0),
            layer_end=kwargs.get("layer_end", None),
            stage_idx=kwargs.get("stage_idx", 0),
            vp_size=kwargs.get("vp_size", 1),
        )

    # pylint: disable=W0221
    def convert_weight_dict(self, source_dict, **kwargs):
        """
        convert HuggingFace weight dict to MindFormers weight dict.

        Args:
            source_dict: origin weight dict.
            kwargs: additional specific kwargs.

        Returns:
            ms_weight_dict: converted weight dict.
        """
        transformer_config = self.get_gpt_transformer_config()

        ms_weight_dict = {}
        # FFN weight keys for MoE expert handling
        w1_keys = []  # gate_proj / gating
        w2_keys = []  # down_proj / linear_fc2
        w3_keys = []  # up_proj / hidden

        for k, v in source_dict.items():
            k = self.convert_name(k)
            ms_weight_dict.update({k: v})

            part = k.split('.')
            # Get experts Keys in MoE
            if part[-2] == 'gating':
                w1_keys.append(k)
            if part[-2] == 'linear_fc2':
                w2_keys.append(k)
            if part[-2] == 'hidden':
                w3_keys.append(k)

        qkv_dict = kwargs.get('qkv_dict', None)
        condition = kwargs.get('condition', None)

        # Note: V4 uses MLA (Multi-Latent Attention), Q/KV projections are separate weights.
        # No QKV concatenation is needed, unlike standard attention models (e.g., Qwen3Moe).

        # Concat gate and up weight to linear_fc1 (for dense layers/shared experts)
        self.concat_ffn_weight_infer(w1_keys, w3_keys, qkv_dict, condition, ms_weight_dict)

        # Stack experts weight in each layer of linear_fc1 and linear_fc2
        self.concat_expert_weight(
            w2_keys=w2_keys, expert_weight_dict=qkv_dict, condition=condition, ms_weight_dict=ms_weight_dict,
            num_layers=transformer_config.num_layers,
            num_experts=transformer_config.num_moe_experts
        )

        return ms_weight_dict

    # pylint: disable=W0221
    def convert_map_dict(self, hf_name_map_dict, **kwargs):
        """
        convert HuggingFace map dict to MindFormers map dict.

        Args:
            hf_name_map_dict: origin weight dict.
            kwargs: additional specific kwargs.

        Returns:
            ms_name_map_dict: converted weight dict.
        """
        ms_name_map_dict = {}
        w1_keys = []
        w2_keys = []

        # Get gate and down keys
        for k, v in hf_name_map_dict.items():
            k = self.convert_name(k)
            ms_name_map_dict.update({k: v})
            part = k.split('.')
            if part[-2] == 'gating':
                w1_keys.append(k)
            if part[-2] == 'linear_fc2':
                w2_keys.append(k)

        # Note: V4 uses MLA, no QKV concat needed.
        # The Q/KV weights are already separate in both HF and MF.

        # For experts: only keep the first expert's key and rename to the stacked weight key.
        # This ensures the weight distribution logic can correctly locate the source weights.
        # For shared experts: replace gating + hidden with linear_fc1 following Qwen3 MoE pattern.
        for w1_key in w1_keys:
            w3_key = w1_key.replace('gating', 'hidden')
            ms_name_map_dict.pop(w3_key)

            if '.experts.' in w1_key:
                # Routed experts: stack gate+up into experts.weight1
                if w1_key.split('.')[-3] == '0':
                    fc1_value = ms_name_map_dict.pop(w1_key)
                    fc1_key = w1_key.split('.experts.0.')[0] + '.experts.weight1'
                    ms_name_map_dict.update({fc1_key: fc1_value})
                else:
                    ms_name_map_dict.pop(w1_key)
            else:
                # Shared experts / dense FFN: concat gate+up into linear_fc1
                w1_value = ms_name_map_dict.pop(w1_key)
                w_gate_hidden_key = w1_key.replace('gating', 'linear_fc1')
                ms_name_map_dict.update({w_gate_hidden_key: w1_value})

        for w2_key in w2_keys:
            if '.experts.' in w2_key:
                # Routed experts: stack down_proj into experts.weight2
                if w2_key.split('.')[-3] == '0':
                    fc2_value = ms_name_map_dict.pop(w2_key)
                    fc2_key = w2_key.split('.experts.0.')[0] + '.experts.weight2'
                    ms_name_map_dict.update({fc2_key: fc2_value})
                else:
                    ms_name_map_dict.pop(w2_key)
            # Shared experts / dense FFN: keep as-is (already linear_fc2)

        return ms_name_map_dict

    def construct(
            self,
            decoder_input: Tensor = None,
            input_ids: Tensor = None,
            labels: Tensor = None,
            attention_mask: Tensor = None,
            loss_mask=None,
            position_ids: Tensor = None,
            actual_seq_len=None
    ):
        """DeepseekV4 construct for training"""
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            loss_mask=loss_mask,
            actual_seq_len=actual_seq_len
        )

    def _update_expert_bias(self, metric_group, metric_group_size):
        return self.model._update_expert_bias(metric_group, metric_group_size)

    def get_load_balancing_loss(
        self, metric_group, metric_group_size, pp_metric_group, pp_metric_group_size, **kwargs
    ):
        return self.model.get_load_balancing_loss(
            metric_group, metric_group_size,
            pp_metric_group, pp_metric_group_size, **kwargs,
        )

    def reset_model_temporary_tensors(self):
        return self.model.reset_model_temporary_tensors()

    def get_mtp_loss(self, metric_group, metric_group_size):
        return self.model.get_mtp_loss(metric_group, metric_group_size)

    def get_index_loss(self, *args, **kwargs):
        return self.model.get_index_loss(*args, **kwargs)
