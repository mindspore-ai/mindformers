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
"""Deepseek-V3 Model for training."""
# MindSpore ``Cell.construct`` intentionally exposes model-specific signatures.
# pylint: disable=arguments-differ
from mindspore import Tensor

from mindformers.models.deepseek3.utils import DeepseekV3PreTrainedModel
from mindformers.pynative.base_models.gpt.gpt_model import GPTModel
from mindformers.pynative.base_models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from mindformers.pynative.base_models.gpt.parallelize import parallelize_gptmodel
from mindformers.pynative.distributed.parallelize import register_parallelize
from mindformers.parallel_core.utils.model_mixin import TrainModelMixin

from .configuration_deepseek_v3 import DeepseekV3Config


@register_parallelize(parallelize_gptmodel)
class PyNativeDeepseekV3ForCausalLM(TrainModelMixin, DeepseekV3PreTrainedModel):
    """DeepseekV3 model for training"""

    def __init__(self, config: DeepseekV3Config, **kwargs):
        super().__init__(config, auto_prefix=False)
        transformer_config = self.convert_to_transformer_config(config, is_mla_model=True)
        if transformer_config.num_moe_experts:
            transformer_layer_spec = get_gpt_decoder_block_spec(transformer_config)
        else:
            raise ValueError("Only MoE model is supported in PyNativeDeepseekV3ForCausalLM.")
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
        """DeepseekV3 construct for training"""
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
