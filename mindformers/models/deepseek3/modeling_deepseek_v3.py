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
"""Deepseek-V3 Model."""
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

from .configuration_deepseek_v3 import DeepseekV3Config

try:
    from .modeling_deepseek_v3_pynative import PyNativeDeepseekV3ForCausalLM
except ImportError as e:
    PyNativeDeepseekV3ForCausalLM = None
    logger.warning(f"Import PyNativeDeepseekV3ForCausalLM failed: {e}.")


@MindFormerRegister.register(MindFormerModuleType.MODELS, search_names=DeepseekV3Config.model_type)
class DeepseekV3ForCausalLM:
    r"""
    Provide DeepseekV3 Model for training and inference.
    Args:
        config (DeepseekV3Config): The config of DeepseekV3 model.

    Returns:
        Tensor, the loss or logits of the network.
    """

    def __new__(cls, config: DeepseekV3Config, *args, **kwargs):  # pylint: disable=unused-argument
        return PyNativeDeepseekV3ForCausalLM(config=config)
