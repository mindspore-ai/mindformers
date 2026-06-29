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
"""Deepseek-V4 Model."""
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

from .configuration_deepseek_v4 import DeepseekV4Config
from .modeling_deepseek_v4_pynative import PyNativeDeepseekV4ForCausalLM

@MindFormerRegister.register(MindFormerModuleType.MODELS, legacy=False)
class DeepseekV4ForCausalLM:
    r"""
    Provide DeepseekV4 Model for training and inference.
    Args:
        config (DeepseekV4Config): The config of DeepseekV4 model.

    Returns:
        Tensor, the loss or logits of the network.
    """

    def __new__(cls, config: DeepseekV4Config, *args, **kwargs):  # pylint: disable=unused-argument
        # get run mode to init different model.
        # predict mode used to deploy.
        # when predict mode not supported, we can use online_predict mode to do inference task.
        return PyNativeDeepseekV4ForCausalLM(config=config)
