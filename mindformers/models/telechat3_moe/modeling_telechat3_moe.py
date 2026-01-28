# Copyright 2026 TeleAI Technologies Co., Ltd
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
"""Telechat3Moe Model."""
import os
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

from .configuration_telechat3_moe import TeleChat3MoeConfig
from .modeling_telechat3_moe_train import TrainingTeleChat3MoeForCausalLM


@MindFormerRegister.register(MindFormerModuleType.MODELS, legacy=False)
class TeleChat3MoeForCausalLM:
    r"""
    Provide TeleChat3 Model for training and inference.
    Args:
        config (TeleChat3Config): The config of TeleChat3 model.

    Returns:
        Tensor, the loss or logits of the network.
    """

    def __new__(cls, config: TeleChat3MoeConfig, *args, **kwargs):
        # get run mode to init different model.
        # we can use train mode to do train task.
        if os.environ.get("RUN_MODE") == "predict":
            raise ValueError("TeleChat3 model is currently only support training.")
        return TrainingTeleChat3MoeForCausalLM(config=config)
