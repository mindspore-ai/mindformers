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
"""Qwen3 models' APIs."""
__all__ = [
    'Qwen3ForCausalLM',
]

from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

from .configuration_qwen3 import Qwen3Config
from .utils import Qwen3PreTrainedModel

try:
    from .modeling_qwen3_train_pynative import PyNativeQwen3ForCausalLM
except ImportError as e:
    PyNativeQwen3ForCausalLM = None
    logger.warning(f"Import PyNativeQwen3ForCausalLM failed: {e}.")


@MindFormerRegister.register(MindFormerModuleType.MODELS, search_names=Qwen3Config.model_type)
class Qwen3ForCausalLM(Qwen3PreTrainedModel):
    r"""
    Provide Qwen3 Model for training and inference.
    Args:
        config (Qwen3Config): The config of Qwen3 model.

    Returns:
        Tensor, the loss or logits of the network.
    """

    def __new__(cls, config):
        r"""
        get pynative model.

        Args:
            config (Qwen3Config): The config of qwen3 model.

        Returns:
            Tensor, the loss or logits of the network

        """
        return PyNativeQwen3ForCausalLM(config=config)
