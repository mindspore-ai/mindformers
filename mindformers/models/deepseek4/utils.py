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
from mindformers.models.deepseek3.utils import DeepseekV3PreTrainedModel
from mindformers.models.deepseek4.configuration_deepseek_v4 import DeepseekV4Config


class DeepseekV4PreTrainedModel(DeepseekV3PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DeepseekV4Config
    base_model_prefix = "Deepseekv4"

    def get_model_parameters(self, only_trainable=True):
        pass
