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
"""Telechat2 config converter."""
from typing import Dict, Any

from mindformers.parallel_core.config_converter import ConfigConverter, ConversionContext


class Telechat2ConfigConverter(ConfigConverter):
    """
    Telechat2ConfigConverter
    """
    CONFIG_MAPPING = {
        # Model Architecture
        "layer_norm_epsilon": "layernorm_epsilon",
        "n_head": "num_attention_heads",
        "n_layer": "num_layers",

        # Flash Attention
        "initializer_range": "init_method_std",
    }

    @classmethod
    def _pre_process(cls, model_config: Dict[str, Any], ctx: ConversionContext) -> None:
        super()._pre_process(model_config, ctx)
