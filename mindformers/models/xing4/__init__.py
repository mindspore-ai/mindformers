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
"""Xing4 model and configuration exports."""

from mindformers.tools.logger import logger

from .configuration_xing4 import Xing4Config, Xing4V4Config, Xing4V41Config
try:
    from .modeling_xing4 import Xing4ForCausalLM
except ImportError as exc:
    Xing4ForCausalLM = None
    logger.warning(f"Import Xing4ForCausalLM failed: {exc}.")

__all__ = ['Xing4Config', 'Xing4V4Config', 'Xing4V41Config', 'Xing4ForCausalLM']
