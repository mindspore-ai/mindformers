# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test build config."""
from mindformers.models.build_config import build_model_config
from mindformers.models.qwen3.configuration_qwen3 import Qwen3Config


class TestBuildModelConfig:
    """A test class for testing build_model_config() method."""

    def test_build_qwen3_config(self):
        """Build a registered PyNative configuration without legacy YAML."""
        model_config = build_model_config({"model_type": "qwen3", "hidden_size": 128})
        assert isinstance(model_config, Qwen3Config)
        assert model_config.hidden_size == 128
