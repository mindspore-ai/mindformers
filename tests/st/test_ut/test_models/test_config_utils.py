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
"""test config utils."""
import os
import pytest

from mindformers import MindFormerConfig
from mindformers.models.qwen3.configuration_qwen3 import Qwen3Config
from mindformers.models.utils import CONFIG_NAME

RMS_NORM_EPS = 1.0e-6
BATCH_SIZE = 4


class TestQwen3Config:
    """A test class for testing qwen3 config utils."""

    @pytest.fixture(autouse=True)
    def local_config_directory(self, tmp_path):
        """Use pytest's managed directory for each configuration test."""
        self.path = str(tmp_path)

    def test_build_qwen3_config_from_wrong_name(self):
        """test build qwen3 config from wrong name."""
        with pytest.raises(ValueError):
            _ = Qwen3Config.from_pretrained("xxxx")

        with pytest.raises(ValueError):
            _ = Qwen3Config.from_pretrained("mindspore/xxxx")

    def test_build_qwen3_config_from_repo(self, monkeypatch):
        """Resolve a repository config using a local cached JSON fixture."""
        config = Qwen3Config(rms_norm_eps=RMS_NORM_EPS)
        config.save_pretrained(self.path, save_json=True)
        monkeypatch.setattr("mindformers.models.configuration_utils.cached_file",
                            lambda *args, **kwargs: os.path.join(self.path, CONFIG_NAME))
        loaded = Qwen3Config.from_pretrained("mindformersinfra/test_qwen3")
        assert isinstance(loaded, Qwen3Config)
        assert loaded.rms_norm_eps == RMS_NORM_EPS

    def test_build_qwen3_config_from_wrong_repo(self, monkeypatch):
        """A repository without a cached config must produce a loading error."""
        monkeypatch.setattr("mindformers.models.configuration_utils.cached_file",
                            lambda *args, **kwargs: None)
        with pytest.raises(RuntimeError):
            _ = Qwen3Config.from_pretrained("repo/xxxx")

    def test_build_qwen3_config_from_dir_or_json(self):
        """test build qwen3 config from dir or json."""
        config = Qwen3Config(batch_size=BATCH_SIZE)
        config.save_pretrained(self.path, save_json=True)

        new_config = Qwen3Config.from_pretrained(self.path)
        assert isinstance(new_config, Qwen3Config)
        assert new_config.batch_size == BATCH_SIZE

        json_path = self.path + "/" + CONFIG_NAME
        new_config = Qwen3Config.from_pretrained(json_path)
        assert isinstance(new_config, Qwen3Config)
        assert new_config.batch_size == BATCH_SIZE

    def test_qwen3_save_pretrained(self):
        """test qwen3 config save pretrained."""
        config = Qwen3Config(batch_size=BATCH_SIZE)
        config.save_pretrained(self.path, save_name="mindspore_model")
        yaml_path = self.path + "/" + "mindspore_model.yaml"
        assert os.path.exists(yaml_path)
        mf_config = MindFormerConfig(yaml_path)
        assert mf_config.model.model_config.batch_size == BATCH_SIZE
