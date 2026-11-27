# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
Test module for testing the interface used for mindformers.
How to run this:
python tests/ut/test_build.py
"""
from dataclasses import dataclass
from typing import Callable
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore import Parameter, Tensor

from mindformers.tools.logger import logger
from mindformers.tools import MindFormerRegister, MindFormerModuleType
from mindformers.models import build_model, build_processor
from mindformers.models import PretrainedConfig, PreTrainedModel
from mindformers.dataset import build_dataset


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class TestDataLoader:
    """Test DataLoader API For Register."""
    def __init__(self, dataset_dir=None):
        self.dataset_dir = dataset_dir


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class TestDataset:
    """Test Dataset API For Register."""
    def __init__(self, dataset_config=None):
        self.dataset_config = dataset_config


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class TestTransforms1:
    """Test Transforms API For Register."""
    def __init__(self):
        pass


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class TestTransforms2:
    """Test Transforms API For Register."""
    def __init__(self):
        pass


@MindFormerRegister.register(MindFormerModuleType.MASK_POLICY)
class TestModelMask:
    """Test Model Mask API For Register."""
    def __init__(self):
        pass


@MindFormerRegister.register(MindFormerModuleType.MODULES)
class TestAttentionModule:
    """Test Module API For Register."""
    def __init__(self):
        pass


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
@dataclass
class TestTextConfig(PretrainedConfig):
    """Test TextConfig API For Register."""
    seq_length: int = 12


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
@dataclass
class TestVisionConfig(PretrainedConfig):
    """Test VisionConfig API For Register."""
    seq_length: int = 12


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
@dataclass
class TestModelConfig(PretrainedConfig):
    """Test ModelConfig API For Register."""
    batch_size: int = 2
    embed_dim: int = 768
    text_config: Callable = TestTextConfig
    vision_config: Callable = TestVisionConfig


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class TestModel(PreTrainedModel):
    """Test Model API For Register."""
    def __init__(self, config: PretrainedConfig = None):
        super().__init__(config)
        self.model_config = config
        self.params = Parameter(Tensor([0.1]))

    def get_model_parameters(self, only_trainable=True):
        pass


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class TestTokenizer:
    """Test Tokenizer API For Register."""
    def __init__(self):
        pass


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class TestLLossMonitor(LossMonitor):
    """Test LossMonitor API For Register."""
    def __init__(self, per_print_times=1):
        super().__init__(per_print_times)
        self.print = per_print_times


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class TestTimeMonitor(TimeMonitor):
    """Test TimeMonitor API For Register."""
    def __init__(self, data_size=1):
        super().__init__(data_size)
        self.data_size = data_size


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class TestProcessor:
    """Test Processor API For Register."""
    def __init__(self):
        pass


def test_build_from_class_name():
    """
    Feature: Build API from class name
    Description: Test build function to instance API from class name
    Expectation: TypeError
    """
    # build dataset
    build_dataset(class_name='TestDataset')
    logger.info("Test Build Dataset Success")
    # build model
    build_model(class_name='TestModel', config=TestModelConfig())
    logger.info("Test Build Network Success")
    # build processor
    build_processor(class_name='TestProcessor')
    logger.info("Test Build Processor Success")
