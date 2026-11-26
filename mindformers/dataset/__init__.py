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
"""MindFormers Dataset."""
from .dataloader import HFDataLoader, BlendedMegatronDatasetDataLoader, OrderedIndexDataLoader
from .transforms import (
    BCHW2BHWC,
    BatchCenterCrop,
    BatchNormalize,
    BatchPILize,
    BatchResize,
    BatchToTensor,
    RandomCropDecodeResize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
)
from .transforms.build_transforms import build_transforms
from .handler import AlpacaInstructDataHandler, CodeAlpacaInstructDataHandler, build_data_handler
from .build_dataset import build_dataset
from .dataloader.build_dataloader import build_dataset_loader
from .base_dataset import BaseDataset
from .causal_language_model_dataset import CausalLanguageModelDataset
from .general_dataset import GeneralDataset
from .utils import check_dataset_config, check_dataset_iterable

__all__ = ['CausalLanguageModelDataset', 'GeneralDataset']
