# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Base Dataset."""
import os

import mindspore as ms
import mindspore.dataset as ds

from mindformers.tools.register import MindFormerConfig
from mindformers.tools.utils import get_real_rank, get_real_group_size
from mindformers.tools.logger import logger


class BaseDataset:
    """
    Base Dataset.

    Args:
        dataset_config (dict): Config for dataset.

    """

    def __init__(self, dataset_config: dict = None):
        self.dataset_config = dataset_config

    @classmethod
    def check_dataset_config(cls, dataset_config, params):
        """Check `dataset_config`, If it is empty, use the input parameter to create a new `dataset_config`."""
        if not dataset_config:
            params.pop("dataset_config")
            kwargs = params.pop("kwargs") if params.get("kwargs") else {}
            params.update(kwargs)
            dataset_config = MindFormerConfig(**params)
        return dataset_config

    @classmethod
    def init_dataset_config(cls, dataset_config):
        """Init dataset config."""
        ds.config.set_seed(dataset_config.get('seed', 0))
        ds.config.set_prefetch_size(dataset_config.get('prefetch_size', 1))
        ds.config.set_numa_enable(dataset_config.get('numa_enable', False))
        ds.config.set_num_parallel_workers(dataset_config.get('num_parallel_workers', 1))

        if dataset_config.auto_tune:
            if dataset_config.profile:
                raise EnvironmentError(
                    "MindSpore's AutoTune is enabled, so Profile cannot be enabled,"
                    "now Profile's flag is True, please set to False!")
            os.makedirs(dataset_config.filepath_prefix, exist_ok=True)
            dataset_config.filepath_prefix = os.path.join(dataset_config.filepath_prefix, "autotune")
            ds.config.set_enable_autotune(True, filepath_prefix=dataset_config.filepath_prefix)
            ds.config.set_autotune_interval(dataset_config.autotune_per_step)

    @classmethod
    def _generate_shard_info(cls):
        """Generate shard info for dataset."""
        shard_id = get_real_rank()
        num_shards = get_real_group_size()

        logger.info("Dataset shard_id: %s, num_shards: %s", shard_id, num_shards)
        return shard_id, num_shards

    @classmethod
    def _is_data_parallel(cls):
        return ms.context.get_auto_parallel_context("parallel_mode") == ms.context.ParallelMode.DATA_PARALLEL
