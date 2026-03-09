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
"""Dataset Utils."""

import os
from typing import List, Union

import mindspore as ms

from mindformers.core.context.build_context import get_context
from mindformers.tools.utils import (
    get_dp_from_dataset_strategy,
    get_real_group_size,
    get_real_rank
)


def is_dataset_built_on_rank() -> bool:
    """check which rank need to build dataset."""
    ds_broadcast_level = get_context("dataset_broadcast_opt_level")

    global_rank_id = get_real_rank()
    stage_num = ms.get_auto_parallel_context("pipeline_stages")
    total_device_num = get_real_group_size() // stage_num
    dp = get_dp_from_dataset_strategy()
    tp = int(total_device_num // dp)

    local_stage_num = int(global_rank_id // (dp * tp))

    # when not stage 0 or last stage, no need to build dataset.
    if 0 < local_stage_num < (stage_num - 1) and ds_broadcast_level in [3, 1]:
        return False

    # In tp group, only need one card to build dataset, others don't need to build dataset.
    if global_rank_id % tp != 0 and ds_broadcast_level in [3, 2]:
        return False

    return True


def _get_mindrecord_files(dataset_files: Union[str, List[str]]) -> List[str]:
    """
    Get all MindRecord format files from given file paths or directories.

    This function recursively traverses specified directories, collects all files with the .mindrecord extension,
    and supports handling single file paths or lists of file paths.

    Args:
        dataset_files (Union[str, List[str]]): A single file path string or a list of file paths.
            Can be a specific .mindrecord file path or a directory path containing .mindrecord files.

    Returns:
        List[str]: A list of full paths to all found MindRecord files. Returns an empty list if no files are found.

    Raises:
        ValueError: Triggered when the input dataset_files is empty.
    """
    if not dataset_files:
        raise ValueError("dataset_files must be provided for MindDataset.")

    def _get_files_from_path(path: str) -> List[str]:
        if not isinstance(path, str):
            return []
        if path.endswith(".mindrecord"):
            return [path]
        if os.path.isdir(path):
            files_in_dir = []
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".mindrecord"):
                        files_in_dir.append(os.path.join(root, file))
            return files_in_dir
        return []

    if isinstance(dataset_files, str):
        dataset_files = [dataset_files]

    mindrecord_files = []
    if isinstance(dataset_files, list):
        for dataset_file in dataset_files:
            mindrecord_files.extend(_get_files_from_path(dataset_file))

    return mindrecord_files
