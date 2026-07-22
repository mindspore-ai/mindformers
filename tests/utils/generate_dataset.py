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
"""Utils for model training"""

import os
import numpy as np
from mindspore.mindrecord import FileWriter, MindRecordException


def generate_mindrecord_file(
    seq_length: int = 128,
    batch_size: int = 1,
    train_steps: int = 1000,
    dataset_path: str = None,
    data_schema: dict = None
):
    """
    Generate mindrecord file.

    Args:
        seq_length (int): Sequence length of each sample. Default: 128.
        batch_size (int): Batch size for training. Default: 1.
        train_steps (int): Number of training steps. Default: 1000.
        dataset_path (str): Path to save the generated mindrecord file.
            If None, defaults to "./test.mindrecord". Default: None.
    """
    if dataset_path is None:
        raise ValueError("dataset_path should be specified.")
    if data_schema is None:
        raise ValueError("data_schema should be specified.")

    data_dir = os.path.dirname(dataset_path)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)

    data_num = batch_size * train_steps
    np.random.seed(0)

    def _resolve_shape(shape):
        return tuple(seq_length if dim == -1 else dim for dim in shape)

    def _generate_data(dtype, shape):
        np_dtype = np.dtype(dtype)
        if np_dtype in (np.int32, np.uint8, np.int64):
            return np.random.randint(0, 1024, size=shape).astype(np_dtype)
        if np_dtype in (np.float16, np.float32, np.float64):
            return np.random.rand(*shape).astype(np_dtype)
        raise ValueError(f"Unsupported dtype: {dtype}")

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            writer = FileWriter(dataset_path)
            writer.add_schema(data_schema, "test-schema")
            for _ in range(data_num):
                features = {}
                for field_name, field_info in data_schema.items():
                    resolved_shape = _resolve_shape(field_info["shape"])
                    features[field_name] = _generate_data(field_info["type"], resolved_shape)
                writer.write_raw_data([features])
            writer.commit()
            retry = False
            success_sig = True
        except MindRecordException as e:
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
            if os.path.exists(dataset_path + ".db"):
                os.remove(dataset_path + ".db")
            print(f"mindrecord data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"mindrecord data initialize failed for {count} times.")


def generate_eod_mindrecord_file(
    seq_length: int = 4096,
    batch_size: int = 1,
    train_steps: int = 1000,
    dataset_path: str = None,
    num_segments: int = 16,
):
    """
    Generate a mindrecord file carrying a valid ``actual_seq_len`` column for EOD/TND training.

    Unlike :func:`generate_mindrecord_file`, ``actual_seq_len`` and ``position_ids`` cannot be
    random: the TND flash-attention layout interprets ``actual_seq_len`` as cumulative packing
    boundaries (``cu_seqlens``) and ``position_ids`` must reset at every segment start. This helper
    packs each sample into ``num_segments`` uniform segments so both are well-formed. The presence
    of the ``actual_seq_len`` column is what makes ``MindDataset`` enable the compressed EOD mask.

    Args:
        seq_length (int): Sequence length of each sample. Must be divisible by ``num_segments``.
        batch_size (int): Batch size used to size the total sample count.
        train_steps (int): Number of training steps used to size the total sample count.
        dataset_path (str): Path to save the generated mindrecord file.
        num_segments (int): Number of packed sub-sequences per sample.
    """
    if dataset_path is None:
        raise ValueError("dataset_path should be specified.")
    if seq_length % num_segments != 0:
        raise ValueError(
            f"seq_length {seq_length} must be divisible by num_segments {num_segments}."
        )

    data_dir = os.path.dirname(dataset_path)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)

    seg_len = seq_length // num_segments
    # cumulative packing boundaries ending at seq_length, e.g. [256, 512, ..., 4096]
    actual_seq_len = np.arange(seg_len, seq_length + seg_len, seg_len, dtype=np.int32)
    # position ids reset to 0 at each segment start
    position_ids = np.tile(np.arange(seg_len, dtype=np.int32), num_segments)

    data_schema = {
        "input_ids": {"type": "int32", "shape": [-1]},
        "labels": {"type": "int32", "shape": [-1]},
        "loss_mask": {"type": "int32", "shape": [-1]},
        "position_ids": {"type": "int32", "shape": [-1]},
        "actual_seq_len": {"type": "int32", "shape": [-1]},
    }
    data_num = batch_size * train_steps
    np.random.seed(0)

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            writer = FileWriter(dataset_path)
            writer.add_schema(data_schema, "test-schema")
            for _ in range(data_num):
                features = {
                    "input_ids": np.random.randint(0, 1024, size=seq_length).astype(np.int32),
                    "labels": np.random.randint(0, 1024, size=seq_length).astype(np.int32),
                    "loss_mask": np.ones(seq_length, dtype=np.int32),
                    "position_ids": position_ids,
                    "actual_seq_len": actual_seq_len,
                }
                writer.write_raw_data([features])
            writer.commit()
            retry = False
            success_sig = True
        except MindRecordException as e:
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
            if os.path.exists(dataset_path + ".db"):
                os.remove(dataset_path + ".db")
            print(f"mindrecord data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"mindrecord data initialize failed for {count} times.")
