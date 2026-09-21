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
"""Test that the TND ``actual_seq_len`` column is carried as host data.

FlashAttention reads ``actual_seq_len`` on the host to build the TND cu_seqlens.
If it travels as a device Tensor, every op on the way to the kernel (the
micro-batch concat/slice and the model-entry reshape) produces a buffer that is
never filled under ``MS_SIMULATION_LEVEL`` -- dryrun launches no kernels -- and
FlashAttentionScore rejects the resulting all-zero cu_seqlens.
"""

import numpy as np
from mindspore import Tensor

from mindformers.pynative.trainer.trainer import Trainer, _keep_actual_seq_len_on_host


def _cu_seqlens(batch_index, pad_length=4, seq_length=16):
    """One sample's cu_seqlens, offset so batches are distinguishable."""
    step = seq_length // pad_length
    return np.arange(step, seq_length + step, step, dtype=np.int32) + batch_index


def test_keep_actual_seq_len_on_host_converts_the_column_to_numpy():
    """Feature: TND actual_seq_len host carriage.
    Description: A Tensor actual_seq_len column is converted to numpy.
    Expectation: The column is a numpy array holding the same values.
    """
    values = _cu_seqlens(0)
    batch = {"actual_seq_len": Tensor(values.reshape(1, -1))}

    result = _keep_actual_seq_len_on_host(batch)

    assert isinstance(result["actual_seq_len"], np.ndarray)
    np.testing.assert_array_equal(result["actual_seq_len"].reshape(-1), values)


def test_keep_actual_seq_len_on_host_leaves_other_columns_untouched():
    """Feature: TND actual_seq_len host carriage.
    Description: Only the actual_seq_len column is moved to the host.
    Expectation: Activation columns stay Tensors; a batch without the column is unchanged.
    """
    input_ids = Tensor(np.ones((1, 16), dtype=np.int32))
    batch = {"input_ids": input_ids, "actual_seq_len": Tensor(_cu_seqlens(0).reshape(1, -1))}

    result = _keep_actual_seq_len_on_host(batch)

    assert isinstance(result["input_ids"], Tensor)
    assert result["input_ids"] is input_ids

    without_column = {"input_ids": input_ids}
    assert _keep_actual_seq_len_on_host(without_column) == without_column


def test_collect_micro_batches_concatenates_actual_seq_len_on_host():
    """Feature: TND actual_seq_len host carriage.
    Description: Micro-batch collection must concatenate the column in numpy, not with ops.concat,
                 so the values survive for the pipeline schedule to slice per micro-batch.
    Expectation: The stacked column is a numpy array carrying every sample's cu_seqlens.
    """
    batches = [
        {
            "input_ids": Tensor(np.full((1, 16), index, dtype=np.int32)),
            "actual_seq_len": _cu_seqlens(index).reshape(1, -1),
        }
        for index in range(3)
    ]

    trainer = Trainer.__new__(Trainer)
    trainer._next_batch = lambda: batches.pop(0)  # pylint: disable=W0212

    stacked = trainer._collect_micro_batches(3)  # pylint: disable=W0212

    actual_seq_len = stacked["actual_seq_len"]
    assert isinstance(actual_seq_len, np.ndarray)
    assert actual_seq_len.shape == (3, 4)
    for index in range(3):
        np.testing.assert_array_equal(actual_seq_len[index], _cu_seqlens(index))
    assert isinstance(stacked["input_ids"], Tensor)
