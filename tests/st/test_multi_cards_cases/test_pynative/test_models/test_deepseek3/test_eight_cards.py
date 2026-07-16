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
"""test eight cards deepseek3 training"""

import os
import pytest

from tests.st.test_multi_cards_cases.utils import TaskType
from tests.utils.precision_utils import assert_expected_values_match
from tests.st.test_multi_cards_cases.test_pynative.test_models.test_deepseek3.utils import (
    build_case_config,
    generate_dataset,
    run_training_and_extract_losses,
    save_model_checkpoints,
)

_LEVEL_0_TASK_TIME = 360
_LEVEL_1_TASK_TIME = 240
_TASK_TYPE = TaskType.EIGHT_CARDS_TASK

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG = os.path.join(CUR_DIR, "pynative_ds3.yaml")
DATASET_PATH = os.path.join(CUR_DIR, "train_dataset")
CHECKPOINT_PATH = os.path.join(CUR_DIR, "deepseekv3_checkpoints")
RUN_SCRIPT_PATH = os.path.join(CUR_DIR, "run_deepseek3.py")


def run_deepseek3_case(config_name, log_dir, worker_num, updates=None):
    """Prepare deterministic data/checkpoint/config and run one DeepSeek3 case."""
    save_model_checkpoints(BASE_CONFIG, CHECKPOINT_PATH)
    generate_dataset(DATASET_PATH)
    local_config_path = build_case_config(
        BASE_CONFIG,
        os.path.join(CUR_DIR, config_name),
        CHECKPOINT_PATH,
        DATASET_PATH,
        updates or {},
    )
    assert os.path.exists(RUN_SCRIPT_PATH), f"Run script not found: {RUN_SCRIPT_PATH}"
    return run_training_and_extract_losses(RUN_SCRIPT_PATH, local_config_path, log_dir, worker_num)


@pytest.mark.level1
def test_fsdp_cp2_ulysses_tp2_gbs2():
    """
    Feature: DeepSeek3 FSDP + Context Parallel (ulysses) + Tensor Parallel training
    Description: Test DeepSeek3 training with FSDP, context parallel 2 using the ulysses
                 method and tensor parallel 2 on 8 cards, global batch size 2.
    Expectation: Training exits with code 0 and loss values match expected values.
    """
    expected_losses = [
        11.854254, 11.851755, 11.855100, 11.824228, 11.790492,
        11.769228, 11.770138, 11.758873, 11.689999, 11.621705,
    ]

    actual_losses = run_deepseek3_case(
        "ds3_fsdp_cp2_ulysses_tp2_gbs2.yaml",
        "log_fsdp_cp2_ulysses_tp2_gbs2",
        worker_num=8,
        updates={
            "parallelism": {
                "context_parallel": 2,
                "context_parallel_method": "ulysses",
                "tensor_parallel": 2,
            },
            "model": {"use_attn_mask_compression": True},
        },
    )
    assert_expected_values_match(actual_losses, expected_losses)


@pytest.mark.level0
def test_fsdp_cp2_ulysses_tp2_ep2_gbs2():
    """
    Feature: DeepSeek3 FSDP + Context Parallel (ulysses) + Tensor Parallel + Expert Parallel training
    Description: Test DeepSeek3 training with FSDP, context parallel 2 using the ulysses
                 method, tensor parallel 2 and expert parallel 2 on 8 cards, global batch size 2.
    Expectation: Training exits with code 0 and loss values match expected values.
    """
    expected_losses = [
        11.854254, 11.851396, 11.855400, 11.824049, 11.790096,
        11.769159, 11.769661, 11.757617, 11.690231, 11.620827,
    ]

    actual_losses = run_deepseek3_case(
        "ds3_fsdp_cp2_ulysses_tp2_ep2_gbs2.yaml",
        "log_fsdp_cp2_ulysses_tp2_ep2_gbs2",
        worker_num=8,
        updates={
            "parallelism": {
                "context_parallel": 2,
                "context_parallel_method": "ulysses",
                "tensor_parallel": 2,
                "expert_parallel": 2,
            },
            "model": {"use_attn_mask_compression": True},
        },
    )
    assert_expected_values_match(actual_losses, expected_losses)


@pytest.mark.level0
def test_fsdp_cp4_hybrid4_gbs2():
    """
    Feature: DeepSeek3 FSDP + Context Parallel (hybrid) training
    Description: Test DeepSeek3 training with FSDP and context parallel 4 using the hybrid
                 method on 8 cards, where the ulysses degree occupies 2 cards, global batch size 2.
    Expectation: Training exits with code 0 and loss values match expected values.
    """
    expected_losses = [
        11.854284, 11.851472, 11.855894, 11.823889, 11.790102,
        11.769896, 11.769511, 11.757105, 11.691558, 11.620907,
    ]

    actual_losses = run_deepseek3_case(
        "ds3_fsdp_cp4_hybrid4_gbs2.yaml",
        "log_fsdp_cp4_hybrid4_gbs2",
        worker_num=8,
        updates={
            "parallelism": {
                "context_parallel": 4,
                "context_parallel_method": "hybrid",
                "ulysses_degree_in_cp": 2,
            },
            "model": {"use_attn_mask_compression": True},
        },
    )
    assert_expected_values_match(actual_losses, expected_losses)
