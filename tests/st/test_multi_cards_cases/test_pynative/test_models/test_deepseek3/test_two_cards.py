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
"""test two cards deepseek3 training"""

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

_LEVEL_0_TASK_TIME = 120
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK

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


@pytest.mark.level0
def test_fsdp_gbs2():
    """
    Feature: DeepSeek3 FSDP training
    Description: Test DeepSeek3 training with FSDP only on 2 cards, global batch size 2.
                 Verifies loss values match the expected standard.
    Expectation: Training exits with code 0 and loss values match expected values.
    """
    expected_losses = [
        11.855703, 11.849623, 11.854856, 11.821927, 11.791636,
        11.773806, 11.766010, 11.758839, 11.687195, 11.617872,
    ]

    actual_losses = run_deepseek3_case("ds3_fsdp_gbs2.yaml", "log_fsdp_gbs2", worker_num=2)
    assert_expected_values_match(actual_losses, expected_losses)


@pytest.mark.level0
def test_fsdp_ep2_gbs2():
    """
    Feature: DeepSeek3 FSDP + Expert Parallel training
    Description: Test DeepSeek3 training with FSDP and expert parallel 2 on 2 cards,
                 global batch size 2. Verifies loss values match the expected standard.
    Expectation: Training exits with code 0 and loss values match expected values.
    """
    expected_losses = [
        11.855703, 11.849949, 11.854990, 11.822470, 11.791677,
        11.773118, 11.766552, 11.759279, 11.686920, 11.617682,
    ]

    actual_losses = run_deepseek3_case(
        "ds3_fsdp_ep2_gbs2.yaml",
        "log_fsdp_ep2_gbs2",
        worker_num=2,
        updates={"parallelism": {"expert_parallel": 2}},
    )
    assert_expected_values_match(actual_losses, expected_losses)
