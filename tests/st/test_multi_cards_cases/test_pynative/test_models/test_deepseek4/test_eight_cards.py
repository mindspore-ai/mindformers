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
"""test eight cards deepseek4 training"""

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

_LEVEL_1_TASK_TIME = 600
_TASK_TYPE = TaskType.EIGHT_CARDS_TASK

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG = os.path.join(CUR_DIR, "pynative_deepseek4.yaml")
DATASET_PATH = os.path.join(CUR_DIR, "train_dataset")
CHECKPOINT_PATH = os.path.join(CUR_DIR, "deepseek4_checkpoints")
RUN_SCRIPT_PATH = os.path.join(os.path.dirname(CUR_DIR), "test_deepseek3", "run_deepseek3.py")


def run_deepseek4_case(config_name, log_dir, worker_num, updates=None):
    """Prepare deterministic data/checkpoint/config and run one deepseek4 8-card case."""
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
def test_tp2_pp2_dp2_ep2_fsdp_vp2():
    """
    Feature: Deepseek4 8-card pynative training
    Description: Test deepseek4 training with TP=2, PP=2, DP=2, EP=2, FSDP(dp_shard=2),
                 VP=2 (interleave_num=2) with 1f1b schedule, selective recompute,
                 alltoall MoE dispatcher, mHC enabled, on 8 cards.
                 Parallelism: TP*PP*CP*DP = 2*2*1*2 = 8; FSDP shard=2 (dp_replicate=1, full shard).
                 This is the baseline configuration.
    Expectation: Training exits with code 0 and loss values match expected values.
    """
    expected_losses = [
        11.812731, 11.734726, 11.631442, 11.505877, 11.381109,
        11.237787, 11.105618, 10.963280, 10.828352, 10.691684,
    ]

    actual_losses = run_deepseek4_case(
        "deepseek4_tp2_pp2_dp2_ep2_fsdp_vp2.yaml",
        "log_tp2_pp2_dp2_ep2_fsdp_vp2",
        worker_num=8,
    )
    assert_expected_values_match(actual_losses, expected_losses)
