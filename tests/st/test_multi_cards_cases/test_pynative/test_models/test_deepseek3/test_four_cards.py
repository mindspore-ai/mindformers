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
"""test four cards deepseek3 training"""

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

_LEVEL_0_TASK_TIME = 240
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.FOUR_CARDS_TASK

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
def test_fsdp_tp2_gbs2():
    """
    Feature: DeepSeek3 FSDP + Tensor Parallel training
    Description: Test DeepSeek3 training with FSDP and tensor parallel 2 on 4 cards,
                 global batch size 2. Verifies loss values match the expected standard.
    Expectation: Training exits with code 0 and loss values match expected values.
    """
    expected_losses = [
        11.855282, 11.850258, 11.854774, 11.822327, 11.791462,
        11.772634, 11.765192, 11.757770, 11.686123, 11.615366,
    ]

    actual_losses = run_deepseek3_case(
        "ds3_fsdp_tp2_gbs2.yaml",
        "log_fsdp_tp2_gbs2",
        worker_num=4,
        updates={"parallelism": {"tensor_parallel": 2}},
    )
    assert_expected_values_match(actual_losses, expected_losses)


@pytest.mark.level0
def test_fsdp_tp2_ep2_gbs2():
    """
    Feature: DeepSeek3 FSDP + Tensor Parallel + Expert Parallel training
    Description: Test DeepSeek3 training with FSDP, tensor parallel 2 and
                 expert parallel 2 on 4 cards, global batch size 2.
                 Verifies loss values match the expected standard.
    Expectation: Training exits with code 0 and loss values match expected values.
    """
    expected_losses = [
        11.855282, 11.850237, 11.855066, 11.822032, 11.791618,
        11.772750, 11.766208, 11.757825, 11.686206, 11.615742,
    ]

    actual_losses = run_deepseek3_case(
        "ds3_fsdp_tp2_ep2_gbs2.yaml",
        "log_fsdp_tp2_ep2_gbs2",
        worker_num=4,
        updates={"parallelism": {"tensor_parallel": 2, "expert_parallel": 2}},
    )
    assert_expected_values_match(actual_losses, expected_losses)


@pytest.mark.level0
def test_fsdp_tp2_ep2_muon_qk_clip_gbs2():
    """
    Feature: DeepSeek3 Muon optimizer QK-clip under tensor parallelism.
    Description: Train DeepSeek3 with the Muon optimizer and qk_clip_enabled=True under
                 tensor parallel 2 (+ expert parallel 2) on 4 cards, global batch size 2.
                 Muon + qk_clip turns on forward max-attention-logit tracking; with tensor
                 parallel >= 2 the per-head ``max_logits_val`` is sharded over the head dim,
                 so ``FlashAttention._update_max_logits`` must compare the local ``amax``
                 head count against the LOCAL shard length of ``max_logits_val``. Comparing
                 it against the GLOBAL head count wrongly takes the head-slice branch and
                 indexes the local shard with global coordinates (e.g. ``max_logits_val[4:8]``
                 on a length-4 local shard -> empty -> ``For 'Maximum' ... input1.shape = [0]``).
                 Regression guard for that qk_clip x tensor-parallel forward crash.
    Expectation: Training exits with code 0 and loss values match expected values.
    """
    expected_losses = [
        11.855282, 11.848942, 11.866066, 11.854048, 11.843377,
        11.849504, 11.844106, 11.852513, 11.839876, 11.816915,
    ]

    actual_losses = run_deepseek3_case(
        "ds3_fsdp_tp2_ep2_muon_qk_clip_gbs2.yaml",
        "log_fsdp_tp2_ep2_muon_qk_clip_gbs2",
        worker_num=4,
        updates={
            "parallelism": {"tensor_parallel": 2, "expert_parallel": 2},
            "optimizer": {
                "type": "Muon",
                "weight_decay": 0.1,
                "matched_adamw_rms": 0.2,
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 5,
                "adamw_betas": [0.95, 0.95],
                "adamw_eps": 1.0e-08,
                "qk_clip_threshold": 100,
                "qk_clip_enabled": True,
                "comm_strategy": "allgather_deredundency",
                "use_fused_adamw": True,
            },
        },
    )
    assert_expected_values_match(actual_losses, expected_losses)


@pytest.mark.level1
def test_fsdp_cp2_colossal_gbs2():
    """
    Feature: DeepSeek3 FSDP + Context Parallel (colossal) training
    Description: Test DeepSeek3 training with FSDP and context parallel 2 using the
                 colossal method on 4 cards, global batch size 2.
    Expectation: Training exits with code 0 and loss values match expected values.
    """
    expected_losses = [
        11.854298, 11.852139, 11.855262, 11.823850, 11.790343,
        11.769970, 11.770279, 11.758461, 11.692724, 11.621376,
    ]

    actual_losses = run_deepseek3_case(
        "ds3_fsdp_cp2_colossal_gbs2.yaml",
        "log_fsdp_cp2_colossal_gbs2",
        worker_num=4,
        updates={
            "parallelism": {"context_parallel": 2, "context_parallel_method": "colossal"},
            "model": {"use_attn_mask_compression": True},
        },
    )
    assert_expected_values_match(actual_losses, expected_losses)


@pytest.mark.level1
def test_fsdp_cp2_ulysses_gbs2():
    """
    Feature: DeepSeek3 FSDP + Context Parallel (ulysses) training
    Description: Test DeepSeek3 training with FSDP and context parallel 2 using the
                 ulysses method on 4 cards, global batch size 2.
    Expectation: Training exits with code 0 and loss values match expected values.
    """
    expected_losses = [
        11.854298, 11.851341, 11.855456, 11.823189, 11.790516,
        11.769073, 11.769979, 11.758433, 11.691092, 11.620988,
    ]

    actual_losses = run_deepseek3_case(
        "ds3_fsdp_cp2_ulysses_gbs2.yaml",
        "log_fsdp_cp2_ulysses_gbs2",
        worker_num=4,
        updates={
            "parallelism": {"context_parallel": 2, "context_parallel_method": "ulysses"},
            "model": {"use_attn_mask_compression": True},
        },
    )
    assert_expected_values_match(actual_losses, expected_losses)


@pytest.mark.level1
def test_fsdp_cp2_ulysses_async_gbs2():
    """
    Feature: DeepSeek3 FSDP + Async Context Parallel (ulysses) training
    Description: Test DeepSeek3 training with FSDP and context parallel 2 using the
                 ulysses method with async context parallel enabled on 4 cards,
                 global batch size 2.
    Expectation: Training exits with code 0 and loss values match expected values.
    """
    expected_losses = [
        11.854298, 11.851341, 11.855456, 11.823189, 11.790516,
        11.769073, 11.769979, 11.758433, 11.691092, 11.620988,
    ]

    actual_losses = run_deepseek3_case(
        "ds3_fsdp_cp2_ulysses_async_gbs2.yaml",
        "log_fsdp_cp2_ulysses_async_gbs2",
        worker_num=4,
        updates={
            "parallelism": {
                "context_parallel": 2,
                "context_parallel_method": "ulysses",
                "context_parallel_async": True,
            },
            "model": {"use_attn_mask_compression": True},
        },
    )
    assert_expected_values_match(actual_losses, expected_losses)


@pytest.mark.level0
def test_fsdp_tp2_ep2_deredundancy_gbs2():
    """
    Feature: DeepSeek3 FSDP + Tensor Parallel + Expert Parallel with alltoall_deredundancy training
    Description: Test DeepSeek3 training with FSDP, tensor parallel 2, expert parallel 2 and
                 moe_token_dispatcher_type alltoall_deredundancy on 4 cards, global batch size 2.
    Expectation: Training exits with code 0 and loss values match expected values.
    """
    expected_losses = [
        11.855282, 11.849820, 11.854507, 11.822180, 11.791955,
        11.772362, 11.765581, 11.757377, 11.686357, 11.615234,
    ]

    actual_losses = run_deepseek3_case(
        "ds3_fsdp_tp2_ep2_deredundancy_gbs2.yaml",
        "log_fsdp_tp2_ep2_deredundancy_gbs2",
        worker_num=4,
        updates={
            "parallelism": {
                "tensor_parallel": 2,
                "expert_parallel": 2,
                "moe_token_dispatcher_type": "alltoall_deredundancy",
                "npu_nums_per_device": 2,
            },
        },
    )
    assert_expected_values_match(actual_losses, expected_losses)
