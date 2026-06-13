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
import random
import subprocess
import yaml
import pytest

from tests.st.test_multi_cards_cases.utils import TaskType
from tests.utils.generate_dataset import generate_mindrecord_file

_LEVEL_0_TASK_TIME = 240
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.FOUR_CARDS_TASK

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG = os.path.join(CUR_DIR, "pynative_ds3.yaml")
DATASET_PATH = os.path.join(CUR_DIR, "train_dataset")


def generate_dataset():
    """Generate mindrecord dataset for two-cards training tests."""
    if not os.path.isdir(DATASET_PATH):
        generate_mindrecord_file(
            seq_length=4096,
            batch_size=2,
            train_steps=1000,
            dataset_path=os.path.join(CUR_DIR, "train_dataset/dataset.mindrecord"),
            data_schema={
                "input_ids": {"type": "int32", "shape": [-1]},
                "labels": {"type": "int32", "shape": [-1]},
                "loss_mask": {"type": "int32", "shape": [-1]},
                "position_ids": {"type": "int32", "shape": [-1]},
            }
        )


@pytest.mark.level0
def test_fsdp_tp2_gbs2():
    """
    Feature: DeepSeek3 FSDP + Tensor Parallel training
    Description: Test DeepSeek3 training with FSDP and tensor parallel 2 on 4 cards, global batch size 2.
    Expectation: msrun script exits with code 0.
    """
    generate_dataset()
    with open(BASE_CONFIG, "r") as fp:
        configs = yaml.safe_load(fp)

    configs["train_dataset"]["dataloader"]["dataset_files"] = [DATASET_PATH]
    configs["training"]["local_batch_size"] = 1
    configs["parallelism"]["tensor_parallel"] = 2
    configs["training"]["steps"] = 10

    local_config_path = f"{CUR_DIR}/ds3_fsdp_tp2_gbs2.yaml"
    with open(local_config_path, "w") as fp:
        yaml.dump(configs, fp, indent=2)

    run_script_path = os.path.join(CUR_DIR, "run_deepseek3.py")
    port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
    assert os.path.exists(
        run_script_path
    ), f"Run script not found: {run_script_path}"
    cmd = [
        "msrun",
        "--worker_num=4",
        "--local_worker_num=4",
        f"--master_port={port_id}",
        f"--log_dir={CUR_DIR}/log_fsdp_tp2_gbs2",
        "--join=True",
        f"{run_script_path}",
        "--config",
        f"{local_config_path}",
    ]
    result = subprocess.run(
        cmd, shell=False, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"DeepSeek3 training (run_deepseek3.py, config={local_config_path}) failed with non-zero exit code: "
        f"{result.returncode}.\n"
        f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    )


@pytest.mark.level0
def test_fsdp_tp2_ep2_gbs2():
    """
    Feature: DeepSeek3 FSDP + Tensor Parallel + Expert Parallel training
    Description: Test DeepSeek3 training with FSDP, tensor parallel 2 and
                 expert parallel 2 on 4 cards, global batch size 2.
    Expectation: msrun script exits with code 0.
    """
    generate_dataset()
    with open(BASE_CONFIG, "r") as fp:
        configs = yaml.safe_load(fp)

    configs["train_dataset"]["dataloader"]["dataset_files"] = [DATASET_PATH]
    configs["training"]["local_batch_size"] = 1
    configs["parallelism"]["tensor_parallel"] = 2
    configs["parallelism"]["expert_parallel"] = 2
    configs["training"]["steps"] = 10

    local_config_path = f"{CUR_DIR}/ds3_fsdp_tp2_ep2_gbs2.yaml"
    with open(local_config_path, "w") as fp:
        yaml.dump(configs, fp, indent=2)

    run_script_path = os.path.join(CUR_DIR, "run_deepseek3.py")
    port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
    assert os.path.exists(
        run_script_path
    ), f"Run script not found: {run_script_path}"
    cmd = [
        "msrun",
        "--worker_num=4",
        "--local_worker_num=4",
        f"--master_port={port_id}",
        f"--log_dir={CUR_DIR}/log_fsdp_tp2_ep2_gbs2",
        "--join=True",
        f"{run_script_path}",
        "--config",
        f"{local_config_path}",
    ]
    result = subprocess.run(
        cmd, shell=False, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"DeepSeek3 training (run_deepseek3.py, config={local_config_path}) failed with non-zero exit code: "
        f"{result.returncode}.\n"
        f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    )


@pytest.mark.level0
def test_fsdp_tp2_ep2_muon_qk_clip_gbs2():
    """
    Feature: DeepSeek3 Muon optimizer QK-clip under tensor parallelism.
    Description: Train DeepSeek3 with the Muon optimizer and qk_clip_enabled=True under
                 tensor parallel 2 (+ expert parallel 2) on 4 cards. Muon + qk_clip turns on
                 forward max-attention-logit tracking; with tensor parallel >= 2 the per-head
                 ``max_logits_val`` is sharded over the head dim, so
                 ``FlashAttention._update_max_logits`` must compare the local ``amax`` head
                 count against the LOCAL shard length of ``max_logits_val``. Comparing it
                 against the GLOBAL head count wrongly takes the head-slice branch and indexes
                 the local shard with global coordinates (e.g. ``max_logits_val[4:8]`` on a
                 length-4 local shard -> empty -> ``For 'Maximum' ... input1.shape = [0]``).
                 Regression guard for that qk_clip x tensor-parallel forward crash.
    Expectation: msrun script exits with code 0.
    """
    generate_dataset()
    with open(BASE_CONFIG, "r") as fp:
        configs = yaml.safe_load(fp)

    configs["train_dataset"]["dataloader"]["dataset_files"] = [DATASET_PATH]
    configs["training"]["local_batch_size"] = 1
    configs["parallelism"]["tensor_parallel"] = 2
    configs["parallelism"]["expert_parallel"] = 2
    configs["training"]["steps"] = 10
    # Muon + qk_clip_enabled is what turns on the forward max-logit tracking; under
    # tensor parallel >= 2 ``max_logits_val`` is head-sharded, exercising the path that
    # previously crashed (local vs global head-count mismatch in _update_max_logits).
    configs["optimizer"] = {
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
    }

    local_config_path = f"{CUR_DIR}/ds3_fsdp_tp2_ep2_muon_qk_clip_gbs2.yaml"
    with open(local_config_path, "w") as fp:
        yaml.dump(configs, fp, indent=2)

    run_script_path = os.path.join(CUR_DIR, "run_deepseek3.py")
    port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
    assert os.path.exists(
        run_script_path
    ), f"Run script not found: {run_script_path}"
    cmd = [
        "msrun",
        "--worker_num=4",
        "--local_worker_num=4",
        f"--master_port={port_id}",
        f"--log_dir={CUR_DIR}/log_fsdp_tp2_ep2_muon_qk_clip_gbs2",
        "--join=True",
        f"{run_script_path}",
        "--config",
        f"{local_config_path}",
    ]
    result = subprocess.run(
        cmd, shell=False, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"DeepSeek3 Muon qk_clip x tensor-parallel training (run_deepseek3.py, "
        f"config={local_config_path}) failed with non-zero exit code: {result.returncode}.\n"
        f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    )


@pytest.mark.level1
def test_fsdp_cp2_colossal_gbs2():
    """
    Feature: DeepSeek3 FSDP + Context Parallel (colossal) training
    Description: Test DeepSeek3 training with FSDP and context parallel 2 using the
                 colossal method on 4 cards, global batch size 2.
    Expectation: msrun script exits with code 0.
    """
    generate_dataset()
    with open(BASE_CONFIG, "r") as fp:
        configs = yaml.safe_load(fp)

    configs["train_dataset"]["dataloader"]["dataset_files"] = [DATASET_PATH]
    configs["training"]["local_batch_size"] = 1
    configs["parallelism"]["context_parallel"] = 2
    configs["parallelism"]["context_parallel_method"] = "colossal"
    configs["training"]["steps"] = 10

    local_config_path = f"{CUR_DIR}/ds3_fsdp_cp2_colossal_gbs2.yaml"
    with open(local_config_path, "w") as fp:
        yaml.dump(configs, fp, indent=2)

    run_script_path = os.path.join(CUR_DIR, "run_deepseek3.py")
    port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
    assert os.path.exists(
        run_script_path
    ), f"Run script not found: {run_script_path}"
    cmd = [
        "msrun",
        "--worker_num=4",
        "--local_worker_num=4",
        f"--master_port={port_id}",
        f"--log_dir={CUR_DIR}/log_fsdp_cp2_colossal_gbs2",
        "--join=True",
        f"{run_script_path}",
        "--config",
        f"{local_config_path}",
    ]
    result = subprocess.run(
        cmd, shell=False, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"DeepSeek3 training (run_deepseek3.py, config={local_config_path}) failed with non-zero exit code: "
        f"{result.returncode}.\n"
        f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    )


@pytest.mark.level1
def test_fsdp_cp2_ulysses_gbs2():
    """
    Feature: DeepSeek3 FSDP + Context Parallel (ulysses) training
    Description: Test DeepSeek3 training with FSDP and context parallel 2 using the
                 ulysses method on 4 cards, global batch size 2.
    Expectation: msrun script exits with code 0.
    """
    generate_dataset()
    with open(BASE_CONFIG, "r") as fp:
        configs = yaml.safe_load(fp)

    configs["train_dataset"]["dataloader"]["dataset_files"] = [DATASET_PATH]
    configs["training"]["local_batch_size"] = 1
    configs["parallelism"]["context_parallel"] = 2
    configs["parallelism"]["context_parallel_method"] = "ulysses"
    configs["training"]["steps"] = 10

    local_config_path = f"{CUR_DIR}/ds3_fsdp_cp2_ulysses_gbs2.yaml"
    with open(local_config_path, "w") as fp:
        yaml.dump(configs, fp, indent=2)

    run_script_path = os.path.join(CUR_DIR, "run_deepseek3.py")
    port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
    assert os.path.exists(
        run_script_path
    ), f"Run script not found: {run_script_path}"
    cmd = [
        "msrun",
        "--worker_num=4",
        "--local_worker_num=4",
        f"--master_port={port_id}",
        f"--log_dir={CUR_DIR}/log_fsdp_cp2_ulysses_gbs2",
        "--join=True",
        f"{run_script_path}",
        "--config",
        f"{local_config_path}",
    ]
    result = subprocess.run(
        cmd, shell=False, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"DeepSeek3 training (run_deepseek3.py, config={local_config_path}) failed with non-zero exit code: "
        f"{result.returncode}.\n"
        f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    )


@pytest.mark.level1
def test_fsdp_cp2_ulysses_async_gbs2():
    """
    Feature: DeepSeek3 FSDP + Async Context Parallel (ulysses) training
    Description: Test DeepSeek3 training with FSDP and context parallel 2 using the
                 ulysses method with async context parallel enabled on 4 cards,
                 global batch size 2.
    Expectation: msrun script exits with code 0.
    """
    generate_dataset()
    with open(BASE_CONFIG, "r") as fp:
        configs = yaml.safe_load(fp)

    configs["train_dataset"]["dataloader"]["dataset_files"] = [DATASET_PATH]
    configs["training"]["local_batch_size"] = 1
    configs["parallelism"]["context_parallel"] = 2
    configs["parallelism"]["context_parallel_method"] = "ulysses"
    configs["parallelism"]["context_parallel_async"] = True
    configs["training"]["steps"] = 10

    local_config_path = f"{CUR_DIR}/ds3_fsdp_cp2_ulysses_async_gbs2.yaml"
    with open(local_config_path, "w") as fp:
        yaml.dump(configs, fp, indent=2)

    run_script_path = os.path.join(CUR_DIR, "run_deepseek3.py")
    port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
    assert os.path.exists(
        run_script_path
    ), f"Run script not found: {run_script_path}"
    cmd = [
        "msrun",
        "--worker_num=4",
        "--local_worker_num=4",
        f"--master_port={port_id}",
        f"--log_dir={CUR_DIR}/log_fsdp_cp2_ulysses_async_gbs2",
        "--join=True",
        f"{run_script_path}",
        "--config",
        f"{local_config_path}",
    ]
    result = subprocess.run(
        cmd, shell=False, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"DeepSeek3 training (run_deepseek3.py, config={local_config_path}) failed with non-zero exit code: "
        f"{result.returncode}.\n"
        f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    )


@pytest.mark.level0
def test_fsdp_tp2_ep2_deredundancy_gbs2():
    """
    Feature: DeepSeek3 FSDP + Tensor Parallel + Expert Parallel with alltoall_deredundancy training
    Description: Test DeepSeek3 training with FSDP, tensor parallel 2, expert parallel 2 and
                 moe_token_dispatcher_type alltoall_deredundancy on 4 cards, global batch size 2.
    Expectation: msrun script exits with code 0.
    """
    generate_dataset()
    with open(BASE_CONFIG, "r") as fp:
        configs = yaml.safe_load(fp)

    configs["train_dataset"]["dataloader"]["dataset_files"] = [DATASET_PATH]
    configs["training"]["local_batch_size"] = 1
    configs["parallelism"]["tensor_parallel"] = 2
    configs["parallelism"]["expert_parallel"] = 2
    configs["parallelism"]["moe_token_dispatcher_type"] = "alltoall_deredundancy"
    configs["parallelism"]["npu_nums_per_device"] = 2
    configs["training"]["steps"] = 10

    local_config_path = f"{CUR_DIR}/ds3_fsdp_tp2_ep2_deredundancy_gbs2.yaml"
    with open(local_config_path, "w") as fp:
        yaml.dump(configs, fp, indent=2)

    run_script_path = os.path.join(CUR_DIR, "run_deepseek3.py")
    port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
    assert os.path.exists(
        run_script_path
    ), f"Run script not found: {run_script_path}"
    cmd = [
        "msrun",
        "--worker_num=4",
        "--local_worker_num=4",
        f"--master_port={port_id}",
        f"--log_dir={CUR_DIR}/log_fsdp_tp2_ep2_deredundancy_gbs2",
        "--join=True",
        f"{run_script_path}",
        "--config",
        f"{local_config_path}",
    ]
    result = subprocess.run(
        cmd, shell=False, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"DeepSeek3 training (run_deepseek3.py, config={local_config_path}) failed with non-zero exit code: "
        f"{result.returncode}.\n"
        f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    )
