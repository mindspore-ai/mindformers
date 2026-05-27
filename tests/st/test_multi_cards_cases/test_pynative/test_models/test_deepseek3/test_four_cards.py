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
BASE_CONFIG = os.path.join(CUR_DIR, "pynarive_ds3.yaml")
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
        f"PrepareModuleOO script failed with non-zero exit code: "
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
        f"PrepareModuleOO script failed with non-zero exit code: "
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
        f"PrepareModuleOO script failed with non-zero exit code: "
        f"{result.returncode}.\n"
        f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    )
