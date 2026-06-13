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
import random
import subprocess
import yaml
import pytest

from tests.st.test_multi_cards_cases.utils import TaskType
from tests.utils.generate_dataset import generate_mindrecord_file

_LEVEL_0_TASK_TIME = 240
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.EIGHT_CARDS_TASK

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG = os.path.join(CUR_DIR, "pynative_ds3.yaml")
DATASET_PATH = os.path.join(CUR_DIR, "train_dataset")


def generate_dataset():
    """Generate mindrecord dataset for eight-cards training tests."""
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


@pytest.mark.level1
def test_fsdp_cp2_ulysses_tp2_gbs2():
    """
    Feature: DeepSeek3 FSDP + Context Parallel (ulysses) + Tensor Parallel training
    Description: Test DeepSeek3 training with FSDP, context parallel 2 using the ulysses
                 method and tensor parallel 2 on 8 cards, global batch size 2.
    Expectation: msrun script exits with code 0.
    """
    generate_dataset()
    with open(BASE_CONFIG, "r") as fp:
        configs = yaml.safe_load(fp)

    configs["train_dataset"]["dataloader"]["dataset_files"] = [DATASET_PATH]
    configs["training"]["local_batch_size"] = 1
    configs["parallelism"]["context_parallel"] = 2
    configs["model"]["use_attn_mask_compression"] = True
    configs["parallelism"]["context_parallel_method"] = "ulysses"
    configs["parallelism"]["tensor_parallel"] = 2
    configs["training"]["steps"] = 10

    local_config_path = f"{CUR_DIR}/ds3_fsdp_cp2_ulysses_tp2_gbs2.yaml"
    with open(local_config_path, "w") as fp:
        yaml.dump(configs, fp, indent=2)

    run_script_path = os.path.join(CUR_DIR, "run_deepseek3.py")
    port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
    assert os.path.exists(
        run_script_path
    ), f"Run script not found: {run_script_path}"
    cmd = [
        "msrun",
        "--worker_num=8",
        "--local_worker_num=8",
        f"--master_port={port_id}",
        f"--log_dir={CUR_DIR}/log_fsdp_cp2_ulysses_tp2_gbs2",
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
def test_fsdp_cp2_ulysses_tp2_ep2_gbs2():
    """
    Feature: DeepSeek3 FSDP + Context Parallel (ulysses) + Tensor Parallel + Expert Parallel training
    Description: Test DeepSeek3 training with FSDP, context parallel 2 using the ulysses
                 method, tensor parallel 2 and expert parallel 2 on 8 cards, global batch size 2.
    Expectation: msrun script exits with code 0.
    """
    generate_dataset()
    with open(BASE_CONFIG, "r") as fp:
        configs = yaml.safe_load(fp)

    configs["train_dataset"]["dataloader"]["dataset_files"] = [DATASET_PATH]
    configs["training"]["local_batch_size"] = 1
    configs["parallelism"]["context_parallel"] = 2
    configs["model"]["use_attn_mask_compression"] = True
    configs["parallelism"]["context_parallel_method"] = "ulysses"
    configs["parallelism"]["tensor_parallel"] = 2
    configs["parallelism"]["expert_parallel"] = 2
    configs["training"]["steps"] = 10

    local_config_path = f"{CUR_DIR}/ds3_fsdp_cp2_ulysses_tp2_ep2_gbs2.yaml"
    with open(local_config_path, "w") as fp:
        yaml.dump(configs, fp, indent=2)

    run_script_path = os.path.join(CUR_DIR, "run_deepseek3.py")
    port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
    assert os.path.exists(
        run_script_path
    ), f"Run script not found: {run_script_path}"
    cmd = [
        "msrun",
        "--worker_num=8",
        "--local_worker_num=8",
        f"--master_port={port_id}",
        f"--log_dir={CUR_DIR}/log_fsdp_cp2_ulysses_tp2_ep2_gbs2",
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
def test_fsdp_cp4_hybrid4_gbs2():
    """
    Feature: DeepSeek3 FSDP + Context Parallel (hybrid) training
    Description: Test DeepSeek3 training with FSDP and context parallel 4 using the hybrid
                 method on 8 cards, where the ulysses degree occupies 2 cards, global batch size 2.
    Expectation: msrun script exits with code 0.
    """
    generate_dataset()
    with open(BASE_CONFIG, "r") as fp:
        configs = yaml.safe_load(fp)

    configs["train_dataset"]["dataloader"]["dataset_files"] = [DATASET_PATH]
    configs["training"]["local_batch_size"] = 1
    configs["parallelism"]["context_parallel"] = 4
    configs["model"]["use_attn_mask_compression"] = True
    configs["parallelism"]["context_parallel_method"] = "hybrid"
    configs["parallelism"]["ulysses_degree_in_cp"] = 2
    configs["training"]["steps"] = 10

    local_config_path = f"{CUR_DIR}/ds3_fsdp_cp4_hybrid4_gbs2.yaml"
    with open(local_config_path, "w") as fp:
        yaml.dump(configs, fp, indent=2)

    run_script_path = os.path.join(CUR_DIR, "run_deepseek3.py")
    port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
    assert os.path.exists(
        run_script_path
    ), f"Run script not found: {run_script_path}"
    cmd = [
        "msrun",
        "--worker_num=8",
        "--local_worker_num=8",
        f"--master_port={port_id}",
        f"--log_dir={CUR_DIR}/log_fsdp_cp4_hybrid4_gbs2",
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
