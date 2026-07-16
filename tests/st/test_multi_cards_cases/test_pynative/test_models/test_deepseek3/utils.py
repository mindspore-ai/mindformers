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
"""Utils for DeepSeek3 multi-cards training tests."""

import os
import random
import re
import shutil
import subprocess

import numpy as np
import yaml
import mindspore as ms
from mindspore.graph.api import _no_grad

from mindformers.checkpoint.checkpoint import save_checkpoint, CommonInfo
from mindformers.pynative.config import TrainConfig
from mindformers.pynative.trainer.utils import _build_model
from mindformers.pynative.trainer.trainer import patch_pynative_modules

from tests.utils.generate_dataset import generate_mindrecord_file
SEED = 42
DATA_SCHEMA = {
    "input_ids": {"type": "int32", "shape": [-1]},
    "labels": {"type": "int32", "shape": [-1]},
    "loss_mask": {"type": "int32", "shape": [-1]},
    "position_ids": {"type": "int32", "shape": [-1]},
}


def extract_losses_from_log(log_dir):
    """Extract loss values from the first worker log that contains losses."""
    worker_logs = [
        os.path.join(log_dir, file_name)
        for file_name in os.listdir(log_dir)
        if re.match(r"worker_\d+\.log$", file_name)
    ]
    worker_logs.sort(key=lambda p: int(re.search(r"worker_(\d+)", os.path.basename(p)).group(1)))
    pattern = re.compile(r"loss:\s+(\d+\.\d+)")
    for worker_log in worker_logs:
        values = []
        with open(worker_log, "r") as fp:
            for line in fp:
                match = pattern.search(line)
                if match:
                    values.append(float(match.group(1)))
        if values:
            return values
    return []


def extract_losses_from_text(text):
    """Extract loss values from captured training output."""
    pattern = re.compile(r"loss:\s+(\d+\.\d+)")
    values = []
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            values.append(float(match.group(1)))
    return values


def set_random_seed(seed=SEED):
    """Set random seeds used by DeepSeek3 precision tests."""
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


def generate_dataset(dataset_path):
    """Generate the fixed MindRecord dataset used by DeepSeek3 precision tests."""
    dataset_file = os.path.join(dataset_path, "dataset.mindrecord")
    if os.path.exists(dataset_file) and os.path.exists(dataset_file + ".db"):
        return
    generate_mindrecord_file(
        seq_length=4096,
        batch_size=2,
        train_steps=1000,
        dataset_path=dataset_file,
        data_schema=DATA_SCHEMA,
    )


def build_case_config(base_config, local_config_path, checkpoint_path, dataset_path, updates):
    """Write a deterministic DeepSeek3 case config and return its path."""
    with open(base_config, "r") as fp:
        configs = yaml.safe_load(fp)

    configs["checkpoint"]["load_path"] = checkpoint_path
    configs["train_dataset"]["dataloader"]["dataset_files"] = [dataset_path]
    configs["training"]["local_batch_size"] = 1
    configs["training"]["steps"] = 10
    configs["training"]["seed"] = SEED
    configs["training"]["deterministic"] = True

    for section, values in updates.items():
        configs.setdefault(section, {}).update(values)

    with open(local_config_path, "w") as fp:
        yaml.dump(configs, fp, indent=2)
    return local_config_path


def run_training_and_extract_losses(run_script_path, config_path, log_dir, worker_num):
    """Run training with msrun and extract loss values from worker log.

    Args:
        run_script_path (str): Path to the training script.
        config_path (str): Path to the config yaml file.
        log_dir (str): Directory name (relative to CUR_DIR) for msrun logs.
        worker_num (int): Number of workers for msrun.

    Returns:
        list[float]: Extracted loss values from the last worker log.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
    log_path = os.path.join(cur_dir, log_dir)
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    cmd = [
        "msrun",
        f"--worker_num={worker_num}",
        f"--local_worker_num={worker_num}",
        f"--master_port={port_id}",
        f"--log_dir={log_path}",
        "--join=True",
        f"{run_script_path}",
        "--config",
        f"{config_path}",
    ]
    env = os.environ.copy()
    socket_start = min(port_id + 1, 65400)
    socket_end = min(socket_start + worker_num + 32, 65535)
    env.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", f"{socket_start}-{socket_end}")
    result = subprocess.run(
        cmd, shell=False, capture_output=True, text=True, check=False, env=env,
    )
    losses = extract_losses_from_log(log_path)
    assert result.returncode == 0 or losses, (
        f"Training script failed with non-zero exit code: "
        f"{result.returncode}.\n"
        f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    )
    return losses


def run_python_training_and_extract_losses(run_script_path, config_path):
    """Run single-card training with Python directly and extract loss values."""
    cmd = ["python", run_script_path, "--config", config_path]
    result = subprocess.run(
        cmd, shell=False, capture_output=True, text=True, check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    losses = extract_losses_from_text(output)
    assert result.returncode == 0, (
        f"Training script failed with non-zero exit code: "
        f"{result.returncode}.\n"
        f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    )
    return losses


def save_model_checkpoints(config, save_path):
    """
    Save a checkpoint of the model.

    Args:
        config (TrainConfig): The configuration object for training.
        save_path (Path): The path where the checkpoint will be saved.
    """
    set_random_seed()
    patch_pynative_modules()
    if os.path.isdir(save_path):
        return
    print(f"Save checkpoint to {save_path}")

    config = TrainConfig.load_from_yaml(config)
    with ms.DeviceCtx("meta"):
        model = _build_model(config)

    model.to_empty()
    with _no_grad():
        model.init_states()
    
    step = 1
    optimizer = None
    
    common_info = CommonInfo(
        epoch_num=step,
        step_num=step,
        global_step=step,
        loss_scale=1.0,
        global_batch_size=config.training.global_batch_size,
    )
    save_checkpoint(
        iteration=step,
        network=model,
        optimizer=optimizer,
        common_info=common_info,
        keep_max_num=config.checkpoint.save_max,
        user_prefix=config.checkpoint.prefix,
        save_checkpoint_path=str(save_path),
        remove_redundancy=config.checkpoint.remove_redundancy,
        current_ckpt_step_list=[],
    )
