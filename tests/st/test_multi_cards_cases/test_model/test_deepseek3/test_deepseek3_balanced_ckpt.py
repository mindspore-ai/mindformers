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
"""Test DeepseekV3 Graph-mode balanced checkpoint saving and resuming under pipeline parallelism."""
import ast
import glob
import json
import os
import random
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from tests.st.test_multi_cards_cases.utils import TaskType

_LEVEL_0_TASK_TIME = 0
_LEVEL_1_TASK_TIME = 240
_TASK_TYPE = TaskType.EIGHT_CARDS_TASK

CUR_DIR = Path(__file__).parent.resolve()
RUN_SCRIPT = CUR_DIR / "run_deepseek3_balanced_ckpt.py"
WORK_DIR = CUR_DIR / "balanced_ckpt_runs"


def _run_training(run_name, port_id, load_path=""):
    """Run one 8-card training with msrun and return its run directory."""
    run_dir = WORK_DIR / run_name
    cmd = [
        "msrun", "--worker_num=8", "--local_worker_num=8", f"--master_port={port_id}",
        f"--log_dir={run_dir / 'log'}", "--join=True",
        str(RUN_SCRIPT), "--save_path", str(run_dir), "--load_path", load_path,
    ]
    env = os.environ.copy()
    env["MS_DEV_JIT_SYNTAX_LEVEL"] = "0"
    env["HCCL_DETERMINISTIC"] = "true"
    result = subprocess.run(cmd, cwd=CUR_DIR, env=env, check=False)
    if result.returncode != 0:
        os.system(f"grep -E 'ERROR|Error' {run_dir / 'log'}/worker_*.log -C 5 | tail -80")
    assert result.returncode == 0, f"{run_name} training failed, see {run_dir / 'log'}."
    return run_dir


def _read_tensors(ckpt_dir):
    """Map (file name, tensor name) to the raw bytes. A sharded tensor has a shard in several files."""
    tensors = {}
    for path in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                start, end = info["data_offsets"]
                f.seek(8 + header_len + start)
                tensors[(os.path.basename(path), name)] = f.read(end - start)
    return tensors


def _check_checkpoint_consistent(ckpt_dir):
    """Every pipeline stage is saved, and 'metadata.json' describes exactly what the files hold."""
    tensors = _read_tensors(ckpt_dir)
    names = {name for _, name in tensors}
    with open(os.path.join(ckpt_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    described = set(metadata["state_dict_metadata"])
    assert described == names, (
        f"'metadata.json' and the files disagree: described only {sorted(described - names)[:5]}, "
        f"on disk only {sorted(names - described)[:5]}")
    for shard_key, records in metadata["storage_data"].items():
        name = ast.literal_eval(shard_key)[0]
        for record in records:
            assert (record["file_name"], name) in tensors, \
                f"'{name}' is recorded in '{record['file_name']}', which does not hold it."

    # Both pipeline stages (decoder layers 0-1 and 2-3, the MTP layer and the output layer on the last
    # stage), with the AdamW states of every model parameter.
    model_params = {name for name in names if not name.startswith(("adam_m.", "adam_v."))}
    for prefix in ("embedding.", "decoder.layers.0.", "decoder.layers.3.", "mtp.", "output_layer."):
        assert any(name.startswith(prefix) for name in model_params), f"No '{prefix}*' parameter saved."
    for name in model_params:
        assert f"adam_m.{name}" in names and f"adam_v.{name}" in names, f"Optimizer state of '{name}' missing."
    # Gradient accumulation buffers belong to the train-one-step wrapper, not to a saved cell.
    assert not any(name.startswith("accu_grads.") for name in names)


@pytest.mark.level1
def test_dp2_mp2_pp2_ep2_balanced_save_and_resume():
    """
    Feature: Graph-mode balanced checkpoint (remove redundancy) under pipeline parallelism
    Description: Train deepseekv3 (dp=mp=pp=ep=2, one MTP layer, AdamW) for 20 steps, saving the
        model and optimizer with redundancy removal every 10 steps; then resume from the step-10
        checkpoint and train to step 20 again.
    Expectation: Both stages are saved and 'metadata.json' matches the files; the resumed run's
        step-20 checkpoint is byte-identical to the uninterrupted run's.
    """
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))

    full_dir = _run_training("full", port_id)
    step10 = full_dir / "checkpoint" / "iteration_00000010"
    _check_checkpoint_consistent(step10)

    resume_dir = _run_training("resume", port_id, load_path=str(step10))

    full_step20 = _read_tensors(full_dir / "checkpoint" / "iteration_00000020")
    resume_step20 = _read_tensors(resume_dir / "checkpoint" / "iteration_00000020")
    assert set(full_step20) == set(resume_step20), "The two step-20 checkpoints hold different shards."
    differing = sorted(key for key, data in full_step20.items() if resume_step20[key] != data)
    assert not differing, f"Resumed training diverged from the uninterrupted run: {differing[:5]}"
    # Checkpoints and logs are kept only when the case fails.
    shutil.rmtree(WORK_DIR, ignore_errors=True)
