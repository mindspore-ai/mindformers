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
"""test checkpoint_health_monitor in resume_ckpt.py"""
import json
import os

import pytest

from mindformers.tools.resume_ckpt import checkpoint_health_monitor

CKPTS = [f"llama_rank_0-1_{step}.ckpt" for step in (2, 4, 6)]


def _write_health_json(record_dir, health_by_ckpt):
    """Write health_ckpts.json the way CheckpointMonitor does on rank 0."""
    records = [{"is_health": is_health, "ckpt_name": name} for name, is_health in health_by_ckpt.items()]
    with open(os.path.join(record_dir, "health_ckpts.json"), "w", encoding="utf-8") as f:
        json.dump(records, f)


def _resume_list(ckpt_dir):
    return [os.path.join(ckpt_dir, "rank_0", name) for name in CKPTS]


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unhealthy_checkpoints_are_removed(tmp_path):
    """
    Feature: checkpoint_health_monitor.
    Description: health_ckpts.json marks the newest checkpoint unhealthy.
    Expectation: it is dropped, so resume picks the newest healthy checkpoint.
    """
    _write_health_json(tmp_path, {CKPTS[0]: 0, CKPTS[1]: 0, CKPTS[2]: 1})
    result = checkpoint_health_monitor(str(tmp_path), _resume_list(tmp_path))
    assert [os.path.basename(p) for p in result] == CKPTS[:2]


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_all_healthy_checkpoints_are_kept(tmp_path):
    """
    Feature: checkpoint_health_monitor.
    Description: every recorded checkpoint is healthy.
    Expectation: the resume list is returned unchanged.
    """
    _write_health_json(tmp_path, {name: 0 for name in CKPTS})
    resume_list = _resume_list(tmp_path)
    assert checkpoint_health_monitor(str(tmp_path), resume_list) == resume_list


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_no_healthy_resumable_checkpoint_raises(tmp_path):
    """
    Feature: checkpoint_health_monitor.
    Description: the only healthy record is not among the resumable checkpoints.
    Expectation: ValueError instead of an empty resume list.
    """
    _write_health_json(tmp_path, {"llama_rank_0-1_0.ckpt": 0, **{name: 1 for name in CKPTS}})
    with pytest.raises(ValueError, match="unhealthy"):
        checkpoint_health_monitor(str(tmp_path), _resume_list(tmp_path))


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("records", [None, [], [{"is_health": 1, "ckpt_name": name} for name in CKPTS]],
                         ids=["no_json", "empty_json", "all_unhealthy"])
def test_no_healthy_record_raises(tmp_path, records):
    """
    Feature: checkpoint_health_monitor.
    Description: health_ckpts.json is missing, empty, or marks every checkpoint unhealthy.
    Expectation: ValueError, as before this fix; no unhealthy checkpoint is returned for resume.
    """
    if records is not None:
        with open(os.path.join(tmp_path, "health_ckpts.json"), "w", encoding="utf-8") as f:
            json.dump(records, f)
    with pytest.raises(ValueError, match="no healthy checkpoints"):
        checkpoint_health_monitor(str(tmp_path), _resume_list(tmp_path))
