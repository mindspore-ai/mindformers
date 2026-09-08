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
"""DeepSeek-V3 model coverage for combined full/select/exclude recompute."""

import json
import os
import re

import pytest

from tests.st.test_multi_cards_cases.utils import TaskType
from tests.utils.precision_utils import assert_expected_values_match
from tests.st.test_multi_cards_cases.test_pynative.test_models.test_deepseek3.utils import (
    build_case_config,
    generate_dataset,
    run_training_and_extract_losses,
    save_model_checkpoints,
)

_LEVEL_0_TASK_TIME = 480
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.FOUR_CARDS_TASK

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG = os.path.join(CUR_DIR, "pynative_ds3.yaml")
DATASET_PATH = os.path.join(CUR_DIR, "train_dataset")
CHECKPOINT_PATH = os.path.join(CUR_DIR, "deepseekv3_checkpoints")
RUN_SCRIPT_PATH = os.path.join(CUR_DIR, "run_deepseek3_recompute_probe.py")
PROBE_PREFIX = "RECOMPUTE_PROBE_JSON="

RECOMPUTE_BASE = {
    "mode": "select",
    "full_recompute_layer": ["0-3"],
    "select_module": {
        "self_attention": ["4-7"],
        "mlp": ["4-7"],
    },
}

EXCLUDE_OP = {
    "self_attention.linear_proj": ["0-1"],
    "add": ["2-3"],
    "self_attention.reshape": ["4"],
    "self_attention.cast": ["5"],
    "mlp.reshape": ["6"],
    "mlp.add": ["7"],
    ".*allgather": ["4-7"],
    ".*reducescatter": ["4-7"],
    ".*expert_counts.alltoall": ["1-7"],
    ".*input.alltoallsingle": ["1-7"],
    ".*output.alltoallsingle": ["1-7"],
}


def _extract_probe_payloads(log_dir):
    """Extract one probe payload per rank from the worker logs."""
    payloads = []
    logs = sorted(f for f in (os.path.join(log_dir, n) for n in os.listdir(log_dir))
                  if re.fullmatch(r"worker_\d+\.log", os.path.basename(f)))
    for log in logs:
        with open(log, "r", encoding="utf-8") as fh:
            matches = [json.loads(l.split(PROBE_PREFIX, 1)[1])
                       for l in fh if PROBE_PREFIX in l]
        assert len(matches) == 1, f"Expected 1 payload in {log}, got {len(matches)}"
        payloads.append(matches[0])
    assert len(payloads) == 4, f"Expected 4 payloads, got {len(payloads)}"
    return sorted(payloads, key=lambda x: x["rank"])


def _sum_matching(counts, pattern, phase):
    return sum(v[phase] for k, v in counts.items() if re.fullmatch(pattern, k))


def _assert_recompute_shape(payload):
    c = payload["counts"]
    assert c["layer0.input_layernorm"]["recompute"] > 0
    assert c["layer8.input_layernorm"]["recompute"] == 0


def _assert_excluded_targets(payload, expect_recompute):
    """Check excluded targets replay only when ``expect_recompute`` is set."""
    counts = payload["counts"]
    for key in ("layer0.self_attention.linear_proj", "layer1.self_attention.linear_proj",
                "layer2.add", "layer3.add", "layer4.self_attention.reshape",
                "layer5.self_attention.cast"):
        assert counts[key]["forward"] > 0, f"{key} not in forward"
        assert (counts[key]["recompute"] > 0) == expect_recompute, \
            f"{key} replay mismatch: expected={expect_recompute}"
    for pat in (r"layer[4-7]\..*allgather", r"layer[4-7]\..*reducescatter",
                r"layer[1-7]\..*expert_counts\.alltoall",
                r"layer[1-7]\..*input\.alltoallsingle",
                r"layer[1-7]\..*output\.alltoallsingle"):
        assert _sum_matching(counts, pat, "forward") > 0
        assert (_sum_matching(counts, pat, "recompute") > 0) == expect_recompute


def _run_variant(name, exclude_op=None):
    """Run one training variant and return its losses and probe payloads."""
    recompute = dict(RECOMPUTE_BASE)
    if exclude_op is not None:
        recompute["exclude_op"] = exclude_op
    config_path = build_case_config(
        BASE_CONFIG, os.path.join(CUR_DIR, f"ds3_recompute_{name}.yaml"),
        CHECKPOINT_PATH, DATASET_PATH,
        updates={"training": {"steps": 3},
                 "parallelism": {"tensor_parallel": 2, "expert_parallel": 2,
                                 "moe_token_dispatcher_type": "alltoall"},
                 "recompute": recompute})
    losses = run_training_and_extract_losses(
        RUN_SCRIPT_PATH, config_path, f"log_recompute_{name}", worker_num=4)
    return losses, _extract_probe_payloads(os.path.join(CUR_DIR, f"log_recompute_{name}"))


@pytest.mark.level0
def test_full_select_exclude_cell_function_tp_and_ep_comm():
    """DSv3 TP2+EP2: full(0-3)+select(4-7)+exclude Cell/TP/EP comm, loss unchanged."""
    save_model_checkpoints(BASE_CONFIG, CHECKPOINT_PATH)
    generate_dataset(DATASET_PATH)
    control_losses, control_payloads = _run_variant("control")
    exclude_losses, exclude_payloads = _run_variant("exclude", EXCLUDE_OP)

    assert_expected_values_match(exclude_losses, control_losses)
    for payload in control_payloads:
        _assert_recompute_shape(payload)
        _assert_excluded_targets(payload, expect_recompute=True)
    for payload in exclude_payloads:
        _assert_recompute_shape(payload)
        _assert_excluded_targets(payload, expect_recompute=False)

    for name in ("control", "exclude"):
        log_dir = os.path.join(CUR_DIR, f"log_recompute_{name}")
        for fname in os.listdir(log_dir):
            if re.fullmatch(r"worker_\d+\.log", fname):
                with open(os.path.join(log_dir, fname), encoding="utf-8") as fh:
                    text = fh.read()
                assert "did not match any module in the model" not in text
                assert "Set full recompute at layer 0" in text
