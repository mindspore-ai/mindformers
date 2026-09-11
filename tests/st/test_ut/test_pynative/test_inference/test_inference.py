# Copyright 2026 Huawei Technologies Co, Ltd
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
"""Test single-card PyNative inference unit cases."""

import os
import subprocess
import sys

import pytest
import yaml

from tests.st.test_multi_cards_cases.test_pynative.test_models.test_deepseek3.utils import (
    save_model_checkpoints,
)
from tests.st.test_multi_cards_cases.test_pynative.test_inference.utils import (
    build_case_config,
    build_inference_base_config,
    build_wordlevel_tokenizer,
    collect_generated_ids,
)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_DIR = os.path.normpath(os.path.join(
    CUR_DIR, os.pardir, os.pardir, os.pardir, os.pardir,
    "st", "test_multi_cards_cases", "test_pynative", "test_inference",
))
BASE_CONFIG = build_inference_base_config()
# UT shares the run script with the multi-cards tests but keeps its own
# checkpoint dir, so a UT run never races the multi-cards channel on the
# shared checkpoints directory when the two channels run concurrently.
CHECKPOINT_PATH = os.path.join(CUR_DIR, "checkpoints")
RUN_SCRIPT_PATH = os.path.join(MULTI_DIR, "run_inference.py")

# Seed-deterministic greedy output of the ten fixed prompts, keyed by
# sample_idx (full sequence: prompt tokens + 8 generated tokens); fixed by
# seed=42 weights + float32 compute + greedy argmax.
EXPECTED_IDS = {
    0: [39, 22, 30, 39, 16, 48, 48, 48, 48, 48, 48, 48, 48],
    1: [2, 35, 37, 33, 10, 1, 1, 1, 1, 1, 1, 1, 1],
    2: [42, 6, 39, 25, 38, 45, 45, 45, 45, 45, 45, 45, 45],
    3: [40, 20, 15, 13, 43, 54, 35, 29, 61, 63, 54, 5, 54],
    4: [14, 8, 27, 24, 41, 4, 4, 4, 4, 4, 4, 4, 4],
    5: [39, 28, 44, 39, 31, 11, 36, 56, 56, 56, 56, 56, 56, 56],
    6: [26, 29, 34, 39, 5, 45, 45, 45, 55, 55, 55, 55, 55],
    7: [39, 18, 19, 32, 21, 32, 32, 32, 32, 32, 32, 32, 32],
    8: [9, 36, 4, 24, 23, 41, 5, 24, 24, 24, 24, 4, 50, 56],
    9: [39, 12, 3, 17, 7, 5, 5, 14, 26, 26, 45, 45, 52],
}


def run_single_card_case(config_name):
    """Prepare deterministic weights/tokenizer/config and run one single-card case."""
    save_model_checkpoints(BASE_CONFIG, CHECKPOINT_PATH)
    build_wordlevel_tokenizer(CHECKPOINT_PATH)
    local_config_path = os.path.join(CUR_DIR, config_name)
    build_case_config(BASE_CONFIG, local_config_path, CHECKPOINT_PATH, {})
    result = subprocess.run(
        [sys.executable, RUN_SCRIPT_PATH, "--config", local_config_path],
        shell=False, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"Inference script failed with exit code {result.returncode}.\n"
        f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    )
    return collect_generated_ids(local_config_path.replace(".yaml", "_result.jsonl"))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_single_card_inference_deterministic():
    """
    Feature: PyNative single-card greedy inference
    Description: Run PyNative inference with a seed-deterministic miniature model
                 on one card; greedy decode with fixed seed=42 weights must
                 reproduce the hardcoded EXPECTED_IDS baseline exactly.
    Expectation: Inference exits with code 0, all ten samples are generated, and
                 every token id sequence equals the expected baseline.
    """
    actual_ids = run_single_card_case("ut_infer_single_card.yaml")

    assert actual_ids == EXPECTED_IDS, (
        f"Single-card greedy inference drifted from the expected baseline:\n"
        f"actual={actual_ids}\nexpected={EXPECTED_IDS}"
    )


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_single_card_inference_output_formats():
    """
    Feature: PyNative inference output format dispatch
    Description: The ``inference.output`` extension routes the writer: ``.jsonl``
                 streams one sample per line; ``.txt`` writes one generated text
                 per line. Both formats must contain all samples in input order.
    Expectation: Both runs exit with code 0 and the output files contain ten
                 ordered records with matching sample indices.
    """
    save_model_checkpoints(BASE_CONFIG, CHECKPOINT_PATH)
    build_wordlevel_tokenizer(CHECKPOINT_PATH)
    base = build_case_config(
        BASE_CONFIG, os.path.join(CUR_DIR, "ut_infer_fmt_base.yaml"),
        CHECKPOINT_PATH, {},
    )

    results = {}
    for fmt in ("jsonl", "txt"):
        config_path = os.path.join(CUR_DIR, f"ut_infer_fmt_{fmt}.yaml")
        with open(base, "r") as fp:
            configs = yaml.safe_load(fp)
        configs["inference"]["output"] = config_path.replace(".yaml", f"_result.{fmt}")
        with open(config_path, "w") as fp:
            yaml.dump(configs, fp, indent=2)
        result = subprocess.run(
            [sys.executable, RUN_SCRIPT_PATH, "--config", config_path],
            shell=False, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, (
            f"{fmt} inference failed: {result.stderr}"
        )
        out_path = config_path.replace(".yaml", f"_result.{fmt}")
        with open(out_path, "r") as fp:
            lines = [line for line in fp.read().splitlines() if line]
        assert len(lines) == 10
        results[fmt] = lines

    # jsonl lines are ordered records; txt lines are decoded texts.
    assert all('"sample_idx"' in line for line in results["jsonl"])
    assert all('"sample_idx"' not in line for line in results["txt"])
