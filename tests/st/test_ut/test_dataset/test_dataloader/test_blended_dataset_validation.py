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
"""UTs for the input validation of the blended Megatron dataset config and helpers."""
import os
import subprocess
import sys

import pytest

from mindformers.dataset.blended_datasets.blended_megatron_dataset_config import (
    BlendedMegatronDatasetConfig,
    parse_and_normalize_split,
)
from mindformers.dataset.blended_datasets.utils import get_blend_from_list

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_blend_from_list_reads_weighted_and_unweighted_blends():
    """
    Feature: get_blend_from_list
    Description: Parse a weighted blend and unweighted blends of even and odd length.
    Expectation: Weights are split from prefixes only when every pair carries a weight.
    """
    assert get_blend_from_list(["30", "p1", "70", "p2"]) == (["p1", "p2"], [30.0, 70.0])
    assert get_blend_from_list(["p1", "p2"]) == (["p1", "p2"], None)
    assert get_blend_from_list(["p1", "p2", "p3"]) == (["p1", "p2", "p3"], None)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_blend_from_list_rejects_partially_weighted_blend():
    """
    Feature: get_blend_from_list
    Description: Parse a blend where only some prefixes carry a weight.
    Expectation: ValueError instead of silently treating the weight as a dataset prefix.
    """
    with pytest.raises(ValueError, match="weight for every dataset prefix or none at all"):
        get_blend_from_list(["30", "p1", "p2", "p3"])


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_parse_and_normalize_split_rejects_extra_ratios():
    """
    Feature: parse_and_normalize_split
    Description: Parse a split string with more ratios than there are splits.
    Expectation: ValueError.
    """
    assert parse_and_normalize_split("98,1,1") == pytest.approx([0.98, 0.01, 0.01])
    with pytest.raises(ValueError, match="split must give at most 3 ratios"):
        parse_and_normalize_split("97,1,1,1")


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("kwargs, message", [
    ({"blend": (["p1"], None), "blend_per_split": [(["p1"], None), None, None]},
     "blend and blend_per_split are incompatible"),
    ({"split": "99,1,0", "blend_per_split": [(["p1"], None), None, None]},
     "split and blend_per_split are incompatible"),
    ({"blend_per_split": [(["p1"], None), None]},
     "blend_per_split must contain 3 blends"),
    ({"blend_per_split": [(["p1", "p2"], [1.0]), None, None]},
     "blend per split prefixes and weights must be equal in number"),
    ({"blend": (["p1", "p2"], [1.0]), "split": "99,1,0"},
     "blend prefixes and weights must be equal in number"),
    ({"blend": (["p1"], None)},
     "split must be provided when blend is not None"),
])
def test_config_rejects_inconsistent_blend_settings(kwargs, message):
    """
    Feature: BlendedMegatronDatasetConfig
    Description: Build the config with each inconsistent blend / split combination.
    Expectation: ValueError naming the inconsistency.
    """
    with pytest.raises(ValueError, match=message):
        BlendedMegatronDatasetConfig(random_seed=1234, sequence_length=32, **kwargs)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_config_validation_holds_under_python_optimize():
    """
    Feature: BlendedMegatronDatasetConfig
    Description: Build an invalid config in an interpreter started with -O, which strips assert statements.
    Expectation: The config is still rejected.
    """
    code = (
        "from mindformers.dataset.blended_datasets.blended_megatron_dataset_config import "
        "BlendedMegatronDatasetConfig\n"
        "try:\n"
        "    BlendedMegatronDatasetConfig(random_seed=1234, sequence_length=32, blend=(['p1'], None))\n"
        "except ValueError as e:\n"
        "    print('rejected:', e)\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")])))
    result = subprocess.run([sys.executable, "-O", "-c", code], env=env, capture_output=True, text=True,
                            timeout=600, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "rejected: split must be provided when blend is not None" in result.stdout
