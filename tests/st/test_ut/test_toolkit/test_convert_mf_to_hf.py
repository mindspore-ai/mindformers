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
"""UTs for the YAML handling of toolkit/weight_convert/convert_mf_to_hf.py."""
import importlib.util
import os

import pytest

from mindformers.tools.check_rules import YAML_MAX_NESTING_DEPTH

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CONVERTER = os.path.join(REPO_ROOT, "toolkit", "weight_convert", "convert_mf_to_hf.py")


def _load_converter():
    """Import the converter script as a module; it is not part of an installed package."""
    spec = importlib.util.spec_from_file_location("convert_mf_to_hf", CONVERTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nested_mapping_yaml(depth):
    """Return a YAML document whose mappings nest ``depth`` levels deep."""
    lines = [f"{'  ' * level}level{level}:" for level in range(depth)]
    lines[-1] += " leaf"
    return "\n".join(lines) + "\n"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_parse_yaml_config_rejects_overly_nested_yaml(tmp_path):
    """
    Feature: convert_mf_to_hf.parse_yaml_config
    Description: Parse a YAML file nested one level deeper than YAML_MAX_NESTING_DEPTH.
    Expectation: The depth check rejects it with ValueError before yaml.safe_load builds it.
    """
    converter = _load_converter()
    yaml_path = tmp_path / "too_deep.yaml"
    yaml_path.write_text(_nested_mapping_yaml(YAML_MAX_NESTING_DEPTH + 1), encoding="utf-8")

    with pytest.raises(ValueError, match="YAML nesting depth"):
        converter.parse_yaml_config(str(yaml_path))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_parse_yaml_config_reads_model_section(tmp_path):
    """
    Feature: convert_mf_to_hf.parse_yaml_config
    Description: Parse an ordinary MindFormers YAML after the depth check has consumed the file once.
    Expectation: The model section is read in full (the file is rewound before yaml.safe_load).
    """
    converter = _load_converter()
    yaml_path = tmp_path / "model.yaml"
    yaml_path.write_text(
        "model:\n"
        "  model_type: deepseek_v3\n"
        "  num_hidden_layers: 4\n"
        "  n_routed_experts: 8\n"
        "  compute_dtype: float16\n",
        encoding="utf-8",
    )

    config = converter.parse_yaml_config(str(yaml_path))

    assert config["model_type"] == "deepseek_v3"
    assert config["num_layers"] == 4
    assert config["n_routed_experts"] == 8
    assert config["compute_dtype"] == "float16"
