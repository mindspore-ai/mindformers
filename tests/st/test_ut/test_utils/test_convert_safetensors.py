# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Safetensors conversion tests with non-legacy configuration fields."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.utils.safetensors.convert_safetensors import (
    _check_valid_input,
    convert_hf_safetensors_multiprocess,
)


class ConvertModel(PreTrainedModel):
    """Minimal model converter using the current conversion protocol."""

    weight_mapping = [("hf.", "model.")]

    @classmethod
    def convert_weight_dict(cls, source_dict, **kwargs):
        """Convert parameter names while preserving tensor values."""
        return {name.replace("hf.", "model."): value for name, value in source_dict.items()}

    @classmethod
    def convert_map_dict(cls, source_dict, **kwargs):
        """Convert names in a sharded checkpoint index."""
        return cls.convert_weight_dict(source_dict, **kwargs)


@pytest.fixture(name="conversion_inputs")
def fixture_conversion_inputs(tmp_path):
    """Prepare a local shard and a configuration with only non-legacy fields."""
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "converted"
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    save_file({"hf.weight": values}, str(source / "model.safetensors"))
    config = SimpleNamespace(num_attention_heads=2, num_key_value_heads=1,
                             hidden_size=4, qkv_concat=False)
    model = ConvertModel(PretrainedConfig())
    return source, destination, config, values, model


@pytest.mark.parametrize("with_index", [False, True])
def test_convert_nonlegacy_checkpoint(conversion_inputs, with_index):
    """Convert real safetensors files and optional index without a legacy selector."""
    source, destination, config, values, model = conversion_inputs
    if with_index:
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"hf.weight": "model.safetensors"}}), encoding="utf-8")
    convert_hf_safetensors_multiprocess(str(source), str(destination), model, config)
    converted = load_file(str(destination / "model.safetensors"))
    np.testing.assert_array_equal(converted["model.weight"], values)
    if with_index:
        index = json.loads((destination / "param_name_map.json").read_text(encoding="utf-8"))
        assert index == {"model.weight": "model.safetensors"}


@pytest.mark.parametrize("field,value,error", [
    ("num_attention_heads", "2", "num_attention_heads"),
    ("num_key_value_heads", "1", "kv_heads"),
    ("hidden_size", "4", "hidden"),
    ("qkv_concat", "False", "is_qkv_concat"),
])
def test_reject_invalid_nonlegacy_config(conversion_inputs, field, value, error):
    """Validate current configuration fields before starting conversion workers."""
    source, destination, config, _, model = conversion_inputs
    setattr(config, field, value)
    with pytest.raises(ValueError, match=error):
        _check_valid_input(str(source), str(destination), model, config)
