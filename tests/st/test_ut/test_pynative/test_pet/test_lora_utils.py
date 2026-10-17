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
# ==========================================================================
"""Tests for deterministic PyNative LoRA initialization."""

import numpy as np
import pytest

from mindspore import nn, Parameter, Tensor

from mindformers.pynative.pet.utils import is_lora_a_param_name, reinitialize_lora_parameters


class _LoRAModel(nn.Cell):
    """Minimal model exposing named LoRA A/B parameters."""

    def __init__(self):
        super().__init__()
        self.lora_a = Parameter(Tensor(np.zeros((3, 5), np.float32)), name="block.lora_a")
        self.lora_b = Parameter(Tensor(np.ones((7, 3), np.float32)), name="block.lora_b")


class _ParameterView:
    """Expose a chosen parameter subset to emulate one pipeline stage."""

    def __init__(self, parameters):
        self.parameters = parameters

    def parameters_and_names(self):
        return ((parameter.name, parameter) for parameter in self.parameters)


def test_is_lora_a_param_name_supports_all_conventions():
    """LoRA A classification is shared by initialization and FSDP policy."""
    assert is_lora_a_param_name("lora_a")
    assert is_lora_a_param_name("block.lora_a")
    assert is_lora_a_param_name("block_lora_a")
    assert not is_lora_a_param_name("block.lora_b")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_reinitialize_lora_parameters_is_name_and_seed_deterministic():
    """The same seed/name pair must reproduce A, while zero-std B remains zero."""
    first = _LoRAModel()
    second = _LoRAModel()

    first_count = reinitialize_lora_parameters(first, seed=42, lora_a_std=0.01, lora_b_std=0.0)
    second_count = reinitialize_lora_parameters(second, seed=42, lora_a_std=0.01, lora_b_std=0.0)

    assert first_count == second_count == 2
    np.testing.assert_array_equal(first.lora_a.asnumpy(), second.lora_a.asnumpy())
    np.testing.assert_array_equal(first.lora_b.asnumpy(), np.zeros((7, 3), np.float32))

    reinitialize_lora_parameters(second, seed=43, lora_a_std=0.01, lora_b_std=0.0)
    assert not np.array_equal(first.lora_a.asnumpy(), second.lora_a.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_reinitialize_lora_parameters_is_pipeline_partition_independent():
    """A full model and pipeline-stage subsets must initialize matching names equally."""
    full_model = _LoRAModel()
    stage_lora_a = Parameter(
        Tensor(np.ones((3, 5), np.float32)), name="block.lora_a"
    )
    stage_lora_b = Parameter(
        Tensor(np.ones((7, 3), np.float32)), name="block.lora_b"
    )

    reinitialize_lora_parameters(full_model, seed=42, lora_a_std=0.01, lora_b_std=0.0)
    reinitialize_lora_parameters(
        [_ParameterView([stage_lora_b]), _ParameterView([stage_lora_a])],
        seed=42,
        lora_a_std=0.01,
        lora_b_std=0.0,
    )

    np.testing.assert_array_equal(full_model.lora_a.asnumpy(), stage_lora_a.asnumpy())
    np.testing.assert_array_equal(full_model.lora_b.asnumpy(), stage_lora_b.asnumpy())
