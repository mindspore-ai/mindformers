#!/usr/bin/env python3
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
"""Tests for expert parallel in pynative mode."""

import pytest
import mindspore as ms
from mindspore import context, nn

import mindspore as ms

from hyper_parallel.core.device_mesh import init_device_mesh
from hyper_parallel.core.placement_types import Shard

from mindformers.pynative.distributed import utils
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.layers.linear import Linear
from mindformers.pynative.transformers.mlp import MLP, MLPSubmodules
from mindformers.pynative.transformers.moe.experts import GroupedMLP
from mindformers.pynative.distributed.expert_parallel import ExpertParallel
import pytest
HIDDEN_SIZE = 32
EXPERT_NUM = 4

@pytest.fixture(name="device_mesh")
def fixture_device_mesh():
    """Provide a lightweight fake device mesh."""
    return object()

class TestExpertParallel:
    """Tests for ExpertParallel."""

    def setup_method(self):
        context.set_context(mode=context.PYNATIVE_MODE)

        # self.device_mesh = init_device_mesh(
        #     device_type="npu",
        #     mesh_shape=(2,),
        #     mesh_dim_names=("ep",)
        # )
        self.device_mesh = object()
        self.config = TransformerConfig(
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=4,
            num_layers=1,
            hidden_act="fusedswiglu",
            num_moe_experts=EXPERT_NUM,
            add_bias_linear=False,
            # MoE specific configs
            moe_ffn_hidden_size=HIDDEN_SIZE * 4, # Standard MLP expansion
            moe_apply_probs_on_input=False,
            gated_linear_unit=True,
        )
        self.set_expert_parallel()

    def set_expert_parallel(self):
        self.expert_parallel = ExpertParallel()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_module_valid(self):
        # Pass in valid module GroupedMLP
        module = self.expert_parallel._apply(GroupedMLP(self.config), self.device_mesh)

        assert isinstance(module, GroupedMLP)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_module_invalid(self):
        # Pass in invalid module MLP
        mlp = MLP(submodules=MLPSubmodules(linear_fc1=Linear, linear_fc2=Linear,),
                  config=self.config,
                  input_size=HIDDEN_SIZE)
        with pytest.raises(TypeError):
            self.expert_parallel._apply(mlp, self.device_mesh)


    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_parameter_sharding_plan(self, monkeypatch):
        captured = {}
        def _fake_shard_module(module, device_mesh, parameter_shard_plan):
            captured["module"] = module
            captured["device_mesh"] = device_mesh
            captured["sharding_plan"] = parameter_shard_plan
            # No actual partitioning is performed to avoid introducing communication/distributed dependencies.
            return module

        monkeypatch.setattr(utils, "shard_module", _fake_shard_module)

        module = self.expert_parallel._apply(GroupedMLP(self.config), self.device_mesh)
        sharding_plan = captured["sharding_plan"]
        plan = sharding_plan.plan

        # expect：weight1/weight2 is Shard(0)
        assert set(plan.keys()) == {"weight1", "weight2"}
        assert plan["weight1"][0] == Shard(0)
        assert plan["weight2"][0] == Shard(0)

