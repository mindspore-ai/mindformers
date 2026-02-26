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
"""Tests for distributed utils in pynative mode."""

import pytest
import mindspore as ms
from mindspore import context, nn

from hyper_parallel.core.placement_types import Replicate

from mindformers.pynative.distributed import utils as dist_utils


@pytest.fixture(name="device_mesh")
def fixture_device_mesh():
    """Provide a lightweight fake device mesh."""
    return object()


class InnerCell(nn.Cell):
    """Inner cell with parameters for shard plan tests."""
    def __init__(self):
        super().__init__()
        self.b = ms.Parameter(ms.Tensor([1.0], ms.float32), name="b")
        self.c = ms.Parameter(ms.Tensor([2.0], ms.float32), name="c", requires_grad=False)

    def construct(self, x):
        return x


class OuterCell(nn.Cell):
    """Outer cell that nests InnerCell to form a.b parameter names."""
    def __init__(self):
        super().__init__()
        self.a = InnerCell()
        self.d = ms.Parameter(ms.Tensor([3.0], ms.float32), name="d")

    def construct(self, x):
        return x


class TestDistributeModule:
    """Tests for distribute_module utility."""

    def setup_method(self):
        context.set_context(mode=context.PYNATIVE_MODE)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default_shard_plan_and_submodule_names(self, monkeypatch, device_mesh):
        """
        Feature: distribute_module
        Description: Verify default shard plan fill-in for submodule params and skip requires_grad=False.
        Expectation: Replicate plan added for a.b, a.c skipped, existing plan preserved.
        """
        module = OuterCell()
        captured = {}
        logger_calls = []

        def fake_shard_module(module_unused, device_mesh_unused, shard_plan):
            _ = (module_unused, device_mesh_unused)
            captured["shard_plan"] = shard_plan.plan

        def fake_logger_info(msg, *args, **kwargs):
            _ = (args, kwargs)
            logger_calls.append(msg)

        monkeypatch.setattr(dist_utils, "shard_module", fake_shard_module)
        monkeypatch.setattr(dist_utils.logger, "info", fake_logger_info)

        custom_plan = {"d": ("custom",)}
        dist_utils.distribute_module(module, device_mesh, parameter_shard_plan=custom_plan)

        shard_plan = captured["shard_plan"]
        assert shard_plan["d"] == ("custom",)
        assert "a.b" in shard_plan
        assert shard_plan["a.b"][0] == Replicate()
        assert "a.c" not in shard_plan
        assert getattr(module, "_distribute_module_applied") is True

        # Verify logger.info was called for a.b
        assert any("Add replicate plan for a.b" in msg for msg in logger_calls)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_distribute_module_only_once(self, monkeypatch, device_mesh):
        """
        Feature: distribute_module
        Description: Ensure repeated calls raise RuntimeError.
        Expectation: RuntimeError is raised on second call.
        """
        module = OuterCell()
        monkeypatch.setattr(dist_utils, "shard_module", lambda *args, **kwargs: None)

        dist_utils.distribute_module(module, device_mesh)
        with pytest.raises(RuntimeError):
            dist_utils.distribute_module(module, device_mesh)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_input_fn_arg_counts(self, monkeypatch, device_mesh):
        """
        Feature: distribute_module
        Description: Validate input_fn argument counts for hook registration.
        Expectation: 3/4 args are accepted, other counts raise ValueError.
        """
        monkeypatch.setattr(dist_utils, "shard_module", lambda *args, **kwargs: None)

        def input_fn_3(device_mesh, module, args):
            _ = (device_mesh, module, args)

        def input_fn_4(device_mesh, module, args, kwargs):
            _ = (device_mesh, module, args, kwargs)

        dist_utils.distribute_module(OuterCell(), device_mesh, input_fn=input_fn_3)
        dist_utils.distribute_module(OuterCell(), device_mesh, input_fn=input_fn_4)

        def input_fn_bad(device_mesh, module):
            _ = (device_mesh, module)

        with pytest.raises(ValueError):
            dist_utils.distribute_module(OuterCell(), device_mesh, input_fn=input_fn_bad)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_output_fn_arg_counts(self, monkeypatch, device_mesh):
        """
        Feature: distribute_module
        Description: Validate output_fn argument counts for hook registration.
        Expectation: 4/5 args are accepted, other counts raise ValueError.
        """
        monkeypatch.setattr(dist_utils, "shard_module", lambda *args, **kwargs: None)

        def output_fn_4(device_mesh, module, args, output):
            _ = (device_mesh, module, args, output)

        def output_fn_5(device_mesh, module, args, kwargs, output):
            _ = (device_mesh, module, args, kwargs, output)

        dist_utils.distribute_module(OuterCell(), device_mesh, output_fn=output_fn_4)
        dist_utils.distribute_module(OuterCell(), device_mesh, output_fn=output_fn_5)

        def output_fn_bad(device_mesh, module, output):
            _ = (device_mesh, module, output)

        with pytest.raises(ValueError):
            dist_utils.distribute_module(OuterCell(), device_mesh, output_fn=output_fn_bad)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_input_output_hooks_effective(self, monkeypatch, device_mesh):
        """
        Feature: distribute_module
        Description: Verify input/output hooks are executed in forward.
        Expectation: Pre-hook and post-hook are both called with expected args.
        """
        monkeypatch.setattr(dist_utils, "shard_module", lambda *args, **kwargs: None)
        module = OuterCell()
        calls = []

        def input_fn_4(device_mesh_unused, module_unused, args, kwargs):
            _ = (device_mesh_unused, module_unused)
            calls.append(("pre", args, kwargs))

        def output_fn_5(device_mesh_unused, module_unused, args, kwargs, output):
            _ = (device_mesh_unused, module_unused)
            calls.append(("post", args, kwargs, output))

        dist_utils.distribute_module(
            module,
            device_mesh,
            input_fn=input_fn_4,
            output_fn=output_fn_5,
        )
        x = ms.Tensor([1.0], ms.float32)
        output = module(x)

        assert len(calls) == 2
        assert calls[0][0] == "pre"
        assert calls[0][1] == (x,)
        assert calls[0][2] == {}
        assert calls[1][0] == "post"
        assert calls[1][1] == (x,)
        assert calls[1][2] == {}
        assert calls[1][3] is output

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_input_output_hooks_without_kwargs(self, monkeypatch, device_mesh):
        """
        Feature: distribute_module
        Description: Verify hooks without kwargs are executed in forward.
        Expectation: Pre-hook and post-hook are both called with expected args.
        """
        monkeypatch.setattr(dist_utils, "shard_module", lambda *args, **kwargs: None)
        module = OuterCell()
        calls = []

        def input_fn_3(device_mesh_unused, module_unused, args):
            _ = (device_mesh_unused, module_unused)
            calls.append(("pre", args))

        def output_fn_4(device_mesh_unused, module_unused, args, output):
            _ = (device_mesh_unused, module_unused)
            calls.append(("post", args, output))

        dist_utils.distribute_module(
            module,
            device_mesh,
            input_fn=input_fn_3,
            output_fn=output_fn_4,
        )
        x = ms.Tensor([2.0], ms.float32)
        output = module(x)

        assert len(calls) == 2
        assert calls[0] == ("pre", (x,))
        assert calls[1][0] == "post"
        assert calls[1][1] == (x,)
        assert calls[1][2] is output
