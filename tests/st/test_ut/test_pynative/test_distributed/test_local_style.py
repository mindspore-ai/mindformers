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
"""Unit tests for local-tensor parallel styles."""

import pytest
from mindspore import Tensor, nn

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from mindformers.pynative.distributed import style as style_module
from mindformers.pynative.distributed.style import (
    AllGather,
    ColwiseParallel,
    NoParallel,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
    ShardTensor,
    build_hp_async_cp_style,
    build_hp_cp_style,
)
from mindformers.pynative.layers.linear import Linear


class _FakeMesh:
    """Two-rank TP mesh with this process acting as rank 1."""

    @staticmethod
    def size():
        return 2

    @staticmethod
    def get_group():
        return "tp-group"

    @staticmethod
    def get_local_rank():
        return 1


class _Identity(nn.Cell):
    def construct(self, value):
        return value


class _FakeHPStyle:
    """Capture Hyper-Parallel CP constructor arguments and apply calls."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def apply(self, module, device_mesh, **kwargs):
        self.applied = (module, device_mesh, kwargs)
        return module


@pytest.fixture(name="mock_local_runtime")
def fixture_mock_local_runtime(monkeypatch):
    """Avoid parameter conversion and emulate two-rank collectives on CPU."""
    plans = []

    def fake_distribute_module(module, device_mesh, parameter_shard_plan, input_fn, output_fn):
        del device_mesh, input_fn, output_fn
        plans.append(parameter_shard_plan)
        return module

    def fake_all_gather(unused_output, value, group):
        del unused_output, group
        return style_module.mint.cat((value, value), dim=0), None

    def fake_reduce_scatter(unused_output, value, group):
        del unused_output, group
        return value[: value.shape[0] // 2], None

    def fake_all_reduce(value, group):
        del group
        return value * 2, None

    monkeypatch.setattr(style_module, "distribute_module", fake_distribute_module)
    monkeypatch.setattr(style_module.comm_func, "all_gather_into_tensor", fake_all_gather)
    monkeypatch.setattr(style_module.comm_func, "reduce_scatter_tensor", fake_reduce_scatter)
    monkeypatch.setattr(style_module.comm_func, "all_reduce", fake_all_reduce)
    return plans


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_local_transforms_cover_dims_and_validation(mock_local_runtime):
    """AllGather and ShardTensor support arbitrary valid dimensions."""
    del mock_local_runtime
    mesh = _FakeMesh()
    value = Tensor([[[0], [1]], [[2], [3]]])

    gathered = style_module._apply_local_transform(value, AllGather(1), mesh)
    assert gathered.shape == (2, 4, 1)
    assert gathered.asnumpy().tolist() == [[[0], [1], [0], [1]], [[2], [3], [2], [3]]]

    sharded = style_module._apply_local_transform(value, ShardTensor(-2), mesh)
    assert sharded.shape == (2, 1, 1)
    assert sharded.asnumpy().tolist() == [[[1]], [[3]]]

    with pytest.raises(ValueError, match="out of range"):
        style_module._apply_local_transform(value, AllGather(3), mesh)
    with pytest.raises(ValueError, match="must be divisible"):
        style_module._apply_local_transform(Tensor([0, 1, 2]), ShardTensor(0), mesh)
    with pytest.raises(TypeError, match="Unsupported local tensor transform"):
        style_module._apply_local_transform(value, object(), mesh)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_no_parallel_and_sequence_parallel(mock_local_runtime):
    """Replicated parameter plans and optional sequence input sharding are applied."""
    linear = Linear(4, 4, "float32", "float32", bias=True)
    NoParallel()._apply(linear, _FakeMesh())
    plan = mock_local_runtime[-1]
    assert isinstance(plan["weight"][0], Replicate)
    assert isinstance(plan["bias"][0], Replicate)

    identity = _Identity()
    SequenceParallel(sequence_dim=0, input_is_parallel=False)._apply(identity, _FakeMesh())
    output = identity(Tensor([[[0]], [[1]], [[2]], [[3]]]))
    assert output.asnumpy().tolist() == [[[2]], [[3]]]

    already_sharded = _Identity()
    SequenceParallel(sequence_dim=0, input_is_parallel=True)._apply(already_sharded, _FakeMesh())
    value = Tensor([[[2]], [[3]]])
    assert already_sharded(value).asnumpy().tolist() == value.asnumpy().tolist()

    with pytest.raises(TypeError, match="sequence_dim must be an int"):
        SequenceParallel(sequence_dim="0")
    with pytest.raises(TypeError, match="input_is_parallel must be a bool"):
        SequenceParallel(input_is_parallel=1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_colwise_hooks_and_parameter_plan(mock_local_runtime):
    """Colwise gathers sequence input and optionally gathers output features."""
    linear = Linear(4, 4, "float32", "float32", bias=False)
    ColwiseParallel(gather_input=True, gather_output=True)._apply(linear, _FakeMesh())
    assert isinstance(mock_local_runtime[-1]["weight"][0], Shard)
    assert mock_local_runtime[-1]["weight"][0].dim == 0

    output = linear(Tensor([[[1, 1, 1, 1]], [[2, 2, 2, 2]]], dtype=linear.compute_dtype))
    assert output.shape == (4, 1, 8)

    with pytest.raises(NotImplementedError, match="only support"):
        ColwiseParallel()._create_weight_sharding_plan(_Identity())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize(
    "reduce_mode,expected_seq",
    [("reduce_scatter", 2), ("all_reduce", 4)],
)
def test_rowwise_communication_modes(mock_local_runtime, reduce_mode, expected_seq):
    """Rowwise supports both partial-output reductions and optional input sharding."""
    linear = Linear(4, 4, "float32", "float32", bias=True)
    style = RowwiseParallel(reduce_mode=reduce_mode, input_is_parallel=True)
    style._apply(linear, _FakeMesh())
    plan = mock_local_runtime[-1]
    assert isinstance(plan["weight"][0], Shard)
    assert plan["weight"][0].dim == 1
    assert isinstance(plan["bias"][0], Replicate)

    output = linear(Tensor([[[1, 1, 1, 1]]] * 4, dtype=linear.compute_dtype))
    assert output.shape == (expected_seq, 1, 4)

    hidden_sharding = Linear(2, 4, "float32", "float32", bias=False)
    RowwiseParallel(reduce_mode="all_reduce", input_is_parallel=False)._apply(
        hidden_sharding, _FakeMesh()
    )
    sharded_output = hidden_sharding(Tensor([[[1, 2, 3, 4]]], dtype=hidden_sharding.compute_dtype))
    assert sharded_output.shape == (1, 1, 4)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_prepare_styles_handle_args_kwargs_and_outputs(mock_local_runtime):
    """Prepare styles transform selected args/kwargs and preserve untouched values."""
    del mock_local_runtime
    mesh = _FakeMesh()
    value = Tensor([[[0]], [[1]], [[2]], [[3]]])
    other = Tensor([9])

    prepare_input = PrepareModuleInput(
        input_transforms=(ShardTensor(0), None),
        input_kwarg_transforms={"mask": ShardTensor(0)},
    )
    args, kwargs = prepare_input._prepare_input_kwarg_fn(
        (value, other), {"mask": value, "scale": 2}, mesh
    )
    assert args[0].asnumpy().tolist() == [[[2]], [[3]]]
    assert args[1] is other
    assert kwargs["mask"].shape == (2, 1, 1)
    assert kwargs["scale"] == 2

    prepare_output = PrepareModuleOutput(output_transforms=(ShardTensor(0), None))
    outputs = prepare_output._prepare_out_fn((value, other), mesh)
    assert outputs[0].shape == (2, 1, 1)
    assert outputs[1] is other

    combined = PrepareModuleInputOutput(
        input_transforms=ShardTensor(0),
        output_transforms=AllGather(0),
    )
    assert "ShardTensor(dim=0)" in repr(combined)
    assert "AllGather(dim=0)" in repr(combined)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_context_parallel_style_builders(monkeypatch):
    """Synchronous and asynchronous CP builders validate methods and layouts."""
    monkeypatch.setattr(style_module, "HPContextParallel", _FakeHPStyle)
    monkeypatch.setattr(style_module, "HPAsyncContextParallel", _FakeHPStyle)

    colossal = build_hp_cp_style("colossal", cp_size=2, input_layout="BNSD")
    assert colossal.hp_style.kwargs == {
        "seq_dim": 1, "head_dim": 2, "ulysses_degree": 1, "qkv_indices": (0, 1, 2)
    }
    tnd = build_hp_cp_style("ulysses", cp_size=2, input_layout="TND")
    assert tnd.hp_style.kwargs["seq_dim"] == 0
    assert tnd.hp_style.kwargs["head_dim"] == 1
    assert tnd.hp_style.kwargs["ulysses_degree"] == 2

    async_style = build_hp_async_cp_style("hybrid", cp_size=4, ulysses_degree_in_cp=2)
    assert async_style.hp_style.kwargs["ulysses_degree"] == 2
    module = _Identity()
    assert async_style._apply(module, _FakeMesh()) is module

    with pytest.raises(ValueError, match="requires ulysses_degree_in_cp"):
        build_hp_cp_style("hybrid", cp_size=4, input_layout="TND")
    with pytest.raises(NotImplementedError, match="unsupported context parallel method"):
        build_hp_cp_style("unknown", cp_size=2)
    with pytest.raises(NotImplementedError, match="currently supports"):
        build_hp_async_cp_style("colossal", cp_size=2)
