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
"""Test the classes of style.py."""
import pytest
from mindspore import Tensor

from mindformers.pynative.distributed.style import (
    AllGather,
    ColwiseParallel,
    ShardTensor,
    PrepareModuleInput,
    PrepareModuleOutput,
    PrepareModuleInputOutput,
    NoParallel,
    RowwiseParallel,
    SequenceParallel,
)


class _FakeMesh:
    """Minimal fake device mesh for style unit tests."""

    @staticmethod
    def size():
        return 2

    @staticmethod
    def get_group():
        return "tp-group"

    @staticmethod
    def get_local_rank():
        return 1


class TestPrepareModule:
    """Test cases for PrepareModuleInput, PrepareModuleOutput, and
    PrepareModuleInputOutput classes."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_prepare_module_input_output_validation(self):
        """
        Feature: PrepareModuleInput/Output/InputOutput
        Description: Test positional input/output transform arity validation.
        Expectation: mismatched transform counts raise ValueError.
        """
        style = PrepareModuleInputOutput(
            input_transforms=(None,),
            output_transforms=(None,),
        )
        with pytest.raises(ValueError, match="inputs and input_transforms should have same length"):
            style.prepare_module_input._prepare_input_fn((object(), object()), None)
        with pytest.raises(ValueError, match="outputs and output_transforms should have same length"):
            style.prepare_module_output._prepare_out_fn((object(), object()), None)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_prepare_module_input_repr(self):
        """
        Feature: PrepareModuleInput
        Description: Test PrepareModuleInput __repr__ method.
        Expectation: repr string contains class name and key attributes.
        """
        prepare_input = PrepareModuleInput(
            input_transforms=(AllGather(0), None),
            input_kwarg_transforms={"mask": ShardTensor(0)},
        )

        repr_str = repr(prepare_input)
        assert "PrepareModuleInput" in repr_str
        assert "input_transforms=(AllGather(dim=0), None)" in repr_str
        assert "input_kwarg_transforms={'mask': ShardTensor(dim=0)}" in repr_str

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_prepare_module_output_repr(self):
        """
        Feature: PrepareModuleOutput
        Description: Test PrepareModuleOutput __repr__ method.
        Expectation: repr string contains class name and key attributes.
        """
        prepare_output = PrepareModuleOutput(
            output_transforms=(ShardTensor(0),),
        )

        repr_str = repr(prepare_output)
        assert "PrepareModuleOutput" in repr_str
        assert "output_transforms=(ShardTensor(dim=0),)" in repr_str

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_prepare_module_output_shards_local_tensor(self):
        """PrepareModuleOutput applies a concrete local output transform."""
        prepare_output = PrepareModuleOutput(output_transforms=ShardTensor(0))
        output = prepare_output._prepare_out_fn(Tensor([0, 1, 2, 3]), _FakeMesh())
        assert output.asnumpy().tolist() == [2, 3]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_sequence_parallel_keeps_sequence_dim(self):
        """SequenceParallel preserves the public sequence-dimension metadata."""
        style = SequenceParallel(sequence_dim=0)
        assert style.sequence_dim == 0
        assert style.input_is_parallel is True
        assert repr(style) == "SequenceParallel(sequence_dim=0, input_is_parallel=True)"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_sequence_parallel_can_shard_input(self):
        """SequenceParallel exposes an explicit full-input to sequence-shard mode."""
        style = SequenceParallel(sequence_dim=0, input_is_parallel=False)
        assert style.input_is_parallel is False

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_no_parallel_and_rowwise_public_contracts(self):
        """Base styles keep stable names and expose both rowwise reductions."""
        assert NoParallel.__name__ == "NoParallel"
        colwise = ColwiseParallel(gather_input=False, gather_output=True)
        assert colwise.gather_input is False
        assert colwise.gather_output is True
        rowwise = RowwiseParallel(reduce_mode="all_reduce", input_is_parallel=False)
        assert rowwise.reduce_mode == "all_reduce"
        assert rowwise.input_is_parallel is False
        assert RowwiseParallel(reduce_mode="reduce_scatter").reduce_mode == "reduce_scatter"
        with pytest.raises(ValueError, match="reduce_mode must be"):
            RowwiseParallel(reduce_mode="invalid")

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_prepare_module_input_output_repr(self):
        """
        Feature: PrepareModuleInputOutput
        Description: Test PrepareModuleInputOutput __repr__ method.
        Expectation: repr string contains class name and key attributes.
        """
        prepare_input_output = PrepareModuleInputOutput(
            input_transforms=(AllGather(0),),
            output_transforms=(ShardTensor(0),),
        )

        repr_str = repr(prepare_input_output)
        assert "PrepareModuleInputOutput" in repr_str
        assert "input_transforms=(AllGather(dim=0),)" in repr_str
        assert "output_transforms=(ShardTensor(dim=0),)" in repr_str
