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

from hyper_parallel.core.placement_types import Shard, Replicate
from mindformers.pynative.distributed.style import (
    PrepareModuleInput,
    PrepareModuleOutput,
    PrepareModuleInputOutput,
)


class TestPrepareModule:
    """Test cases for PrepareModuleInput, PrepareModuleOutput, and
    PrepareModuleInputOutput classes."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_prepare_module_input_output_validation(self):
        """
        Feature: PrepareModuleInput/Output/InputOutput
        Description: Test PrepareModuleInputOutput validation, which covers
        PrepareModuleInput and PrepareModuleOutput validation.
        Expectation: mismatched layouts raise ValueError.
        """
        with pytest.raises(
            ValueError, match="desired module inputs should not be None"
        ):
            PrepareModuleInputOutput(
                input_layouts=(Shard(0),),
                desired_input_layouts=None,
                output_layouts=(Shard(0),),
                desired_output_layouts=(Replicate(),),
            )

        with pytest.raises(ValueError, match="should have same length"):
            PrepareModuleInputOutput(
                input_layouts=(Shard(0),),
                desired_input_layouts=(Replicate(), Shard(0)),
                output_layouts=(Shard(0),),
                desired_output_layouts=(Replicate(),),
            )

        with pytest.raises(ValueError, match="should have same length"):
            PrepareModuleInputOutput(
                input_kwarg_layouts={"mask": Shard(0)},
                desired_input_kwarg_layouts={
                    "mask": Replicate(),
                    "other": Replicate(),
                },
                output_layouts=(Shard(0),),
                desired_output_layouts=(Replicate(),),
            )

        with pytest.raises(ValueError, match="should have same length"):
            PrepareModuleInputOutput(
                output_layouts=(Shard(0),),
                desired_output_layouts=(Replicate(), Shard(0)),
            )

        with pytest.raises(ValueError, match="should have same length"):
            PrepareModuleInputOutput(
                output_layouts=(Shard(0), Replicate()),
                desired_output_layouts=(Shard(1),),
            )

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_prepare_module_input_repr(self):
        """
        Feature: PrepareModuleInput
        Description: Test PrepareModuleInput __repr__ method.
        Expectation: repr string contains class name and key attributes.
        """
        prepare_input = PrepareModuleInput(
            input_layouts=(Shard(0),),
            desired_input_layouts=(Replicate(),),
            use_local_output=True,
        )

        repr_str = repr(prepare_input)
        assert "PrepareModuleInput" in repr_str
        assert "input_layouts" in repr_str
        assert "desired_input_layouts" in repr_str
        assert "use_local_output=True" in repr_str

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_prepare_module_output_repr(self):
        """
        Feature: PrepareModuleOutput
        Description: Test PrepareModuleOutput __repr__ method.
        Expectation: repr string contains class name and key attributes.
        """
        prepare_output = PrepareModuleOutput(
            output_layouts=(Shard(0),),
            desired_output_layouts=(Replicate(),),
            use_local_output=True,
        )

        repr_str = repr(prepare_output)
        assert "PrepareModuleOutput" in repr_str
        assert "output_layouts" in repr_str
        assert "desired_output_layouts" in repr_str
        assert "use_local_output=True" in repr_str

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    def test_prepare_module_input_output_repr(self):
        """
        Feature: PrepareModuleInputOutput
        Description: Test PrepareModuleInputOutput __repr__ method.
        Expectation: repr string contains class name and key attributes.
        """
        prepare_input_output = PrepareModuleInputOutput(
            input_layouts=(Shard(0),),
            desired_input_layouts=(Replicate(),),
            output_layouts=(Replicate(),),
            desired_output_layouts=(Shard(0),),
            use_local_input=True,
            use_local_output=True,
        )

        repr_str = repr(prepare_input_output)
        assert "PrepareModuleInputOutput" in repr_str
        assert "input_layouts" in repr_str
        assert "desired_input_layouts" in repr_str
        assert "output_layouts" in repr_str
        assert "desired_output_layouts" in repr_str
        assert "use_local_input=True" in repr_str
        assert "use_local_output=True" in repr_str
