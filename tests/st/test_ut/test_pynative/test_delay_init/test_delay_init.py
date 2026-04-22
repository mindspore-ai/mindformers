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
"""Test delay initialization feature in TrainModelMixin"""

from unittest.mock import patch, MagicMock
import pytest
import numpy as np

import mindspore as ms
from mindspore import nn, Tensor, Parameter
from mindspore.common.api import _no_grad

from mindformers.parallel_core.utils.model_mixin import TrainModelMixin


class MockCellWithReset(nn.Cell):
    """Mock cell with reset_parameter method for testing"""

    def __init__(self, name="mock_cell"):
        super().__init__()
        self.name = name
        self.reset_called = False
        self.weight = Parameter(Tensor(np.random.rand(10, 10).astype(np.float32)), name="weight")

    def reset_parameter(self):
        """Reset parameter implementation"""
        with _no_grad():
            self.weight.normal_(mean=0.0, std=0.01)
        self.reset_called = True


class MockCellWithoutReset(nn.Cell):
    """Mock cell without reset_parameter method"""

    def __init__(self, name="mock_cell_no_reset"):
        super().__init__()
        self.name = name
        self.weight = Parameter(Tensor(np.random.rand(10, 10).astype(np.float32)), name="weight")


class MockModel(nn.Cell):
    """Mock model for testing delay initialization"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.layer1 = MockCellWithReset("layer1")
        self.layer2 = MockCellWithReset("layer2")
        self.layer3 = MockCellWithoutReset("layer3")

    def construct(self, x):
        return x


class TestInitStatesModeHandling:
    """Test init_states behavior in different modes"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_graph_mode_no_op(self):
        """
        Feature: Delay Initialization
        Description: Test init_states in Graph mode should be no-op.
        Expectation: Returns immediately without processing.
        """
        class TestModel(TrainModelMixin):
            def __init__(self):
                super().__init__()
                self.model = MockModel()

        mixin = TestModel()
        with patch('mindspore.get_context', return_value=ms.GRAPH_MODE):
            mixin.init_states()
            # In Graph mode, reset_parameter should not be called
            assert not mixin.model.layer1.reset_called

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_no_model_attribute(self):
        """
        Feature: Delay Initialization
        Description: Test init_states when model attribute is missing.
        Expectation: Logs warning and returns without error.
        """
        mixin = TrainModelMixin()
        with patch('mindspore.get_context', return_value=ms.PYNATIVE_MODE):
            # Should not raise exception
            mixin.init_states()


class TestInitStatesParameterReset:
    """Test init_states parameter reset functionality"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_normal_parameter_reset(self):
        """
        Feature: Delay Initialization
        Description: Test normal parameter reset in PyNative mode.
        Expectation: All cells with reset_parameter are called.
        """
        class TestModel(TrainModelMixin):
            def __init__(self):
                super().__init__()
                self.model = MockModel()

        mixin = TestModel()
        with patch('mindspore.get_context', return_value=ms.PYNATIVE_MODE):
            mixin.init_states()
            # layer1 and layer2 have reset_parameter, layer3 does not
            assert mixin.model.layer1.reset_called
            assert mixin.model.layer2.reset_called

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reset_parameter_exception_handling(self):
        """
        Feature: Delay Initialization
        Description: Test exception handling when reset_parameter fails.
        Expectation: Exception is raised and propagated.
        """
        class MockCellWithFailingReset(nn.Cell):
            def __init__(self):
                super().__init__()
                self.weight = Parameter(Tensor(np.random.rand(10, 10).astype(np.float32)))

            def reset_parameter(self):
                raise ValueError("reset_parameter failed")

        class TestModelWithFailingLayer(TrainModelMixin):
            def __init__(self):
                super().__init__()
                self.model = MockModel()
                self.model.failing_layer = MockCellWithFailingReset()

        mixin = TestModelWithFailingLayer()
        with patch('mindspore.get_context', return_value=ms.PYNATIVE_MODE):
            with pytest.raises(RuntimeError, match="reset_parameter failed"):
                mixin.init_states()


class TestInitStatesReturnValues:
    """Test init_states return values and statistics"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reset_statistics(self):
        """
        Feature: Delay Initialization
        Description: Test that _reset_all_parameters returns correct statistics.
        Expectation: Returns tuple of (reset_count, skip_count, error_count).
        """
        class TestModel(TrainModelMixin):
            def __init__(self):
                super().__init__()
                self.model = MockModel()

        mixin = TestModel()
        with patch('mindspore.get_context', return_value=ms.PYNATIVE_MODE):
            reset_count, skip_count, error_count = mixin._reset_all_parameters(mixin.model)
            # cells_and_names() traverses: MockModel(root), layer1, layer2, layer3
            # layer1 and layer2 have reset_parameter, MockModel and layer3 do not
            assert reset_count == 2
            assert skip_count == 2  # MockModel itself + layer3
            assert error_count == 0


class TestIntegration:
    """Integration tests for delay initialization workflow"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_full_workflow_pynative(self):
        """
        Feature: Delay Initialization
        Description: Test complete workflow in PyNative mode.
        Expectation: Parameters are reset correctly after to_empty.
        """
        class TestModel(TrainModelMixin):
            def __init__(self):
                super().__init__()
                self.model = MockModel()

        mixin = TestModel()
        with patch('mindspore.get_context', return_value=ms.PYNATIVE_MODE):
            # Simulate Trainer calling to_empty first (already done by Trainer)
            # Then call init_states to reset parameters
            mixin.init_states()

            # Verify all layers with reset_parameter were called
            assert mixin.model.layer1.reset_called
            assert mixin.model.layer2.reset_called

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_multiple_init_states_calls(self):
        """
        Feature: Delay Initialization
        Description: Test calling init_states multiple times.
        Expectation: Each call resets parameters independently.
        """
        class TestModel(TrainModelMixin):
            def __init__(self):
                super().__init__()
                self.model = MockModel()

        mixin = TestModel()
        with patch('mindspore.get_context', return_value=ms.PYNATIVE_MODE):
            # First call
            mixin.init_states()
            first_reset = mixin.model.layer1.reset_called
            
            # Reset the flag for second call test
            mixin.model.layer1.reset_called = False
            
            # Second call
            mixin.init_states()
            second_reset = mixin.model.layer1.reset_called
            
            # Both calls should execute reset_parameter
            assert first_reset is True
            assert second_reset is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
