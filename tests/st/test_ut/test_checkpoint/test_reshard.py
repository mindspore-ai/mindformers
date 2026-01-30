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
"""Test checkpoint resharding."""

import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter

from mindformers.checkpoint.reshard import (
    smart_slice,
    balance_load
)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_smart_slice_full_slice_tensor():
    """
    Feature: smart_slice function - full slice scenario
    Description: Test smart_slice with a complete slice (no actual slicing needed)
    Expectation: Returns the original tensor when slice covers entire tensor
    """
    # Create a 2D tensor
    tensor = Tensor(np.arange(24).reshape(4, 6), dtype=ms.float32)
    slice_ranges = [(0, 4), (0, 6)]  # Full slice

    result = smart_slice(tensor, slice_ranges, load_from_multi_rank=False)

    assert result is tensor  # Should return original tensor for full slice
    assert np.array_equal(result.asnumpy(), tensor.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_smart_slice_partial_slice_tensor():
    """
    Feature: smart_slice function - partial slice scenario
    Description: Test smart_slice with partial slicing
    Expectation: Returns correctly sliced tensor
    """
    tensor = Tensor(np.arange(24).reshape(4, 6), dtype=ms.float32)
    slice_ranges = [(1, 3), (2, 5)]  # Partial slice

    result = smart_slice(tensor, slice_ranges, load_from_multi_rank=False)

    expected = tensor[1:3, 2:5]
    assert np.array_equal(result.asnumpy(), expected.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_smart_slice_full_slice_with_multi_rank():
    """
    Feature: smart_slice function - full slice with load_from_multi_rank=True
    Description: Test smart_slice forces slicing even for full slice when load_from_multi_rank=True
    Expectation: Returns numpy array even for full slice
    """
    tensor = Tensor(np.arange(24).reshape(4, 6), dtype=ms.float32)
    slice_ranges = [(0, 4), (0, 6)]  # Full slice

    result = smart_slice(tensor, slice_ranges, load_from_multi_rank=True)

    # Should return numpy array, not original tensor
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, tensor.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_smart_slice_partial_slice_with_multi_rank():
    """
    Feature: smart_slice function - partial slice with load_from_multi_rank=True
    Description: Test smart_slice with partial slicing and multi-rank loading
    Expectation: Returns numpy array with correct slice
    """
    tensor = Tensor(np.arange(24).reshape(4, 6), dtype=ms.float32)
    slice_ranges = [(1, 3), (2, 5)]

    result = smart_slice(tensor, slice_ranges, load_from_multi_rank=True)

    assert isinstance(result, np.ndarray)
    expected = tensor.asnumpy()[1:3, 2:5]
    assert np.array_equal(result, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_smart_slice_parameter_type():
    """
    Feature: smart_slice function - Parameter type
    Description: Test smart_slice with Parameter type input
    Expectation: Handles Parameter type correctly
    """
    param = Parameter(Tensor(np.arange(12).reshape(3, 4), dtype=ms.float32))
    slice_ranges = [(0, 2), (1, 3)]

    result = smart_slice(param, slice_ranges, load_from_multi_rank=False)

    expected = param[0:2, 1:3]
    assert np.array_equal(result.asnumpy(), expected.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_smart_slice_numpy_array():
    """
    Feature: smart_slice function - numpy array input
    Description: Test smart_slice with numpy array input
    Expectation: Handles numpy array correctly
    """
    arr = np.arange(24).reshape(4, 6)
    slice_ranges = [(1, 3), (2, 5)]

    result = smart_slice(arr, slice_ranges, load_from_multi_rank=False)

    expected = arr[1:3, 2:5]
    assert np.array_equal(result, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_smart_slice_dimension_mismatch():
    """
    Feature: smart_slice function - dimension mismatch error
    Description: Test smart_slice raises ValueError when slice dimension doesn't match tensor dimension
    Expectation: Raises ValueError with appropriate message
    """
    tensor = Tensor(np.arange(24).reshape(4, 6), dtype=ms.float32)
    slice_ranges = [(0, 4), (0, 6), (0, 2)]  # Wrong dimension count

    with pytest.raises(ValueError) as exc_info:
        smart_slice(tensor, slice_ranges, load_from_multi_rank=False)

    assert "dimension count" in str(exc_info.value).lower()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_smart_slice_3d_tensor():
    """
    Feature: smart_slice function - 3D tensor slicing
    Description: Test smart_slice with 3D tensor
    Expectation: Correctly slices 3D tensor
    """
    tensor = Tensor(np.arange(60).reshape(3, 4, 5), dtype=ms.float32)
    slice_ranges = [(1, 3), (0, 2), (2, 4)]

    result = smart_slice(tensor, slice_ranges, load_from_multi_rank=False)

    expected = tensor[1:3, 0:2, 2:4]
    assert np.array_equal(result.asnumpy(), expected.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_smart_slice_single_dimension():
    """
    Feature: smart_slice function - 1D tensor slicing
    Description: Test smart_slice with 1D tensor
    Expectation: Correctly slices 1D tensor
    """
    tensor = Tensor(np.arange(10), dtype=ms.float32)
    slice_ranges = [(2, 7)]

    result = smart_slice(tensor, slice_ranges, load_from_multi_rank=False)

    expected = tensor[2:7]
    assert np.array_equal(result.asnumpy(), expected.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balance_load_basic():
    """
    Feature: balance_load function - basic load balancing
    Description: Test balance_load distributes parameters evenly
    Expectation: Parameters are distributed across groups with balanced sizes
    """
    params = [
        {"size": 100, "name": "param1"},
        {"size": 200, "name": "param2"},
        {"size": 150, "name": "param3"},
        {"size": 50, "name": "param4"},
    ]
    num_groups = 2

    result = balance_load(params, num_groups)

    assert len(result) == num_groups
    # Check all parameters are distributed
    total_params = sum(len(group) for group in result)
    assert total_params == len(params)

    # Check load balancing (largest group should not be too much larger than smallest)
    group_sizes = [sum(p["size"] for p in group) for group in result]
    max_size = max(group_sizes)
    min_size = min(group_sizes)
    # Difference should be reasonable (within 50% of average)
    avg_size = sum(group_sizes) / len(group_sizes)
    assert (max_size - min_size) <= avg_size * 0.5


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balance_load_single_group():
    """
    Feature: balance_load function - single group
    Description: Test balance_load with single group
    Expectation: All parameters go to single group
    """
    params = [
        {"size": 100, "name": "param1"},
        {"size": 200, "name": "param2"},
    ]
    num_groups = 1

    result = balance_load(params, num_groups)

    assert len(result) == 1
    assert len(result[0]) == len(params)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balance_load_more_groups_than_params():
    """
    Feature: balance_load function - more groups than parameters
    Description: Test balance_load when num_groups > len(params)
    Expectation: Some groups will be empty
    """
    params = [
        {"size": 100, "name": "param1"},
        {"size": 200, "name": "param2"},
    ]
    num_groups = 5

    result = balance_load(params, num_groups)

    assert len(result) == num_groups
    non_empty_groups = [g for g in result if len(g) > 0]
    assert len(non_empty_groups) == len(params)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balance_load_large_imbalance():
    """
    Feature: balance_load function - large size imbalance
    Description: Test balance_load with very different parameter sizes
    Expectation: Large parameters are distributed first to balance load
    """
    params = [
        {"size": 1000, "name": "large1"},
        {"size": 1000, "name": "large2"},
        {"size": 10, "name": "small1"},
        {"size": 10, "name": "small2"},
        {"size": 10, "name": "small3"},
    ]
    num_groups = 2

    result = balance_load(params, num_groups)

    # Large parameters should be split first
    group_sizes = [sum(p["size"] for p in group) for group in result]
    # Both groups should have similar total sizes
    assert abs(group_sizes[0] - group_sizes[1]) <= 20  # Allow small difference


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balance_load_empty_params():
    """
    Feature: balance_load function - empty parameter list
    Description: Test balance_load with empty parameter list
    Expectation: Returns empty groups
    """
    params = []
    num_groups = 3

    result = balance_load(params, num_groups)

    assert len(result) == num_groups
    assert all(len(group) == 0 for group in result)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balance_load_sorted_order():
    """
    Feature: balance_load function - sorting order
    Description: Test that balance_load sorts parameters by size (descending)
    Expectation: Largest parameters are assigned first
    """
    params = [
        {"size": 50, "name": "small"},
        {"size": 300, "name": "large"},
        {"size": 100, "name": "medium"},
    ]
    num_groups = 2

    result = balance_load(params, num_groups)

    # First group should get the largest parameter
    first_group_sizes = [p["size"] for p in result[0]]
    assert 300 in first_group_sizes  # Largest should be in first group


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balance_load_equal_sizes():
    """
    Feature: balance_load function - equal sized parameters
    Description: Test balance_load with parameters of equal size
    Expectation: Parameters distributed evenly
    """
    params = [
        {"size": 100, "name": f"param{i}"} for i in range(6)
    ]
    num_groups = 3

    result = balance_load(params, num_groups)

    # Each group should have 2 parameters
    assert all(len(group) == 2 for group in result)
    # Each group should have total size 200
    assert all(sum(p["size"] for p in group) == 200 for group in result)
