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

import operator
from functools import reduce
from unittest.mock import MagicMock

import os
import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore.parallel import Layout

from mindformers.checkpoint.reshard import (
    smart_slice,
    balance_load,
    infer_slice_area_by_rank,
    ReshardHandler,
    ReshardLoader
)
from mindformers.checkpoint.sharded_tensor import build_sharded_tensor
from mindformers.checkpoint.utils import get_sharded_tensor_shard_id


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


# Test ReshardHandler
def get_slice_data(full_data, offset):
    area = ()
    for begin, end in offset:
        area += (slice(begin, end),)
    return full_data[area]


def reshard_tensor_func(param_name, full_shape, from_layout, to_layout, to_rank_id):
    """reshard tensor and verify"""
    reshard = ReshardHandler(param_name, full_shape, from_layout, to_layout, to_rank_id)
    reshard.infer_all_tensor_offset()
    all_offset = reshard.global_union_area_map

    # Generate fake from data
    ele_num = reduce(operator.mul, full_shape)
    full_data = np.array(range(ele_num), np.int32).reshape(full_shape)
    from_tensor_map = {}
    for rank, offset in all_offset.items():
        from_tensor_map[rank] = Tensor(get_slice_data(full_data, offset))

    # Transfer and verify
    actual_result = reshard.get_real_tensor(from_tensor_map).asnumpy()
    to_layout_dict = to_layout.to_dict()
    to_offset = infer_slice_area_by_rank(
        to_layout_dict['device_matrix'],
        to_layout_dict['tensor_map'],
        to_layout_dict['rank_list'].index(to_rank_id),
        full_shape
    )
    expect_result = get_slice_data(full_data, to_offset)
    assert np.all(actual_result == expect_result)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_reshard_between_fully_shard():
    """
    Feature: Tensor resharding between fully sharded modes.
    Description: Test bidirectional tensor resharding between two different fully sharded layouts.
    Expectation: The reshard_tensor_func executes successfully without throwing exceptions,
                 completing bidirectional tensor resharding between the two fully sharded layouts.
    """
    param_name = "weight"
    full_shape = (64, 64)

    layout_0 = Layout((2, 2, 2, 2), ('dp', 'cp', 'rep', 'tp'), rank_list=list(range(16, 32)))
    from_layout = layout_0(('rep', 'cp'), 'tp')

    layout_1 = Layout((2, 4), ('dp', 'tp'), rank_list=list(range(8, 16)))
    to_layout = layout_1('dp', 'tp')

    from_rank_id = 19
    to_rank_id = 14
    reshard_tensor_func(param_name, full_shape, from_layout, to_layout, to_rank_id)
    reshard_tensor_func(param_name, full_shape, to_layout, from_layout, from_rank_id)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_reshard_between_fully_shard_and_not_shard():
    """
    Feature: Tensor resharding between fully sharded and non-sharded modes.
    Description: Test bidirectional tensor resharding between a fully sharded layout and a non-sharded layout.
    Expectation: The reshard_tensor_func executes successfully without throwing exceptions,
                 completing bidirectional tensor resharding between the fully sharded and non-sharded layouts.
    """
    param_name = "weight"
    full_shape = (64, 64)

    layout_0 = Layout((2, 4, 4), ('dp', 'cp', 'tp'), rank_list=list(range(32, 64)))
    from_layout = layout_0(('cp', 'dp'), 'tp')

    layout_1 = Layout((8,), ('dp',), rank_list=list(range(0, 8)))
    to_layout = layout_1('None', 'None')

    from_rank_id = 35
    to_rank_id = 7
    reshard_tensor_func(param_name, full_shape, from_layout, to_layout, to_rank_id)
    reshard_tensor_func(param_name, full_shape, to_layout, from_layout, from_rank_id)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_reshard_between_not_fully_shard():
    """
    Feature: Tensor resharding between non-fully sharded modes.
    Description: Test bidirectional tensor resharding between two different non-fully sharded layouts.
    Expectation: The reshard_tensor_func executes successfully without throwing exceptions,
                 completing bidirectional tensor resharding between the two non-fully sharded layouts.
    """
    param_name = "weight"
    full_shape = (64, 64)

    layout_0 = Layout((2, 4, 4), ('dp', 'cp', 'tp'), rank_list=list(range(32, 64)))
    from_layout = layout_0('cp', 'tp')

    layout_1 = Layout((2, 4, 4), ('dp', 'cp', 'tp'), rank_list=list(range(0, 32)))
    to_layout = layout_1('tp', 'dp')

    from_rank_id = 35
    to_rank_id = 7
    reshard_tensor_func(param_name, full_shape, from_layout, to_layout, to_rank_id)
    reshard_tensor_func(param_name, full_shape, to_layout, from_layout, from_rank_id)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_from_tensor_map_missing_rank():
    """
    Feature: Exception handling for incomplete from_tensor_map.
    Description: Test the scenario where from_tensor_map is missing a required rank.
    Expectation: A ValueError exception is raised when calling get_real_tensor()
                 with an incomplete from_tensor_map that lacks a necessary rank entry.
    """
    param_name = "weight"
    full_shape = (64, 64)

    layout_0 = Layout((2, 4, 4), ('dp', 'cp', 'tp'), rank_list=list(range(32, 64)))
    from_layout = layout_0(('cp', 'dp'), 'tp')

    layout_1 = Layout((8,), ('dp',), rank_list=list(range(0, 8)))
    to_layout = layout_1('None', 'None')

    to_rank_id = 7
    reshard = ReshardHandler(param_name, full_shape, from_layout, to_layout, to_rank_id)
    reshard.infer_all_tensor_offset()
    all_offset = reshard.global_union_area_map

    # Generate fake from data
    ele_num = reduce(operator.mul, full_shape)
    full_data = np.array(range(ele_num), np.int32).reshape(full_shape)
    from_tensor_map = {}
    pop_rank = 0
    for rank, offset in all_offset.items():
        pop_rank = rank
        from_tensor_map[rank] = Tensor(get_slice_data(full_data, offset))
    from_tensor_map.pop(pop_rank)
    with pytest.raises(ValueError):
        reshard.get_real_tensor(from_tensor_map)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_from_tensor_map_has_unexpected_data():
    """
    Feature: Exception handling for invalid data in from_tensor_map.
    Description: Test the scenario where a rank in from_tensor_map contains data with an unexpected shape.
    Expectation: A ValueError exception is raised when calling get_real_tensor()
                 with from_tensor_map containing a rank with data that has an unexpected shape.
    """
    param_name = "weight"
    full_shape = (64, 64)

    layout_0 = Layout((2, 4, 4), ('dp', 'cp', 'tp'), rank_list=list(range(32, 64)))
    from_layout = layout_0(('cp', 'dp'), 'tp')

    layout_1 = Layout((8,), ('dp',), rank_list=list(range(0, 8)))
    to_layout = layout_1('None', 'None')

    to_rank_id = 7
    reshard = ReshardHandler(param_name, full_shape, from_layout, to_layout, to_rank_id)
    reshard.infer_all_tensor_offset()
    all_offset = reshard.global_union_area_map

    # Generate fake from data
    ele_num = reduce(operator.mul, full_shape)
    full_data = np.array(range(ele_num), np.int32).reshape(full_shape)
    from_tensor_map = {}
    modify_rank = 0
    for rank, offset in all_offset.items():
        modify_rank = rank
        from_tensor_map[rank] = Tensor(get_slice_data(full_data, offset))
    modify_shape = from_tensor_map[modify_rank].shape
    modify_shape = modify_shape[:-1] + (modify_shape[-1] + 2,)
    from_tensor_map[modify_rank] = Tensor(np.zeros(modify_shape, full_data.dtype))
    with pytest.raises(ValueError):
        reshard.get_real_tensor(from_tensor_map)


# Test ReshardLoader
class TestReshardLoader:
    """Test class for ReshardLoader functionality."""

    @pytest.fixture
    def mock_checkpoint_dir(self, tmp_path):
        """Create a mock checkpoint directory with safetensors files."""
        checkpoint_dir = os.path.join(tmp_path, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create mock safetensors files
        for i in range(2):
            file_path = os.path.join(checkpoint_dir, f"model-{i:08d}.safetensors")
            with open(file_path, 'wb') as f:
                f.write(b"mock_safetensors_data")

        return checkpoint_dir

    @pytest.fixture
    def simple_sharded_tensor(self):
        """Create a simple ShardedTensor for testing."""
        layout = Layout((1,), ('dp',), rank_list=[0])
        simple_layout = layout('None', 'None')

        return build_sharded_tensor(
            param_name="test.weight",
            param_dtype=ms.float32,
            local_shape=(10, 20),
            global_shape=(10, 20),
            global_offset=(0, 0),
            axis_fragmentations=(1, 1),
            layout=simple_layout
        )

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reshard_loader_init_without_template(self, mock_checkpoint_dir, simple_sharded_tensor):
        """
        Feature: ReshardLoader initialization without template
        Description: Test ReshardLoader initialization for self-trained weights (no template)
        Expectation: ReshardLoader is initialized successfully with correct bidirectional mapping
        """
        dst_metas = {
            "test.weight": simple_sharded_tensor
        }
        src_metas = {
            "test.weight": [simple_sharded_tensor]
        }
        # Use get_sharded_tensor_shard_id to generate the correct key format (string)
        mapping_key = get_sharded_tensor_shard_id("test.weight", (0, 0))
        param_file_mappings = {
            mapping_key: [{
                "file_name": "model-00000000.safetensors",
                "storage_rank": 0,
                "rank_group": [0]
            }]
        }

        loader = ReshardLoader(
            checkpoint_dir=mock_checkpoint_dir,
            dst_sharded_tensor_metas=dst_metas,
            src_sharded_tensor_metas=src_metas,
            param_file_mappings=param_file_mappings,
            reshard_worker_num=1,
            template=None
        )

        assert loader.checkpoint_dir == mock_checkpoint_dir
        assert loader.dst_metas == dst_metas
        assert loader.src_metas == src_metas
        assert loader.template is None
        assert loader.get_dst_name("test.weight") == "test.weight"
        assert loader.get_src_names("test.weight") == ["test.weight"]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reshard_loader_bidirectional_mapping_without_template(self, mock_checkpoint_dir, simple_sharded_tensor):
        """
        Feature: ReshardLoader bidirectional mapping without template
        Description: Test bidirectional mapping construction for self-trained weights
        Expectation: src_to_dst and dst_to_src mappings are correctly built
        """
        dst_metas = {
            "param1.weight": simple_sharded_tensor,
            "param2.weight": simple_sharded_tensor
        }
        src_metas = {
            "param1.weight": [simple_sharded_tensor],
            "param2.weight": [simple_sharded_tensor]
        }
        param_file_mappings = {}

        loader = ReshardLoader(
            checkpoint_dir=mock_checkpoint_dir,
            dst_sharded_tensor_metas=dst_metas,
            src_sharded_tensor_metas=src_metas,
            param_file_mappings=param_file_mappings,
            reshard_worker_num=1,
            template=None
        )

        # For self-trained weights, src_name == dst_name
        assert loader.get_dst_name("param1.weight") == "param1.weight"
        assert loader.get_dst_name("param2.weight") == "param2.weight"
        assert loader.get_src_names("param1.weight") == ["param1.weight"]
        assert loader.get_src_names("param2.weight") == ["param2.weight"]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reshard_loader_bidirectional_mapping_with_template(self, mock_checkpoint_dir, simple_sharded_tensor):
        """
        Feature: ReshardLoader bidirectional mapping with template
        Description: Test bidirectional mapping construction for HuggingFace weights with template
        Expectation: src_to_dst and dst_to_src mappings are correctly built using template
        """
        # Mock template
        mock_template = MagicMock()
        mock_template.get_mf_name = MagicMock(
            side_effect=lambda x: ("qkv.weight",)
            if "q_proj" in x
               or "k_proj" in x
               or "v_proj" in x
            else ("other.weight",)
        )

        dst_metas = {
            "qkv.weight": simple_sharded_tensor
        }
        src_metas = {
            "q_proj.weight": [simple_sharded_tensor],
            "k_proj.weight": [simple_sharded_tensor],
            "v_proj.weight": [simple_sharded_tensor]
        }
        param_file_mappings = {}

        loader = ReshardLoader(
            checkpoint_dir=mock_checkpoint_dir,
            dst_sharded_tensor_metas=dst_metas,
            src_sharded_tensor_metas=src_metas,
            param_file_mappings=param_file_mappings,
            reshard_worker_num=1,
            template=mock_template
        )

        # For HF weights, multiple src_names map to one dst_name
        src_names = loader.get_src_names("qkv.weight")
        assert "q_proj.weight" in src_names
        assert "k_proj.weight" in src_names
        assert "v_proj.weight" in src_names

        assert loader.get_dst_name("q_proj.weight") == "qkv.weight"
        assert loader.get_dst_name("k_proj.weight") == "qkv.weight"
        assert loader.get_dst_name("v_proj.weight") == "qkv.weight"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reshard_loader_get_dst_name_nonexistent(self, mock_checkpoint_dir, simple_sharded_tensor):
        """
        Feature: ReshardLoader get_dst_name for nonexistent parameter
        Description: Test get_dst_name returns None for parameters not in mapping
        Expectation: Returns None for nonexistent parameters
        """
        dst_metas = {
            "test.weight": simple_sharded_tensor
        }
        src_metas = {
            "test.weight": [simple_sharded_tensor]
        }
        param_file_mappings = {}

        loader = ReshardLoader(
            checkpoint_dir=mock_checkpoint_dir,
            dst_sharded_tensor_metas=dst_metas,
            src_sharded_tensor_metas=src_metas,
            param_file_mappings=param_file_mappings,
            reshard_worker_num=1,
            template=None
        )

        assert loader.get_dst_name("nonexistent.weight") is None
        assert loader.get_src_names("nonexistent.weight") is None

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reshard_loader_missing_src_in_dst(self, mock_checkpoint_dir, simple_sharded_tensor):
        """
        Feature: ReshardLoader handles missing source parameters
        Description: Test ReshardLoader initialization when src_metas has parameters not in dst_metas
        Expectation: Parameters not in dst_metas are excluded from mapping
        """
        dst_metas = {
            "param1.weight": simple_sharded_tensor
        }
        src_metas = {
            "param1.weight": [simple_sharded_tensor],
            "param2.weight": [simple_sharded_tensor]  # `param2.weight` not in `dst_metas`
        }
        param_file_mappings = {}

        loader = ReshardLoader(
            checkpoint_dir=mock_checkpoint_dir,
            dst_sharded_tensor_metas=dst_metas,
            src_sharded_tensor_metas=src_metas,
            param_file_mappings=param_file_mappings,
            reshard_worker_num=1,
            template=None
        )

        # param1 should be mapped
        assert loader.get_dst_name("param1.weight") == "param1.weight"
        # param2 should not be mapped (not in dst_metas)
        assert loader.get_dst_name("param2.weight") is None
