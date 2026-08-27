# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test sharded tensor."""

import pytest
import mindspore as ms
from mindspore import nn
from mindspore.parallel import Layout
from mindspore.common.initializer import initializer, Normal

from mindformers.checkpoint.sharded_tensor import (
    ShardedTensor,
    build_sharded_tensor,
    is_main_replica,
    get_sharded_tensor_from_cell,
    get_strategy_info_from_sharded_tensor,
    _rank_id_with_slice_id,
    _alias_name_with_rank_id,
    _flatten_tensor_map,
    _tensor_map_with_rank_id
)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sharded_tensor_creation():
    """
    Feature: ShardedTensor creation
    Description: Create a ShardedTensor instance with all required parameters
    Expectation: ShardedTensor is created successfully with correct attributes
    """
    st = ShardedTensor(
        key="test.weight",
        org_key="original.test.weight",
        dtype=ms.float32,
        local_shape=(10,),
        global_shape=(100,),
        global_offset=(0,),
        axis_fragmentations=(10,)
    )

    assert st.key == "test.weight"
    assert st.org_key == "original.test.weight"
    assert st.dtype == ms.float32
    assert st.local_shape == (10,)
    assert st.global_shape == (100,)
    assert st.global_offset == (0,)
    assert st.axis_fragmentations == (10,)
    assert st.replica_id == 0
    assert st.allow_shape_mismatch is False
    assert st.allow_to_save is True


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_build_sharded_tensor():
    """
    Feature: build_sharded_tensor function
    Description: Call build_sharded_tensor helper function to create ShardedTensor
    Expectation: ShardedTensor is created successfully with correct attributes
    """
    st = build_sharded_tensor(
        param_name="layer.weight",
        param_dtype=ms.float16,
        local_shape=[5],
        global_shape=[50],
        axis_fragmentations=[10],
        global_offset=[0]
    )

    assert isinstance(st, ShardedTensor)
    assert st.key == "layer.weight"
    assert st.dtype == ms.float16
    assert st.local_shape == (5,)
    assert st.global_shape == (50,)
    assert st.axis_fragmentations == (10,)
    assert st.global_offset == (0,)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_main_replica_zero():
    """
    Feature: is_main_replica function
    Description: Check if integer replica_id 0 is considered main replica
    Expectation: Returns True for replica_id 0
    """
    result = is_main_replica(0)
    assert result is True


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_main_replica_tuple_all_zeros():
    """
    Feature: is_main_replica function
    Description: Check if tuple of all zeros is considered main replica
    Expectation: Returns True for tuple with all zero elements
    """
    result = is_main_replica((0, 0))
    assert result is True


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_main_replica_nonzero_integer():
    """
    Feature: is_main_replica function
    Description: Check if nonzero integer is considered main replica
    Expectation: Returns False for nonzero integer replica_id
    """
    result = is_main_replica(1)
    assert result is False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_main_replica_mixed_tuple():
    """
    Feature: is_main_replica function
    Description: Check if tuple with mixed values is considered main replica
    Expectation: Returns False for tuple containing non-zero elements
    """
    result = is_main_replica((0, 1))
    assert result is False


class SimpleNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.dense = nn.Dense(10, 5)

    def construct(self, x):
        return self.dense(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_sharded_tensor_from_cell():
    """
    Feature: get_sharded_tensor_from_cell function
    Description: Extract sharded tensors from a neural network cell
    Expectation: Returns dict of ShardedTensor objects for cell parameters
    """
    net = SimpleNet()

    # Initialize parameters
    net.dense.weight.set_data(initializer(Normal(), net.dense.weight.shape, net.dense.weight.dtype))
    net.dense.bias.set_data(initializer('zeros', net.dense.bias.shape, net.dense.bias.dtype))

    sharded_tensors = get_sharded_tensor_from_cell(net)

    assert len(sharded_tensors) >= 2  # Weight and bias

    weight_tensor = next(t for t in sharded_tensors.values() if 'weight' in t.key)
    bias_tensor = next(t for t in sharded_tensors.values() if 'bias' in t.key)

    assert weight_tensor.local_shape == net.dense.weight.shape
    assert bias_tensor.local_shape == net.dense.bias.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sharded_tensor_with_custom_attributes():
    """
    Feature: ShardedTensor creation with custom attributes
    Description: Create a ShardedTensor instance with custom replica_id, allow_shape_mismatch and allow_to_save
    Expectation: ShardedTensor is created successfully with custom attributes
    """
    st = ShardedTensor(
        key="test.weight",
        org_key="original.test.weight",
        dtype=ms.float32,
        local_shape=(10,),
        global_shape=(100,),
        global_offset=(0,),
        axis_fragmentations=(10,),
        replica_id=(1, 2),
        allow_shape_mismatch=True,
        allow_to_save=False
    )

    assert st.replica_id == (1, 2)
    assert st.allow_shape_mismatch is True
    assert st.allow_to_save is False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_build_sharded_tensor_with_layout():
    """
    Feature: build_sharded_tensor function with layout
    Description: Call build_sharded_tensor helper function with layout parameter
    Expectation: ShardedTensor is created successfully with layout
    """
    layout = Layout(device_matrix=(2, 2), alias_name=("dp", "mp"))

    st = build_sharded_tensor(
        param_name="layer.weight",
        param_dtype=ms.float16,
        local_shape=[5],
        global_shape=[50],
        axis_fragmentations=[10],
        global_offset=[0],
        layout=layout
    )

    assert isinstance(st, ShardedTensor)
    assert st.layout == layout


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_main_replica_single_element_tuple():
    """
    Feature: is_main_replica function
    Description: Check if single element tuple with zero is considered main replica
    Expectation: Returns True for single element tuple with zero
    """
    result = is_main_replica((0,))
    assert result is True


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_main_replica_single_nonzero_element_tuple():
    """
    Feature: is_main_replica function
    Description: Check if single element tuple with nonzero value is considered main replica
    Expectation: Returns False for single element tuple with nonzero value
    """
    result = is_main_replica((1,))
    assert result is False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_sharded_tensor_from_cell_with_optimizer():
    """
    Feature: get_sharded_tensor_from_cell function with optimizer
    Description: Extract sharded tensors from a neural network cell and optimizer
    Expectation: Returns dict of ShardedTensor objects for both cell and optimizer parameters
    """
    net = SimpleNet()

    # Initialize parameters
    net.dense.weight.set_data(initializer(Normal(), net.dense.weight.shape, net.dense.weight.dtype))
    net.dense.bias.set_data(initializer('zeros', net.dense.bias.shape, net.dense.bias.dtype))

    # Create optimizer
    optim = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)

    sharded_tensors = get_sharded_tensor_from_cell(net, optim)

    # Should have weight, bias from net and optimizer states (momentum, etc.)
    assert len(sharded_tensors) >= 2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_strategy_info_from_sharded_tensor():
    """
    Feature: get_strategy_info_from_sharded_tensor function
    Description: Extract strategy information from a ShardedTensor object
    Expectation: Returns global_shape, axis_fragmentations, and global_offset as a tuple
    """
    st = ShardedTensor(
        key="test.weight",
        org_key="original.test.weight",
        dtype=ms.float32,
        local_shape=(10,),
        global_shape=(100,),
        global_offset=(5,),
        axis_fragmentations=(10,)
    )

    global_shape, axis_fragmentations, global_offset = get_strategy_info_from_sharded_tensor(st)

    assert global_shape == (100,)
    assert axis_fragmentations == (10,)
    assert global_offset == (5,)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_flatten_tensor_map():
    """
    Feature: _flatten_tensor_map function
    Description: Flatten nested tensor map structure
    Expectation: Returns flattened list of tensor map elements
    """
    # Test with nested structure
    tensor_map = [[1, 2], [3, [4, 5]], 6]
    flattened = _flatten_tensor_map(tensor_map)
    assert flattened == [1, 2, 3, 4, 5, 6]

    # Test with simple list
    tensor_map = [1, 2, 3]
    flattened = _flatten_tensor_map(tensor_map)
    assert flattened == [1, 2, 3]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_alias_name_with_rank_id():
    """
    Feature: _alias_name_with_rank_id function
    Description: Generate alias name to rank list mapping
    Expectation: Returns dictionary with alias names mapped to device numbers and rank tables
    """
    dev_matrix = [2, 2]
    alias_name = ["dp", "mp"]
    rank_list = [0, 1, 2, 3]

    result = _alias_name_with_rank_id(dev_matrix, alias_name, rank_list)

    assert "dp" in result
    assert "mp" in result
    assert len(result["dp"]) == 2
    assert len(result["mp"]) == 2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_map_with_rank_id():
    """
    Feature: _tensor_map_with_rank_id function
    Description: Map tensor dimensions to rank IDs
    Expectation: Returns list with rank tables and strides for each tensor dimension
    """
    dev_matrix = [2, 2]
    alias_name = ["dp", "mp"]
    rank_list = [0, 1, 2, 3]
    tensor_map = [0, 1]

    dev_arrange = _alias_name_with_rank_id(dev_matrix, alias_name, rank_list)

    flat_tensor_map = _flatten_tensor_map(tensor_map)
    result = _tensor_map_with_rank_id(dev_matrix, flat_tensor_map, alias_name, dev_arrange)

    assert len(result) == len(flat_tensor_map)
    # Each element should be a list with rank table and stride
    for elem in result:
        if elem is not None:
            assert len(elem) == 2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_rank_id_with_slice_id():
    """
    Feature: _rank_id_with_slice_id function
    Description: Convert alias rank stride information to rank ID vs slice ID mapping
    Expectation: Returns rank slice table and global offset tuple
    """
    # Mock alias_rank_stride data
    alias_rank_stride = [
        [[0, 1, 0, 1], 2],  # rank_table, stride
        [[0, 0, 1, 1], 1]
    ]

    rank_slice_table, global_offset = _rank_id_with_slice_id(alias_rank_stride)

    assert isinstance(rank_slice_table, list)
    assert isinstance(global_offset, tuple)
    assert len(rank_slice_table) == 4  # 4 ranks
    assert len(global_offset) == 4


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_all_sharded_tensor_on_rank0_single_rank():
    """
    Feature: get_all_sharded_tensor_on_rank0 in single-rank jobs
    Description: Test that sharded tensor metas are built locally from the layout index
    Expectation: Return the sharded tensor metas keyed by rank 0
    """
    from unittest.mock import patch, MagicMock
    from mindformers.checkpoint.sharded_tensor import get_all_sharded_tensor_on_rank0

    fake_layout_index = (
        {
            "param1": [None, "float32", (10,)],
            "param2": [None, "float32", (20,)]
        },
        {0: ["param1", "param2"]}
    )
    with patch("mindformers.checkpoint.sharded_tensor.LayoutAdapter.get_all_layouts_on_rank0",
               return_value=fake_layout_index):
        result = get_all_sharded_tensor_on_rank0(MagicMock())

    assert set(result.keys()) == {0}
    assert set(result[0].keys()) == {"param1", "param2"}
    assert result[0]["param1"].local_shape == (10,)
    assert result[0]["param1"].global_offset == (0,)
    assert result[0]["param2"].global_shape == (20,)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_all_sharded_tensor_on_rank0_no_strategy_info():
    """
    Feature: get_all_sharded_tensor_on_rank0 with no strategy info
    Description: Test that a RuntimeError is raised when no strategy metadata is available
    Expectation: RuntimeError is raised
    """
    from unittest.mock import patch, MagicMock
    from mindformers.checkpoint.sharded_tensor import get_all_sharded_tensor_on_rank0

    with patch("mindformers.checkpoint.sharded_tensor.LayoutAdapter.get_all_layouts_on_rank0",
               return_value=None):
        with pytest.raises(RuntimeError):
            get_all_sharded_tensor_on_rank0(MagicMock())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_all_sharded_tensor_on_rank0_non_rank0_returns_none():
    """
    Feature: get_all_sharded_tensor_on_rank0 on non-zero ranks
    Description: Test that non-zero ranks only contribute to the collection and get None,
        with the mode difference hidden inside LayoutAdapter
    Expectation: Return None on non-zero ranks and no global metadata is built
    """
    from unittest.mock import patch, MagicMock
    from mindformers.checkpoint.sharded_tensor import get_all_sharded_tensor_on_rank0

    with patch("mindformers.checkpoint.sharded_tensor.get_real_group_size", return_value=8), \
         patch("mindformers.checkpoint.sharded_tensor.get_real_rank", return_value=3), \
         patch("mindformers.checkpoint.sharded_tensor.LayoutAdapter.get_all_layouts_on_rank0",
               return_value=None) as mock_collect:
        result = get_all_sharded_tensor_on_rank0(MagicMock())

    mock_collect.assert_called_once()
    assert result is None


def _make_multi_rank_raw_layouts():
    """Build per-rank raw layouts covering 3D sharding, a PP split and unsharded parameters."""
    world = 16
    stage0 = list(range(0, 8))
    stage1 = list(range(8, 16))
    specs = [
        # name, rank_list, device_matrix, tensor_map, full_shape
        ("embedding.weight", list(range(world)), (4, 2, 2), (1, 0), (512, 256)),
        ("layer0.qkv.weight", stage0, (2, 2, 2), (1, 2), (256, 128)),
        ("layer1.qkv.weight", stage1, (2, 2, 2), (2, 1), (256, 128)),
    ]
    per_rank = {r: {} for r in range(world)}
    for name, rank_list, dev_matrix, tensor_map, full_shape in specs:
        raw = {
            "device_matrix": list(dev_matrix),
            "alias_name": ["dp", "tp", "sp"],
            "rank_list": list(rank_list),
            "tensor_map": tensor_map,
            "type": ms.bfloat16,
            "full_shape": full_shape,
        }
        for rank_id in rank_list:
            per_rank[rank_id][name] = raw
    for rank_id in stage0:
        per_rank[rank_id]["layer0.norm.weight"] = {"type": ms.float32, "full_shape": (128,)}
    for rank_id in stage1:
        per_rank[rank_id]["layer1.norm.weight"] = {"type": ms.float32, "full_shape": (128,)}
    return world, per_rank


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_layout_index_matches_full_global_layout():
    """
    Feature: Deduplicated layout index used by get_all_sharded_tensor_on_rank0
    Description: Test that building the global sharded tensor metadata from one layout per
        parameter is identical to building it from every rank's full layout
    Expectation: Both paths produce the same parameters, order and ShardedTensor fields
    """
    from unittest.mock import patch
    import mindformers.checkpoint.layout_adapter as layout_adapter_module
    from mindformers.checkpoint.layout_adapter import LayoutAdapter
    from mindformers.checkpoint.sharded_tensor import (
        _build_all_sharded_tensor_from_global_layout,
        _build_all_sharded_tensor_from_layout_index,
    )

    world, per_rank = _make_multi_rank_raw_layouts()

    for filter_func in (None, lambda name: "norm" not in name, lambda name: name.startswith("layer1")):
        with patch("mindformers.checkpoint.sharded_tensor.get_real_group_size", return_value=world):
            expected = _build_all_sharded_tensor_from_global_layout(
                LayoutAdapter._convert_global_layout_dict({r: per_rank[r] for r in range(world)}),
                filter_func
            )

            contributions = []
            for rank_id in range(world):
                with patch.object(layout_adapter_module, "get_real_rank", return_value=rank_id):
                    contributions.append(
                        LayoutAdapter._build_local_layout_contribution({rank_id: per_rank[rank_id]})
                    )
            param_layouts, rank_param_names = LayoutAdapter._merge_layout_contributions(contributions)
            actual = _build_all_sharded_tensor_from_layout_index(param_layouts, rank_param_names, filter_func)

        # One ms.Layout per parameter instead of one per (rank, parameter).
        assert len(param_layouts) == 5
        assert list(expected.keys()) == list(actual.keys())
        for rank_id in expected:
            assert tuple(expected[rank_id].keys()) == tuple(actual[rank_id].keys())
            for param_name, expected_tensor in expected[rank_id].items():
                actual_tensor = actual[rank_id][param_name]
                assert actual_tensor.local_shape == expected_tensor.local_shape
                assert actual_tensor.global_shape == expected_tensor.global_shape
                assert actual_tensor.global_offset == expected_tensor.global_offset
                assert actual_tensor.axis_fragmentations == expected_tensor.axis_fragmentations
                assert actual_tensor.replica_id == expected_tensor.replica_id
                assert actual_tensor.dtype == expected_tensor.dtype
                if expected_tensor.layout is None:
                    assert actual_tensor.layout is None
                else:
                    assert actual_tensor.layout.to_dict() == expected_tensor.layout.to_dict()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_build_local_layout_contribution_sends_owned_layouts_only():
    """
    Feature: LayoutAdapter._build_local_layout_contribution
    Description: Test that a rank reports all its parameter names but only ships the layout of the
        sharded parameters it owns, i.e. those whose rank list starts at this rank
    Expectation: Only the lowest rank of a rank list ships that parameter's layout
    """
    from unittest.mock import patch
    import mindformers.checkpoint.layout_adapter as layout_adapter_module
    from mindformers.checkpoint.layout_adapter import LayoutAdapter

    _, per_rank = _make_multi_rank_raw_layouts()

    with patch.object(layout_adapter_module, "get_real_rank", return_value=0):
        names_rank0, owned_rank0 = LayoutAdapter._build_local_layout_contribution({0: per_rank[0]})
    with patch.object(layout_adapter_module, "get_real_rank", return_value=5):
        names_rank5, owned_rank5 = LayoutAdapter._build_local_layout_contribution({5: per_rank[5]})
    with patch.object(layout_adapter_module, "get_real_rank", return_value=8):
        _, owned_rank8 = LayoutAdapter._build_local_layout_contribution({8: per_rank[8]})

    # Every rank reports all the names it holds, so no parameter can be lost.
    assert names_rank0 == {0: list(per_rank[0].keys())}
    assert names_rank5 == {5: list(per_rank[5].keys())}

    # Rank 0 starts both rank lists it belongs to, rank 5 starts neither.
    assert set(owned_rank0) == {"embedding.weight", "layer0.qkv.weight", "layer0.norm.weight"}
    assert set(owned_rank5) == {"layer0.norm.weight"}
    assert set(owned_rank8) == {"layer1.qkv.weight", "layer1.norm.weight"}


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_merge_layout_contributions_detects_missing_owner():
    """
    Feature: LayoutAdapter._merge_layout_contributions
    Description: Test that a parameter reported by a rank but owned by none is reported loudly
        instead of being silently dropped from 'metadata.json'
    Expectation: RuntimeError naming the missing parameter is raised
    """
    from mindformers.checkpoint.layout_adapter import LayoutAdapter

    contributions = [
        ({0: ["param1", "param2"]}, {"param1": {"type": "float32", "full_shape": (10,)}}),
    ]
    with pytest.raises(RuntimeError, match="param2"):
        LayoutAdapter._merge_layout_contributions(contributions)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_all_layouts_on_rank0_graph_builds_index_locally():
    """
    Feature: LayoutAdapter.get_all_layouts_on_rank0 in Graph mode
    Description: rank 0 builds the deduplicated layout index from the local compiler
        strategy metadata without communication; non-zero ranks get None
    Expectation: Index holds one layout entry per parameter plus per-rank name lists
    """
    from unittest.mock import patch, MagicMock
    from mindformers.checkpoint.layout_adapter import LayoutAdapter

    fake_graph_layout = {
        0: {"param1": ["layout1", "float32", (10,)], "param2": ["layout2", "float32", (20,)]},
        1: {"param1": ["layout1", "float32", (10,)]},
    }
    with patch("mindformers.checkpoint.layout_adapter.LayoutAdapter.is_pynative_mode",
               return_value=False), \
         patch("mindformers.checkpoint.layout_adapter.get_real_rank", return_value=0), \
         patch("mindformers.checkpoint.layout_adapter.LayoutAdapter._get_layout_from_graph",
               return_value=fake_graph_layout):
        layout_index = LayoutAdapter.get_all_layouts_on_rank0(MagicMock())

    param_layouts, rank_param_names = layout_index
    assert param_layouts == {"param1": ["layout1", "float32", (10,)],
                             "param2": ["layout2", "float32", (20,)]}
    assert rank_param_names == {0: ["param1", "param2"], 1: ["param1"]}

    with patch("mindformers.checkpoint.layout_adapter.LayoutAdapter.is_pynative_mode",
               return_value=False), \
         patch("mindformers.checkpoint.layout_adapter.get_real_rank", return_value=1):
        assert LayoutAdapter.get_all_layouts_on_rank0(MagicMock()) is None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_current_layout_pynative_is_local_only():
    """
    Feature: LayoutAdapter.get_current_layout in PyNative mode
    Description: Test that the current rank's layout is obtained locally from hyper_parallel
        without gathering the global layout
    Expectation: hyper_parallel's get_current_layout is used and get_global_layout is never called
    """
    from unittest.mock import patch, MagicMock
    from mindformers.checkpoint.layout_adapter import LayoutAdapter

    raw_local_layout = {
        "0": {
            "param1": {"type": "float32", "full_shape": (10,)},
        }
    }
    with patch("mindformers.checkpoint.layout_adapter.LayoutAdapter.is_pynative_mode",
               return_value=True), \
         patch("mindformers.checkpoint.layout_adapter.get_current_layout",
               return_value=raw_local_layout) as mock_local, \
         patch("mindformers.checkpoint.layout_adapter.get_global_layout") as mock_global, \
         patch("mindformers.checkpoint.layout_adapter.get_real_rank", return_value=0):
        result = LayoutAdapter.get_current_layout(MagicMock())

    mock_local.assert_called_once()
    mock_global.assert_not_called()
    assert result == {"param1": [None, "float32", (10,)]}
