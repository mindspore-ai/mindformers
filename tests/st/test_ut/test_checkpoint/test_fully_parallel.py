#  Copyright 2024 Huawei Technologies Co., Ltd
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""Test for fully_parallel.py"""
# pylint: disable=W0621, W0212, W0613
import os
from unittest.mock import patch, MagicMock

import pytest
from mindspore import nn

from mindformers.checkpoint.utils import FileType
from mindformers.checkpoint.fully_parallel import (
    BalancedSaveStrategy,
    BalancedShardPlan,
    distribute_shards,
    apply_balance_shard_strategy
)


class MockShardTensor:
    """Mock ShardTensor class for testing"""

    def __init__(self, key, global_offset, local_shape, dtype, size=100):
        self.key = key
        self.global_offset = global_offset
        self.local_shape = local_shape
        self.dtype = dtype
        self.size = size


@pytest.fixture
def mock_network():
    """Create a mock network for testing"""
    network = MagicMock(spec=nn.Cell)
    return network


@pytest.fixture
def mock_get_all_sharded_tensor():
    """Mock get_all_sharded_tensor function"""
    mock_shard_tensor1 = MockShardTensor("param1", (0,), (10,), "float32")
    mock_shard_tensor2 = MockShardTensor("param2", (10,), (10,), "float32")
    mock_shard_tensor3 = MockShardTensor("param3", (0,), (10,), "float32")

    with patch("mindformers.checkpoint.fully_parallel.get_all_sharded_tensor") as mock:
        mock.return_value = {
            0: {"param1": mock_shard_tensor1, "param2": mock_shard_tensor2},
            1: {"param3": mock_shard_tensor3}
        }
        yield mock


@pytest.fixture
def mock_get_all_sharded_tensor_on_rank0():
    """Mock get_all_sharded_tensor_on_rank0 function"""
    mock_shard_tensor1 = MockShardTensor("param1", (0,), (10,), "float32")
    mock_shard_tensor2 = MockShardTensor("param2", (10,), (10,), "float32")
    mock_shard_tensor3 = MockShardTensor("param3", (0,), (10,), "float32")

    with patch("mindformers.checkpoint.fully_parallel.get_all_sharded_tensor_on_rank0") as mock:
        mock.return_value = {
            0: {"param1": mock_shard_tensor1, "param2": mock_shard_tensor2},
            1: {"param3": mock_shard_tensor3}
        }
        yield mock


@pytest.fixture
def mock_get_real_rank():
    """Mock get_real_rank function"""
    with patch("mindformers.checkpoint.fully_parallel.get_real_rank") as mock:
        mock.return_value = 0
        yield mock


@pytest.fixture
def mock_save_checkpoint():
    """Mock save_checkpoint function"""
    with patch("mindformers.checkpoint.fully_parallel.save_checkpoint") as mock:
        yield mock


@pytest.fixture
def mock_get_metadata_filename():
    """Mock get_metadata_filename function"""
    with patch("mindformers.checkpoint.fully_parallel.get_metadata_filename") as mock:
        mock.return_value = "metadata.json"
        yield mock


@pytest.fixture
def mock_get_checkpoint_name():
    """Mock get_checkpoint_name function"""
    with patch("mindformers.checkpoint.fully_parallel.get_checkpoint_name") as mock:
        mock.return_value = "checkpoint_0-2"
        yield mock


@pytest.fixture
def mock_get_checkpoint_iter_dir():
    """Mock get_checkpoint_iter_dir function"""
    with patch("mindformers.checkpoint.fully_parallel.get_checkpoint_iter_dir") as mock:
        mock.return_value = "./checkpoint_iter_0"
        yield mock


@pytest.fixture
def mock_save_metadata():
    """Mock save_metadata function"""
    with patch("mindformers.checkpoint.fully_parallel.save_metadata") as mock:
        yield mock


@pytest.fixture
def mock_reverse_sharded_tensor_shard_id():
    """Mock _reverse_sharded_tensor_shard_id function"""
    with patch("mindformers.checkpoint.fully_parallel._reverse_sharded_tensor_shard_id") as mock:
        mock.return_value = "param1"
        yield mock


@pytest.fixture
def mock_sharded_tensor_shard_id():
    """Mock sharded_tensor_shard_id function"""
    with patch("mindformers.checkpoint.fully_parallel.sharded_tensor_shard_id") as mock:
        mock.side_effect = lambda key, offset: f"{key}_{offset}"
        yield mock


@pytest.fixture
def mock_get_shard_size():
    """Mock _get_shard_size function"""
    with patch("mindformers.checkpoint.fully_parallel._get_shard_size") as mock:
        mock.return_value = 100
        yield mock


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_distribute_shards_basic():
    """
    Feature: distribute_shards function basic functionality
    Description: Test distribute_shards function with basic input data, including different shard coverage and sizes
    Expectation: All shards are assigned to valid ranks, and each shard is assigned to a rank that covers it
    """
    shard_coverage = {
        "shard1": [0, 1],
        "shard2": [0],
        "shard3": [1]
    }
    shard_sizes = {
        "shard1": 100,
        "shard2": 200,
        "shard3": 150
    }
    total_ranks = 2

    result = distribute_shards(shard_coverage, shard_sizes, total_ranks)

    # Check that all shards are assigned
    assert len(result) == 3
    for shard_id, rank_info in result.items():
        selected_rank, rank_group = rank_info
        # Check that each shard is assigned to a valid rank
        assert 0 <= selected_rank < total_ranks
        # Check that shards are assigned to ranks that cover them
        assert selected_rank in shard_coverage[shard_id]
        assert rank_group == tuple(shard_coverage[shard_id])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_distribute_shards_empty():
    """
    Feature: distribute_shards function with empty input
    Description: Test distribute_shards function when shard_coverage and shard_sizes are empty
    Expectation: Return an empty dictionary
    """
    shard_coverage = {}
    shard_sizes = {}
    total_ranks = 2

    result = distribute_shards(shard_coverage, shard_sizes, total_ranks)

    assert result == {}


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_distribute_shards_single_rank():
    """
    Feature: distribute_shards function with single rank
    Description: Test distribute_shards function when there is only one rank available
    Expectation: All shards are assigned to the single rank
    """
    shard_coverage = {
        "shard1": [0],
        "shard2": [0]
    }
    shard_sizes = {
        "shard1": 100,
        "shard2": 200
    }
    total_ranks = 1

    result = distribute_shards(shard_coverage, shard_sizes, total_ranks)

    assert result == {"shard1": (0, (0,)), "shard2": (0, (0,))}


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_balance_shard_strategy(
        mock_network, mock_get_all_sharded_tensor, mock_get_real_rank,
        mock_sharded_tensor_shard_id, mock_get_shard_size
):
    """
    Feature: apply_balance_shard_strategy function
    Description: Test apply_balance_shard_strategy function with mock network and related fixtures
    Expectation: Return three dictionaries: shard_to_saving_rank, shard_id_to_tensor, and dst_sharded_tensor_metas
    """
    result = apply_balance_shard_strategy(mock_network, None)

    assert isinstance(result, dict)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_init(mock_network, mock_get_real_rank):
    """
    Feature: BalancedSaveStrategy initialization
    Description: Test BalancedSaveStrategy class initialization with various parameters
    Expectation: All attributes are correctly set according to the input parameters
    """

    strategy = BalancedSaveStrategy(
        network=mock_network,
        user_prefix="test",
        do_cache_distribution=True,
        checkpoint_path="./checkpoint"
    )

    assert strategy.network == mock_network
    assert strategy.user_prefix == "test"
    assert strategy.do_cache_distribution is True
    assert strategy.cached_distribution is None
    assert strategy.checkpoint_path == "./checkpoint"
    assert strategy.file_type == FileType.MODEL


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_apply_saving_parallelization(
        mock_network, mock_get_real_rank, mock_get_all_sharded_tensor,
        mock_sharded_tensor_shard_id, mock_get_shard_size
):
    """
    Feature: BalancedSaveStrategy.apply_saving_parallelization method
    Description: Test apply_saving_parallelization method without cache
    Expectation: Return a BalancedShardPlan with the current rank's shard assignment
    """
    strategy = BalancedSaveStrategy(
        network=mock_network,
        checkpoint_path="./checkpoint"
    )

    result = strategy.apply_saving_parallelization()

    assert isinstance(result, BalancedShardPlan)
    assert isinstance(result.cur_rank_shards, dict)
    assert result.total_files_num >= 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_apply_saving_parallelization_with_cache(
        mock_network, mock_get_real_rank, mock_get_all_sharded_tensor,
        mock_sharded_tensor_shard_id, mock_get_shard_size
):
    """
    Feature: BalancedSaveStrategy.apply_saving_parallelization method with cache
    Description: Test apply_saving_parallelization method with cache enabled
    Expectation: First call computes distribution, second call uses cached distribution without recomputing
    """
    strategy = BalancedSaveStrategy(
        network=mock_network,
        do_cache_distribution=True,
        checkpoint_path="./checkpoint"
    )

    # First call - should compute distribution
    result1 = strategy.apply_saving_parallelization()

    # Second call - should use cached distribution
    with patch("mindformers.checkpoint.fully_parallel.apply_balance_shard_strategy") as mock_apply:
        mock_apply.return_value = {}
        result2 = strategy.apply_saving_parallelization()
        # Check that apply_balance_shard_strategy was not called again
        mock_apply.assert_not_called()

    assert result1 == result2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_get_total_files(
        mock_network, mock_get_real_rank, mock_get_all_sharded_tensor,
        mock_sharded_tensor_shard_id, mock_get_shard_size
):
    """
    Feature: BalancedSaveStrategy.get_total_files method
    Description: Test get_total_files method to get the total number of checkpoint files
    Expectation: Return a non-negative integer representing the total number of files
    """
    strategy = BalancedSaveStrategy(
        network=mock_network,
        checkpoint_path="./checkpoint"
    )

    total_files = strategy.get_total_files()

    assert isinstance(total_files, int)
    assert total_files >= 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_get_cur_rank_file_id(
        mock_network, mock_get_real_rank, mock_get_all_sharded_tensor,
        mock_sharded_tensor_shard_id, mock_get_shard_size
):
    """
    Feature: BalancedSaveStrategy.get_cur_rank_file_id method
    Description: Test get_cur_rank_file_id method to get the current rank's file ID
    Expectation: Return a non-negative integer representing the current rank's file ID
    """
    strategy = BalancedSaveStrategy(
        network=mock_network,
        checkpoint_path="./checkpoint"
    )

    cur_rank_file_id = strategy.get_cur_rank_file_id()

    assert isinstance(cur_rank_file_id, int)
    assert cur_rank_file_id >= 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_save(
        tmp_path, mock_network, mock_get_real_rank, mock_get_all_sharded_tensor,
        mock_get_all_sharded_tensor_on_rank0,
        mock_sharded_tensor_shard_id, mock_get_shard_size, mock_save_checkpoint,
        mock_get_metadata_filename, mock_get_checkpoint_name, mock_get_checkpoint_iter_dir,
        mock_save_metadata, mock_reverse_sharded_tensor_shard_id
):
    """
    Feature: BalancedSaveStrategy.save method
    Description: Test save method saves the model checkpoint and returns the plan;
        metadata writing is the caller's responsibility (save_balanced_metadata)
    Expectation: save_checkpoint is called, get_checkpoint_iter_dir is called, get_checkpoint_name is called
    """
    checkpoint_path = str(tmp_path / "checkpoint")
    os.makedirs(checkpoint_path, exist_ok=True)

    strategy = BalancedSaveStrategy(
        network=mock_network,
        checkpoint_path=checkpoint_path
    )

    with patch("mindformers.checkpoint.fully_parallel.os.path.exists", return_value=False):
        plan = strategy.save(0)

    # Check that save_checkpoint was called
    mock_save_checkpoint.assert_called_once()
    # Check that get_checkpoint_iter_dir was called
    mock_get_checkpoint_iter_dir.assert_called_once_with(checkpoint_path, 0)
    # Check that get_checkpoint_name was called
    mock_get_checkpoint_name.assert_called()
    # The built plan is returned for the caller to write metadata
    assert plan is not None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_shard_plan_properties(
        mock_network, mock_get_real_rank
):
    """
    Feature: BalancedShardPlan properties
    Description: Test cur_rank_param_names and cur_rank_sharded_tensors properties
    Expectation: Return parameter names and ShardedTensor mapping of the current rank's shards
    """
    _ = mock_network, mock_get_real_rank

    mock_tensor1 = MagicMock()
    mock_tensor1.key = "param1"
    mock_tensor3 = MagicMock()
    mock_tensor3.key = "param3"

    plan = BalancedShardPlan(
        cur_rank_shards={"shard1": (mock_tensor1, (0,)), "shard3": (mock_tensor3, (0,))}
    )

    assert set(plan.cur_rank_param_names) == {"param1", "param3"}
    assert set(plan.cur_rank_sharded_tensors.keys()) == {"param1", "param3"}
    assert plan.cur_rank_sharded_tensors["param1"] is mock_tensor1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_file_numbering(
        mock_network, mock_get_real_rank
):
    """
    Feature: BalancedSaveStrategy file numbering
    Description: Test get_total_files and get_cur_rank_file_id derived from the balanced plan
    Expectation: Return correct total file number and current rank's file ID
    """
    mock_tensor1 = MagicMock()
    mock_tensor1.key = "param1"
    mock_tensor2 = MagicMock()
    mock_tensor2.key = "param2"
    mock_tensor3 = MagicMock()
    mock_tensor3.key = "param3"

    # rank 0 owns two shards, rank 1 owns none (absent), rank 2 owns one shard.
    shared_distribution = {
        0: {"shard1": (mock_tensor1, (0,)), "shard3": (mock_tensor3, (0,))},
        2: {"shard2": (mock_tensor2, (2,))}
    }

    with patch("mindformers.checkpoint.fully_parallel.apply_balance_shard_strategy") as mock_apply:
        mock_apply.return_value = shared_distribution

        strategy = BalancedSaveStrategy(
            network=mock_network,
            checkpoint_path="./checkpoint"
        )
        assert strategy.get_total_files() == 2
        assert strategy.get_cur_rank_file_id() == 0

        with patch("mindformers.checkpoint.fully_parallel.get_real_rank", return_value=2):
            strategy_rank2 = BalancedSaveStrategy(
                network=mock_network,
                checkpoint_path="./checkpoint"
            )
            assert strategy_rank2.get_total_files() == 2
            assert strategy_rank2.get_cur_rank_file_id() == 1

        with patch("mindformers.checkpoint.fully_parallel.get_real_rank", return_value=1):
            strategy_rank1 = BalancedSaveStrategy(
                network=mock_network,
                checkpoint_path="./checkpoint"
            )
            assert strategy_rank1.get_total_files() == 2
            assert strategy_rank1.get_cur_rank_file_id() is None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_plan_param_redundancy(
        mock_network, mock_get_real_rank
):
    """
    Feature: BalancedShardPlan param redundancy
    Description: Test that the plan carries the redundant rank groups containing the current rank
    Expectation: param_redundancy only contains groups including the current rank
    """
    mock_tensor1 = MagicMock()
    mock_tensor1.key = "param1"
    mock_tensor2 = MagicMock()
    mock_tensor2.key = "param2"

    # shard1 is redundantly held by ranks (0, 1) and assigned to rank 0;
    # shard2 is only held by rank 1 and assigned to rank 1.
    shared_distribution = {
        0: {"shard1": (mock_tensor1, (0, 1))},
        1: {"shard2": (mock_tensor2, (1,))}
    }

    with patch("mindformers.checkpoint.fully_parallel.apply_balance_shard_strategy") as mock_apply:
        mock_apply.return_value = shared_distribution

        strategy = BalancedSaveStrategy(
            network=mock_network,
            checkpoint_path="./checkpoint"
        )
        plan = strategy.apply_saving_parallelization()

        assert plan.param_redundancy == {(0, 1): ["param1"]}
        assert set(plan.cur_rank_shards.keys()) == {"shard1"}
        assert plan.full_assignment == {0: {"shard1": (0, 1)}, 1: {"shard2": (1,)}}


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_save_strategy_get_total_files_and_cur_rank_file_id(
        mock_network, mock_get_real_rank, mock_get_all_sharded_tensor,
        mock_sharded_tensor_shard_id, mock_get_shard_size
):
    """
    Feature: BalancedSaveStrategy.get_total_files and get_cur_rank_file_id methods with caching
    Description: Test that calling get_total_files and get_cur_rank_file_id caches the results
    Expectation: Second calls to these methods should use cached values without recomputing
    """
    strategy = BalancedSaveStrategy(
        network=mock_network,
        checkpoint_path="./checkpoint"
    )

    total_files = strategy.get_total_files()
    cur_rank_file_id = strategy.get_cur_rank_file_id()

    # Check that values are cached
    assert strategy.total_files_num == total_files
    assert strategy.cur_rank_file_id == cur_rank_file_id

    # Check that second calls use cached values
    with patch("mindformers.checkpoint.fully_parallel.apply_balance_shard_strategy") as mock_apply:
        mock_apply.return_value = {}
        total_files2 = strategy.get_total_files()
        cur_rank_file_id2 = strategy.get_cur_rank_file_id()

        # Mock should not be called since values are cached
        mock_apply.assert_not_called()

        assert total_files2 == total_files
        assert cur_rank_file_id2 == cur_rank_file_id
