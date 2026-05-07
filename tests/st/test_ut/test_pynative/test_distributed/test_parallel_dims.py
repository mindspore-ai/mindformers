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
"""Test parallel dims with various configurations"""
from unittest.mock import Mock, patch

import pytest

from mindformers.pynative.distributed.parallel_dims import ParallelDims


def tensor_to_numpy(data):
    return data.asnumpy()

@pytest.fixture(name="mock_platform")
def fixture_mock_platform():
    """Mock platform-related interfaces (avoid dependency on real hardware/distributed environment)"""
    with patch("hyper_parallel.core.dtensor.device_mesh.platform") as platform_mock:
        # Mock rank=0, world_size=8
        platform_mock.get_rank.return_value = 0
        platform_mock.get_world_size.return_value = 16
        # Mock communication group creation (return Mock object)
        mock_group = Mock()
        platform_mock.create_group.return_value = mock_group
        platform_mock.tensor_to_numpy = tensor_to_numpy
        yield platform_mock




@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestDeviceMesh:
    """Test suite for DeviceMesh class and related functions"""

    def test_build_mesh_without_ep(self, mock_platform):
        """
        Feature: init_device_mesh function.
        Description: Test basic functionality including automatic rank_list generation
            and cache mechanism.
        Expectation: Run success, mesh properties match expected values,
            same parameters return the same cached instance.
        """
        _ = mock_platform  # Ensure mock is active
        # Automatically generate rank_list from mesh_shape
        parallel_dims = ParallelDims(
            dp_replicate=1, dp_shard=2, cp=2, tp=2, pp=2,
            ep=1, etp=1,
            world_size=16
        )
        mesh = parallel_dims.world_mesh
        assert mesh.mesh_shape == (2, 1, 2, 2, 2)
        assert mesh.mesh_dim_names == ('pp', 'dp_replicate', 'dp_shard', 'cp', 'tp')
        assert mesh.rank_list == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

        assert mesh['pp'].mesh_shape == (2,)
        assert mesh['pp'].mesh_dim_names == ('pp',)
        assert mesh['pp'].rank_list == (0, 8)

        assert mesh['dp_replicate'].mesh_shape == (1,)
        assert mesh['dp_replicate'].mesh_dim_names == ('dp_replicate',)
        assert mesh['dp_replicate'].rank_list == (0,)

        assert mesh['dp_shard'].mesh_shape == (2,)
        assert mesh['dp_shard'].mesh_dim_names == ('dp_shard',)
        assert mesh['dp_shard'].rank_list == (0, 4)

        assert mesh['cp'].mesh_shape == (2,)
        assert mesh['cp'].mesh_dim_names == ('cp',)
        assert mesh['cp'].rank_list == (0, 2)

        assert mesh['tp'].mesh_shape == (2,)
        assert mesh['tp'].mesh_dim_names == ('tp',)
        assert mesh['tp'].rank_list == (0, 1)

    def test_build_mesh_with_ep_reuse_tp(self, mock_platform):
        """
        Feature: init_device_mesh function.
        Description: Test basic functionality including automatic rank_list generation
            and cache mechanism.
        Expectation: Run success, mesh properties match expected values,
            same parameters return the same cached instance.
        """
        _ = mock_platform  # Ensure mock is active
        # Automatically generate rank_list from mesh_shape
        parallel_dims = ParallelDims(
            dp_replicate=1, dp_shard=2, cp=2, tp=2, pp=2,
            ep=2, etp=1,
            world_size=16
        )
        mesh = parallel_dims.world_mesh
        assert mesh.mesh_shape == (2, 1, 2, 2, 1, 2)
        assert mesh.mesh_dim_names == ('pp', 'dp_replicate', 'dp_shard', 'cp', 'tp_mod_ep', 'ep')
        assert mesh.rank_list == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

        assert mesh['pp'].mesh_shape == (2,)
        assert mesh['pp'].mesh_dim_names == ('pp',)
        assert mesh['pp'].rank_list == (0, 8)

        assert mesh['dp_replicate'].mesh_shape == (1,)
        assert mesh['dp_replicate'].mesh_dim_names == ('dp_replicate',)
        assert mesh['dp_replicate'].rank_list == (0,)

        assert mesh['dp_shard'].mesh_shape == (2,)
        assert mesh['dp_shard'].mesh_dim_names == ('dp_shard',)
        assert mesh['dp_shard'].rank_list == (0, 4)

        assert mesh['cp'].mesh_shape == (2,)
        assert mesh['cp'].mesh_dim_names == ('cp',)
        assert mesh['cp'].rank_list == (0, 2)

        assert mesh['tp_mod_ep'].mesh_shape == (1,)
        assert mesh['tp_mod_ep'].mesh_dim_names == ('tp_mod_ep',)
        assert mesh['tp_mod_ep'].rank_list == (0,)

        assert mesh['ep'].mesh_shape == (2,)
        assert mesh['ep'].mesh_dim_names == ('ep',)
        assert mesh['ep'].rank_list == (0, 1)

    def test_build_mesh_with_ep_reuse_cp_tp(self, mock_platform):
        """
        Feature: init_device_mesh function.
        Description: Test basic functionality including automatic rank_list generation
            and cache mechanism.
        Expectation: Run success, mesh properties match expected values,
            same parameters return the same cached instance.
        """
        _ = mock_platform  # Ensure mock is active
        # Automatically generate rank_list from mesh_shape
        parallel_dims = ParallelDims(
            dp_replicate=1, dp_shard=2, cp=2, tp=2, pp=2,
            ep=4, etp=1,
            world_size=16
        )
        mesh = parallel_dims.world_mesh
        assert mesh.mesh_shape == (2, 1, 2, 1, 2, 2)
        assert mesh.mesh_dim_names == ('pp', 'dp_replicate', 'dp_shard', 'cp_tp_mod_ep', 'ep_mod_tp', 'tp')
        assert mesh.rank_list == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

        assert mesh['pp'].mesh_shape == (2,)
        assert mesh['pp'].mesh_dim_names == ('pp',)
        assert mesh['pp'].rank_list == (0, 8)

        assert mesh['dp_replicate'].mesh_shape == (1,)
        assert mesh['dp_replicate'].mesh_dim_names == ('dp_replicate',)
        assert mesh['dp_replicate'].rank_list == (0,)

        assert mesh['dp_shard'].mesh_shape == (2,)
        assert mesh['dp_shard'].mesh_dim_names == ('dp_shard',)
        assert mesh['dp_shard'].rank_list == (0, 4)

        assert mesh['cp_tp_mod_ep'].mesh_shape == (1,)
        assert mesh['cp_tp_mod_ep'].mesh_dim_names == ('cp_tp_mod_ep',)
        assert mesh['cp_tp_mod_ep'].rank_list == (0,)

        assert mesh['ep_mod_tp'].mesh_shape == (2,)
        assert mesh['ep_mod_tp'].mesh_dim_names == ('ep_mod_tp',)
        assert mesh['ep_mod_tp'].rank_list == (0, 2)

        assert mesh['tp'].mesh_shape == (1,)
        assert mesh['tp'].mesh_dim_names == ('tp',)
        assert mesh['tp'].rank_list == (0, 1)

    def test_build_mesh_with_ep_reuse_dp_cp_tp(self, mock_platform):
        """
        Feature: init_device_mesh function.
        Description: Test basic functionality including automatic rank_list generation
            and cache mechanism.
        Expectation: Run success, mesh properties match expected values,
            same parameters return the same cached instance.
        """
        _ = mock_platform  # Ensure mock is active
        # Automatically generate rank_list from mesh_shape
        parallel_dims = ParallelDims(
            dp_replicate=1, dp_shard=2, cp=2, tp=2, pp=2,
            ep=8, etp=1,
            world_size=16
        )
        mesh = parallel_dims.world_mesh
        assert mesh.mesh_shape == (2, 1, 1, 2, 2, 2)
        assert mesh.mesh_dim_names == ('pp', 'dp_replicate', 'efsdp', 'ep_mod_cp_tp', 'cp', 'tp')
        assert mesh.rank_list == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

        assert mesh['pp'].mesh_shape == (2,)
        assert mesh['pp'].mesh_dim_names == ('pp',)
        assert mesh['pp'].rank_list == (0, 8)

        assert mesh['dp_replicate'].mesh_shape == (1,)
        assert mesh['dp_replicate'].mesh_dim_names == ('dp_replicate',)
        assert mesh['dp_replicate'].rank_list == (0,)

        assert mesh['efsdp'].mesh_shape == (1,)
        assert mesh['efsdp'].mesh_dim_names == ('efsdp',)
        assert mesh['efsdp'].rank_list == (0,)

        assert mesh['ep_mod_cp_tp'].mesh_shape == (2,)
        assert mesh['ep_mod_cp_tp'].mesh_dim_names == ('ep_mod_cp_tp',)
        assert mesh['ep_mod_cp_tp'].rank_list == (0, 4)

        assert mesh['cp'].mesh_shape == (2,)
        assert mesh['cp'].mesh_dim_names == ('cp',)
        assert mesh['cp'].rank_list == (0, 2)

        assert mesh['tp'].mesh_shape == (2,)
        assert mesh['tp'].mesh_dim_names == ('tp',)
        assert mesh['tp'].rank_list == (0, 1)
