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
"""Test parallel dims with various configurations.

Aligned with the TorchTitan-ported ParallelDims contract: ``world_mesh`` is a
flat 1D mesh and sub-meshes are looked up by axis name via ``get_mesh`` (instead
of indexing a multi-dimensional ``world_mesh``). The expert region is a single
"sparse" mesh carrying ``efsdp`` / ``ep`` axes (no tp_mod_ep / cp_tp_mod_ep /
ep_mod_tp / ep_mod_cp_tp splitting). The configurations below mirror the
pre-port tests; only the assertions are adapted to the new API.
"""
from unittest.mock import Mock, patch

import pytest

from mindformers.pynative.distributed.parallel_dims import ParallelDims


def tensor_to_numpy(data):
    return data.asnumpy()


@pytest.fixture(name="mock_platform")
def fixture_mock_platform():
    """Mock platform-related interfaces (avoid dependency on real hardware/distributed environment)"""
    with patch("hyper_parallel.core.dtensor.device_mesh.platform") as platform_mock:
        platform_mock.get_rank.return_value = 0
        platform_mock.get_world_size.return_value = 16
        platform_mock.create_group.return_value = Mock()
        platform_mock.tensor_to_numpy = tensor_to_numpy
        yield platform_mock


class TestDeviceMesh:
    """Test suite for ParallelDims device-mesh construction and partitioning."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_degree_properties(self):
        """
        Feature: ParallelDims scalar degree properties consumed downstream.
        Description: parallelize.py reads ``parallel_dims.fsdp`` (= dp_shard * cp)
            when collecting MoE replicate params; guard the public attribute
            surface so a missing property is caught at UT level (not only in
            multi-card training).
        Expectation: fsdp == dp_shard * cp; enabled flags reflect the degrees.
        """
        parallel_dims = ParallelDims(
            dp_replicate=1, dp_shard=2, cp=2, tp=2, pp=1, ep=1, world_size=8
        )
        assert parallel_dims.fsdp == parallel_dims.dp_shard * parallel_dims.cp == 4
        assert parallel_dims.fsdp_enabled is True
        assert parallel_dims.cp_enabled is True
        assert parallel_dims.tp_enabled is True
        assert parallel_dims.pp_enabled is False
        assert parallel_dims.ep_enabled is False

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_mesh_without_ep(self, mock_platform):
        """
        Feature: ParallelDims.build_mesh without expert parallelism.
        Description: world_mesh is a flat 1D mesh; sub-meshes are obtained via
            get_mesh. Verify each sub-mesh groups the expected global ranks.
        Expectation: pp/cp/tp/fsdp rank_lists match; dp_replicate (size 1) is
            not enabled; the old standalone 'dp_shard' axis is folded into
            'fsdp' (= dp_shard * cp).
        """
        _ = mock_platform
        parallel_dims = ParallelDims(
            dp_replicate=1, dp_shard=2, cp=2, tp=2, pp=2, ep=1, world_size=16
        )
        mesh = parallel_dims.world_mesh
        assert mesh.mesh_shape == (16,)
        assert mesh.mesh_dim_names == ("world",)

        assert parallel_dims.get_mesh("pp").mesh_dim_names == ("pp",)
        assert parallel_dims.get_mesh("pp").rank_list == (0, 8)

        # dp_replicate has size 1 -> not enabled.
        assert parallel_dims.get_optional_mesh("dp_replicate") is None

        # The pre-port standalone 'dp_shard' axis is folded into 'fsdp'.
        assert parallel_dims.get_mesh("fsdp").rank_list == (0, 2, 4, 6)

        assert parallel_dims.get_mesh("cp").mesh_dim_names == ("cp",)
        assert parallel_dims.get_mesh("cp").rank_list == (0, 2)

        assert parallel_dims.get_mesh("tp").mesh_dim_names == ("tp",)
        assert parallel_dims.get_mesh("tp").rank_list == (0, 1)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_mesh_with_ep_reuse_tp(self, mock_platform):
        """
        Feature: ParallelDims.build_mesh with ep <= tp.
        Description: The expert region is a single sparse mesh (efsdp, ep);
            dense axes (pp/cp/tp/fsdp) are unchanged by EP.
        Expectation: ep groups (0, 1); efsdp groups (0, 2, 4, 6).
        """
        _ = mock_platform
        parallel_dims = ParallelDims(
            dp_replicate=1, dp_shard=2, cp=2, tp=2, pp=2, ep=2, world_size=16
        )
        assert parallel_dims.get_mesh("pp").rank_list == (0, 8)
        assert parallel_dims.get_mesh("cp").rank_list == (0, 2)
        assert parallel_dims.get_mesh("tp").rank_list == (0, 1)
        assert parallel_dims.get_mesh("fsdp").rank_list == (0, 2, 4, 6)

        assert parallel_dims.get_mesh("ep").mesh_dim_names == ("ep",)
        assert parallel_dims.get_mesh("ep").rank_list == (0, 1)
        assert parallel_dims.get_mesh("efsdp").rank_list == (0, 2, 4, 6)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_mesh_with_ep_reuse_cp_tp(self, mock_platform):
        """
        Feature: ParallelDims.build_mesh with ep == cp * tp.
        Description: ep absorbs both cp and tp degrees into the sparse mesh.
        Expectation: ep groups (0, 1, 2, 3); efsdp shrinks to (0, 4).
        """
        _ = mock_platform
        parallel_dims = ParallelDims(
            dp_replicate=1, dp_shard=2, cp=2, tp=2, pp=2, ep=4, world_size=16
        )
        assert parallel_dims.get_mesh("pp").rank_list == (0, 8)
        assert parallel_dims.get_mesh("cp").rank_list == (0, 2)
        assert parallel_dims.get_mesh("tp").rank_list == (0, 1)
        assert parallel_dims.get_mesh("fsdp").rank_list == (0, 2, 4, 6)

        assert parallel_dims.get_mesh("ep").rank_list == (0, 1, 2, 3)
        assert parallel_dims.get_mesh("efsdp").rank_list == (0, 4)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_mesh_with_ep_reuse_dp_cp_tp(self, mock_platform):
        """
        Feature: ParallelDims.build_mesh with ep > cp * tp.
        Description: ep additionally absorbs dp_shard; efsdp shrinks to size 1
            but is still kept (so MoE layers can be FSDP-wrapped).
        Expectation: ep groups all 8 ranks (0..7); efsdp groups (0,).
        """
        _ = mock_platform
        parallel_dims = ParallelDims(
            dp_replicate=1, dp_shard=2, cp=2, tp=2, pp=2, ep=8, world_size=16
        )
        assert parallel_dims.get_mesh("pp").rank_list == (0, 8)
        assert parallel_dims.get_mesh("cp").rank_list == (0, 2)
        assert parallel_dims.get_mesh("tp").rank_list == (0, 1)

        assert parallel_dims.get_mesh("ep").rank_list == (0, 1, 2, 3, 4, 5, 6, 7)
        assert parallel_dims.get_mesh("efsdp").rank_list == (0,)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_ep_exceeds_region_is_rejected(self):
        """
        Feature: ParallelDims validation when ep does not divide the EP region.
        Description: EP is carved out of the (dp_shard * cp * tp) region within a
            single dp_replicate group, so ``efsdp = dp_shard * cp * tp // ep``.
            With dp_replicate=4, dp_shard=2, cp=1, tp=2 (16 cards), the region is
            only dp_shard*cp*tp = 4, so ep=8 cannot be satisfied; without the
            guard, efsdp would silently floor to 0 and fail later as an opaque
            mesh-size mismatch.
        Expectation: construction raises ValueError at validation time (before any
            mesh is built), pointing at the EP-region constraint.
        """
        with pytest.raises(ValueError, match=r"ep\(8\) must divide"):
            ParallelDims(
                dp_replicate=4, dp_shard=2, cp=1, tp=2, pp=1, ep=8, world_size=16
            )


class TestParallelDimsFromConfig:
    """New cases: ParallelDims.from_config maps MindFormers field semantics."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_config_pure_fsdp(self):
        """
        Feature: from_config with default data_parallel_shard (-1).
        Description: No replicate; shard over the whole DP group.
        Expectation: dp_replicate=1, dp_shard=data_parallel.
        """
        # world_size=8, tp=2 -> data_parallel=4 (computed by the trainer beforehand).
        parallelism = _FakeParallelism(data_parallel=4, data_parallel_shard=-1, tp=2)
        parallel_dims = ParallelDims.from_config(parallelism, world_size=8)
        assert (parallel_dims.dp_replicate, parallel_dims.dp_shard) == (1, 4)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_config_hsdp(self):
        """
        Feature: from_config with HSDP (data_parallel_shard > 0).
        Description: Shard within data_parallel_shard ranks, replicate the rest.
        Expectation: dp_replicate * dp_shard == data_parallel.
        """
        # data_parallel=8, shard=2 -> replicate=4, shard=2; product 8.
        parallelism = _FakeParallelism(data_parallel=8, data_parallel_shard=2)
        parallel_dims = ParallelDims.from_config(parallelism, world_size=8)
        assert (parallel_dims.dp_replicate, parallel_dims.dp_shard) == (4, 2)


class _FakeParallelism:
    """Minimal stand-in for MindFormers ParallelismConfig."""

    def __init__(self, data_parallel, data_parallel_shard, tp=1, cp=1, pp=1, ep=1):
        self.data_parallel = data_parallel
        self.data_parallel_shard = data_parallel_shard
        self.tensor_parallel = tp
        self.context_parallel = cp
        self.pipeline_parallel = pp
        self.expert_parallel = ep
