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
"""Test module for testing pynative profiler with mock."""

import os
from unittest.mock import patch, MagicMock

import pytest
from mindspore.profiler.common.constant import ProfilerLevel

from mindformers.pynative.tools.profiler import Profiler, _DummyProfiler
from mindformers.pynative.config.config import ProfilerConfig


class TestProfilerDisabled:
    """Test profiler disabled scenario."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch("mindformers.pynative.tools.profiler.profile")
    def test_disabled_returns_dummy_profiler(self, mock_profile):
        """enable_profiling=False should return _DummyProfiler."""
        config = ProfilerConfig(enable_profiling=False, start_step=1, end_step=5)
        profiler = Profiler(config)
        with profiler as p:
            assert isinstance(p, _DummyProfiler)
            p.step()
        mock_profile.assert_not_called()


class TestProfilerLevelMapping:
    """Test profiler_level integer to ProfilerLevel enum mapping."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch("mindformers.pynative.tools.profiler.profile")
    @patch("mindformers.pynative.tools.profiler._ExperimentalConfig")
    def test_level_1(self, mock_exp_config, mock_profile):
        """profiler_level=1 should map to ProfilerLevel.Level1."""
        _ = mock_profile
        mock_profile.return_value = MagicMock()

        config = ProfilerConfig(
            enable_profiling=True, start_step=1, end_step=5,
            output_path="./profile", profiler_level=1,
        )
        with Profiler(config):
            pass

        call_kwargs = mock_exp_config.call_args[1]
        assert call_kwargs["profiler_level"] == ProfilerLevel.Level1

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch("mindformers.pynative.tools.profiler.profile")
    @patch("mindformers.pynative.tools.profiler._ExperimentalConfig")
    def test_invalid_int_maps_to_level_none(self, mock_exp_config, mock_profile):
        """profiler_level with invalid int (e.g. -1) should map to LevelNone."""
        _ = mock_profile
        mock_profile.return_value = MagicMock()

        config = ProfilerConfig(
            enable_profiling=True, start_step=1, end_step=5,
            output_path="./profile", profiler_level=-1,
        )
        with Profiler(config):
            pass

        call_kwargs = mock_exp_config.call_args[1]
        assert call_kwargs["profiler_level"] == ProfilerLevel.LevelNone

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_non_int_raises_value_error(self):
        """profiler_level with non-int type should raise ValueError."""
        config = ProfilerConfig(
            enable_profiling=True, start_step=1, end_step=5,
            output_path="./profile",
        )
        config.profiler_level = "invalid"
        profiler = Profiler(config)
        with pytest.raises(ValueError, match="Invalid profiler_level type"):
            with profiler:
                pass


class TestRankFiltering:
    """Test profiler rank filtering logic."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch("mindformers.pynative.tools.profiler.get_real_rank", return_value=0)
    def test_rank_in_list_enabled(self, mock_get_real_rank):
        """Current rank in profiler_rank list should enable profiling."""
        _ = mock_get_real_rank
        config = ProfilerConfig(
            enable_profiling=True, start_step=1, end_step=5,
            profiler_rank=[0, 1],
        )
        profiler = Profiler(config)
        assert profiler._enabled is True


class TestOutputPath:
    """Test output path generation logic."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch("mindformers.pynative.tools.profiler.get_real_rank", return_value=0)
    def test_custom_output_path(self, mock_get_real_rank):
        """Custom output_path should be used with rank subdirectory."""
        _ = mock_get_real_rank
        config = ProfilerConfig(
            enable_profiling=True, start_step=1, end_step=5,
            output_path="./profile",
        )
        profiler = Profiler(config)
        result = profiler._get_output_path()
        assert result.endswith(os.path.join("profile", "rank_0"))
