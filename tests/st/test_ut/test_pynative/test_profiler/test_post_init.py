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
"""Test module for testing pynative profiler for mindformers."""

import pytest
from mindformers.pynative.config.config import ProfilerConfig


class TestProfilerConfigPostInit:
    """Test ProfilerConfig post-initialization validation."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_post_init_normal_case(self):
        """Test normal case where start_step <= end_step and both are positive."""
        config = ProfilerConfig(start_step=5, end_step=5)
        assert config.start_step == 5
        assert config.end_step == 5

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_post_init_negative_start_step(self):
        """Test that negative start_step raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ProfilerConfig(start_step=-1, end_step=5)
        assert "start_step" in str(exc_info.value)
        assert "positive" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_post_init_negative_end_step(self):
        """Test that negative end_step raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ProfilerConfig(start_step=3, end_step=-1)
        assert "end_step" in str(exc_info.value)
        assert "positive" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_post_init_both_negative(self):
        """Test that both negative values raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ProfilerConfig(start_step=-2, end_step=-5)
        assert "positive" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_post_init_zero_start_step(self):
        """Test that zero start_step raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ProfilerConfig(start_step=0, end_step=5)
        assert "start_step" in str(exc_info.value)
        assert "positive" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_post_init_zero_end_step(self):
        """Test that zero end_step raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ProfilerConfig(start_step=5, end_step=0)
        assert "end_step" in str(exc_info.value)
        assert "positive" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_post_init_start_step_greater_than_end_step(self):
        """Test that start_step > end_step raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ProfilerConfig(start_step=10, end_step=5)
        assert "start_step must be less than or equal to end_step" in str(exc_info.value)
