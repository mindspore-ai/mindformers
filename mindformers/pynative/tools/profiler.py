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
"""Profiler module for MindFormers pynative training."""

import os
from mindspore.profiler import (
    _ExperimentalConfig,
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from mindspore.profiler.common.constant import ProfilerLevel

from mindformers.tools.logger import logger
from mindformers.tools.utils import get_real_rank
from mindformers.pynative.config.config import ProfilerConfig


class _DummyProfiler:
    def step(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Profiler:
    """
    Profiler context manager for pynative training.

    This class provides a convenient way to wrap the training loop with
    mindspore profiler, supporting configurable profiling activities,
    schedule, and output options.

    Args:
        config: ProfilerConfig containing profiler settings.
            Expected fields:
                - enable_profiling: bool, whether to enable profiling
                - start_step: int, step to start profiling
                - end_step: int, step to stop profiling
                - output_path: str, profiling result save path
                - profiler_rank: list, rank ids to enable profiling
                - profiler_level: int, profiling detail level (0/1/2)
                - profile_memory: bool, whether to profile memory
                - with_stack: bool, whether to collect stack trace
    """

    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.profiler = None
        self._enabled = getattr(config, 'enable_profiling', False)
        if self._enabled:
            self._enabled = self._is_profiler_rank()

    def _is_profiler_rank(self):
        profiler_rank = getattr(self.config, 'profiler_rank', None)
        if profiler_rank is None:
            return True
        rank_id = get_real_rank()
        return rank_id in profiler_rank

    def _get_output_path(self):
        output_path = getattr(self.config, 'output_path', None)
        rank_id = get_real_rank()
        if output_path:
            return os.path.join(output_path, f'rank_{rank_id}')
        return os.path.join(os.getcwd(), 'profile', f'rank_{rank_id}')

    def __enter__(self):
        """Enter the profiler context."""
        if not self._enabled:
            return _DummyProfiler()

        start_step = getattr(self.config, 'start_step', 1)
        end_step = getattr(self.config, 'end_step', 1)

        active = end_step - start_step + 1
        skip_first = start_step

        dir_name = self._get_output_path()
        on_trace_ready = tensorboard_trace_handler(dir_name)

        profile_memory = getattr(self.config, 'profile_memory', False)
        with_stack = getattr(self.config, 'with_stack', True)

        schedule_config = schedule(
            wait=0,
            active=active,
            warmup=0,
            repeat=1,
            skip_first=skip_first
        )

        profiler_level = getattr(self.config, 'profiler_level', 0)
        if isinstance(profiler_level, int):
            if profiler_level in (0, 1, 2):
                profiler_level = getattr(ProfilerLevel, f"Level{profiler_level}")
            else:
                logger.warning(f"Invalid profiler_level: {profiler_level}, using LevelNone instead")
                profiler_level = ProfilerLevel.LevelNone
        else:
            raise ValueError(f"Invalid profiler_level type: {type(profiler_level)}, must be int")
        mstx = getattr(self.config, 'mstx', False)
        experimental_config = _ExperimentalConfig(
            profiler_level=profiler_level,
            mstx=mstx
        )

        activities = [
            ProfilerActivity.CPU,
            ProfilerActivity.NPU,
        ]

        self.profiler = profile(
            activities=activities,
            schedule=schedule_config,
            on_trace_ready=on_trace_ready,
            profile_memory=profile_memory,
            with_stack=with_stack,
            experimental_config=experimental_config
        )

        if self.profiler is not None:
            self.profiler.start()
            self.profiler.step()
            logger.info("Profiler started.")

        logger.info(
            f"Profiler initialized: start_step={start_step}, end_step={end_step}, "
            f"profile_memory={profile_memory}, profiler_level={profiler_level}"
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit profiler context and finalize profiling."""
        if self.profiler is not None:
            self.profiler.stop()
            logger.info("Profiler stopped.")

    def step(self):
        """Notify profiler that a training step has completed."""
        if self.profiler is not None:
            self.profiler.step()
