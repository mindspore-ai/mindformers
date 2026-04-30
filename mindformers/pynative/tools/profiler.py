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

from typing import Optional, List
import mindspore.profiler
from mindspore.profiler import ProfilerActivity

from mindformers.tools.logger import logger
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
    """

    def __init__(self, config: ProfilerConfig):
        """
        Initialize the Profiler with configuration.

        Args:
            config: ProfilerConfig containing profiler settings.
                Expected fields:
                    - enable_profiling: bool, whether to enable profiling
                    - save_traces_folder: str, trace files location.
                    - start_step: int, step to start profiling
                    - stop_step: int, step to stop profiling
                    - profiler_level: str, profiling detail level
                    - profile_memory: bool, whether to profile memory
        """
        self.config = config
        self.profiler = None
        self._enabled = getattr(config, 'enable_profiling', False)

    def __enter__(self):
        """Enter the profiler context."""
        if not self._enabled:
            return _DummyProfiler()

        wait = self.config.profiler_skip_first_wait
        active = self.config.profiler_active
        warmup = self.config.profiler_warmup
        repeat = self.config.profiler_repeat
        skip_first = self.config.profiler_skip_first

        dir_name = self.config.save_traces_folder
        on_trace_ready = mindspore.profiler.tensorboard_trace_handler(dir_name)

        profile_memory = getattr(self.config, 'enable_memory', False)

        schedule = mindspore.profiler.schedule(
            wait=wait,
            active=active,
            warmup=warmup,
            repeat=repeat,
            skip_first=skip_first
        )

        experimental_config = None
        if hasattr(self.config, 'profiler_level'):
            profiler_level = getattr(self.config, 'profiler_level', 'Level0')
            experimental_config = mindspore.profiler.ExperimentalConfig(
                profiler_level=profiler_level
            )

        activities = [
            ProfilerActivity.CPU,
            ProfilerActivity.NPU,
        ]

        self.profiler = mindspore.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            profile_memory=profile_memory,
            experimental_config=experimental_config
        )

        if self.profiler is not None:
            self.profiler.start()
            logger.info("Profiler started.")

        start_step = skip_first + 1
        stop_step = start_step + repeat * (wait + warmup + active)
        logger.info(f"Profiler initialized: start_step={start_step}, stop_step={stop_step}, "
                    f"profile_memory={profile_memory}")

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
