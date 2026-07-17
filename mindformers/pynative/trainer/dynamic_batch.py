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
"""GBS growth scheduler for Dynamic Batch Training."""

import math
from typing import Optional

from mindformers.tools.logger import logger


class BatchSizeScheduler:
    """GBS scheduler driven by ``consumed_samples``.

    Args:
        rampup_batch_size (list[int]): ``[start_gbs, batch_size_increment,
            rampup_samples]``.
            ``start_gbs`` is the initial GBS at step 0.
        global_batch_size (int): ``training.global_batch_size``, the GBS upper bound (max_gbs).
        base_units (int): ``data_parallel * local_batch_size`` — samples per gradient
            accumulation step. GBS at every level must be divisible by ``base_units`` so
            ``num_accumulation_steps`` is integral.
    """

    def __init__(self, rampup_batch_size, global_batch_size: int, base_units: int):
        start_gbs, batch_size_increment, rampup_samples = rampup_batch_size
        self.base_units = int(base_units)
        self.start_gbs = int(start_gbs)
        self.max_gbs = int(global_batch_size)
        self.rampup_samples = int(rampup_samples)
        self.batch_size_increment = int(batch_size_increment)
        self.num_increments = self._num_increments_to_cap()
        self.samples_per_increment = (
            self.rampup_samples / self.num_increments
            if (self.num_increments > 0 and self.rampup_samples > 0)
            else 0.0
        )
        self._validate()

    def _num_increments_to_cap(self) -> int:
        """Number of increment levels to first reach ``max_gbs`` (0 when ``start >= max``
        or ``batch_size_increment`` invalid; ``_validate`` raises on the latter)."""
        if self.start_gbs >= self.max_gbs or self.batch_size_increment <= 0:
            return 0
        return (self.max_gbs - self.start_gbs) // self.batch_size_increment

    def _raw_gbs_at_increment(self, k: int) -> int:
        """Uncapped GBS at increment level ``k`` = ``start_gbs + k * batch_size_increment``."""
        return self.start_gbs + k * self.batch_size_increment

    def gbs_at_consumed(self, consumed_samples: int) -> int:
        """GBS at ``consumed_samples`` (capped at ``max_gbs``). """
        if consumed_samples > self.rampup_samples or self.samples_per_increment <= 0:
            return self.max_gbs
        steps = int(consumed_samples / self.samples_per_increment)
        return min(self._raw_gbs_at_increment(steps), self.max_gbs)

    def num_accum_at_consumed(self, consumed_samples: int) -> int:
        """Gradient accumulation steps at ``consumed_samples`` = ``gbs // base_units``.

        Divisibility is guaranteed by the startup pre-scan in ``_validate``."""
        return self.gbs_at_consumed(consumed_samples) // self.base_units

    def consumed_samples_at_step(self, step: int) -> int:
        """Total samples consumed after ``step`` optimizer steps — the inverse of the
        runtime recurrence ``consumed(s+1) = consumed(s) + gbs(consumed(s))``. Used as
        the resume fallback when a checkpoint lacks persisted ``consumed_samples``.
        O(num_increments)."""
        if step <= 0:
            return 0
        # No rampup: GBS is max_gbs throughout.
        if self.samples_per_increment <= 0:
            return step * self.max_gbs
        consumed, remaining = 0, step
        for k in range(self.num_increments):
            gbs_k = self._raw_gbs_at_increment(k)
            boundary = (k + 1) * self.samples_per_increment
            if consumed >= boundary:
                continue  # overshoot from the previous level
            steps_to_end = math.ceil((boundary - consumed) / gbs_k)
            if remaining <= steps_to_end:
                return consumed + remaining * gbs_k
            consumed += steps_to_end * gbs_k
            remaining -= steps_to_end
        return consumed + remaining * self.max_gbs  # capped phase

    def _validate(self) -> None:
        """Startup validation: config sanity + per-level divisibility (fail-fast).

        Ordered so divisibility checks (which crash on a zero ``batch_size_increment``)
        run after the increment-range check.
        ``(max-start) % increment == 0`` and every GBS divisible by ``base_units``."""
        if self.base_units <= 0:
            raise ValueError(f"data_parallel*local_batch_size must be positive, got {self.base_units}.")
        if self.max_gbs <= 0:
            raise ValueError(
                f"training.global_batch_size must be positive, got {self.max_gbs}."
            )
        if self.batch_size_increment <= 0:
            raise ValueError(
                f"rampup_batch_size[1] (batch_size_increment) must be > 0, "
                f"got {self.batch_size_increment}."
            )
        if self.rampup_samples < 0:
            raise ValueError(
                f"rampup_batch_size[2] (rampup_samples) must be >= 0, "
                f"got {self.rampup_samples}."
            )
        if self.start_gbs <= 0:
            raise ValueError(
                f"rampup_batch_size[0] start_global_batch_size({self.start_gbs}) must be positive."
            )
        if self.max_gbs < self.start_gbs:
            raise ValueError(
                f"training.global_batch_size({self.max_gbs}) must be >= "
                f"rampup_batch_size[0] start_global_batch_size({self.start_gbs})."
            )
        diff = self.max_gbs - self.start_gbs
        if diff % self.batch_size_increment != 0:
            raise ValueError(
                f"(global_batch_size - start_global_batch_size)={diff} must be divisible by "
                f"batch_size_increment={self.batch_size_increment}."
            )
        if self.start_gbs % self.base_units != 0:
            raise ValueError(
                f"rampup_batch_size[0] start_global_batch_size({self.start_gbs}) must be divisible by "
                f"data_parallel*local_batch_size({self.base_units})."
            )
        if self.max_gbs % self.base_units != 0:
            raise ValueError(
                f"training.global_batch_size({self.max_gbs}) must be divisible by "
                f"data_parallel*local_batch_size({self.base_units})."
            )
        # Pre-scan divisibility of every intermediate GBS during growth (fail-fast).
        for k in range(self.num_increments):
            raw_gbs = self._raw_gbs_at_increment(k)
            if raw_gbs % self.base_units != 0:
                raise ValueError(
                    f"rampup produces gbs={raw_gbs} at increment k={k}, which is not "
                    f"divisible by data_parallel*local_batch_size({self.base_units}). "
                    f"Adjust batch_size_increment so every intermediate gbs is divisible."
                )


def build_dynamic_scheduler(
    config, global_batch_size: int, base_units: int
) -> Optional[BatchSizeScheduler]:
    """Build a scheduler from the top-level config; returns None when not enabled.

    Args:
        config: Top-level ``TrainConfig`` (must have ``training.rampup_batch_size`` field and
            ``training.global_batch_size``).
        global_batch_size (int): ``training.global_batch_size``, as max_gbs.
        base_units (int): ``data_parallel * local_batch_size``.

    Returns:
        Optional[BatchSizeScheduler]: Scheduler instance if ``training.rampup_batch_size``
        is set, else None.
    """
    rampup = getattr(config.training, "rampup_batch_size", None)
    if rampup is None:
        return None
    return BatchSizeScheduler(rampup, global_batch_size, base_units)
