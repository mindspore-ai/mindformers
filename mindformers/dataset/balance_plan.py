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
"""Pure step planning with baseline micro-batches as indivisible scheduling units."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StepSpec:
    """Immutable data-step identity; prefetch advances only virtual copies."""

    step_id: int
    consumed_samples: int
    micro_cursor: int
    global_batch_size: int
    data_parallel: int
    local_batch_size: int
    micro_batch_num: int


@dataclass(frozen=True)
class StepPlan:
    """Global rank/micro layouts, preserving each baseline micro-batch's contents."""

    spec: StepSpec
    indices: tuple
    epochs: tuple
    balanced: bool

    def next_state(self) -> dict:
        """Return the exact committed boundary after successful consumption."""
        return {
            'step_id': self.spec.step_id + 1,
            'consumed_samples': self.spec.consumed_samples + self.spec.global_batch_size,
            'micro_cursor': self.spec.micro_cursor + self.spec.micro_batch_num,
        }


def proxy_cost(loads: np.ndarray) -> tuple:
    """Maximum rank work and sum of synchronous micro-batch maxima."""
    return int(loads.sum(axis=1).max()), int(loads.max(axis=0).sum())


class StepPlanner:
    """Plan real consumption windows without changing source or loss normalization.

    The provider returns native baseline rank indices. Whole micro-batches are
    moved, retaining CE/CP/MTP denominators even when local_batch_size > 1.
    Cross-epoch and short-micro steps use the complete baseline layout.
    """

    def __init__(self, provider, metadata, data_parallel: int, local_batch_size: int,
                 global_batch_size: int, drop_remainder: bool,
                 scheduler=None):
        self.provider = provider
        self.metadata = metadata
        self.degree = data_parallel
        self.batch = local_batch_size
        self.global_batch_size = global_batch_size
        self.scheduler = scheduler
        if (min(self.degree, self.batch, self.global_batch_size) <= 0
                or self.global_batch_size % (self.degree * self.batch)):
            raise ValueError('GBS must be positive and divisible by DP * local_batch_size.')
        rows = provider.rows_per_rank
        self.micros_per_epoch = rows // self.batch if drop_remainder else (rows + self.batch - 1) // self.batch
        if self.micros_per_epoch <= 0:
            raise ValueError('Balance dataset has no consumable micro-batches.')
        self.state = self.initial_state()

    def initial_state(self) -> dict:
        """Return a JSON-compatible rank-independent initial state."""
        return {'step_id': 0, 'consumed_samples': 0, 'micro_cursor': 0}

    def restore(self, state: dict) -> None:
        """Restore a consistent runtime consumption cursor."""
        for key in ('step_id', 'consumed_samples', 'micro_cursor'):
            if not isinstance(state.get(key), int) or state[key] < 0:
                raise ValueError(f'Invalid balance consumption cursor {key}.')
        if state['consumed_samples'] != state['micro_cursor'] * self.degree * self.batch:
            raise ValueError('Balance consumption budget does not match the micro-batch cursor.')
        self.state = {key: state[key] for key in self.initial_state()}

    def next_plan(self) -> StepPlan:
        """Advance a virtual forecast, never trainer/checkpoint committed state."""
        state = self.state
        gbs = self.scheduler.gbs_at_consumed(state['consumed_samples']) if self.scheduler else self.global_batch_size
        units = self.degree * self.batch
        if gbs <= 0 or gbs % units:
            raise ValueError('Scheduler produced an invalid balance GBS.')
        micros = gbs // units
        spec = StepSpec(state['step_id'], state['consumed_samples'], state['micro_cursor'],
                        gbs, self.degree, self.batch, micros)
        ranks = [[] for _ in range(self.degree)]
        epochs = []
        complete = True
        for micro in range(micros):
            epoch, position = divmod(state['micro_cursor'] + micro, self.micros_per_epoch)
            epochs.append(epoch)
            count = min(self.batch, self.provider.rows_per_rank - position * self.batch)
            complete = complete and count == self.batch
            matrix = self.provider.indices(epoch, position * self.batch, count)
            if matrix.shape != (self.degree, count):
                raise ValueError('Baseline provider returned an invalid rank layout.')
            for rank in range(self.degree):
                ranks[rank].append(tuple(int(x) for x in matrix[rank]))
        balanced = False
        if complete and epochs[0] == epochs[-1] and self.degree > 1 and micros > 1:
            flat = np.array(ranks, dtype=np.int64)
            loads = np.asarray(self.metadata(flat.reshape(-1)))
            if loads.shape != (flat.size,) or loads.dtype.kind not in 'iu' or np.any(loads < 0):
                raise ValueError('Metadata provider returned invalid attention loads.')
            # Sum in Python ints to detect overflow rather than silently wrap.
            if sum(int(x) for x in loads) > np.iinfo(np.int64).max:
                raise OverflowError('Step attention load exceeds int64.')
            loads = loads.astype(np.int64, copy=False)
            group_loads = loads.reshape(self.degree, micros, self.batch).sum(axis=2)
            if np.any(group_loads != group_loads.flat[0]):
                # Stable singleton/micro-batch scheduling is O(G log G), with
                # no capacity scan over every group for every source sample.
                order = np.argsort(group_loads.reshape(-1), kind='stable')
                groups = flat.reshape(self.degree * micros, self.batch)
                candidate = np.empty_like(flat)
                candidate_loads = np.empty_like(group_loads)
                for index, original in enumerate(order):
                    micro, rank = divmod(index, self.degree)
                    if micro % 2:
                        rank = self.degree - rank - 1
                    candidate[rank, micro] = groups[original]
                    candidate_loads[rank, micro] = group_loads.reshape(-1)[original]
                old_cost, new_cost = proxy_cost(group_loads), proxy_cost(candidate_loads)
                if all(new <= old for new, old in zip(new_cost, old_cost)) and new_cost != old_cost:
                    ranks = [[tuple(int(x) for x in row) for row in rank] for rank in candidate]
                    balanced = True
        indices = tuple(tuple(rank) for rank in ranks)
        plan = StepPlan(spec, indices, tuple(epochs), balanced)
        self.state = plan.next_state()
        return plan
