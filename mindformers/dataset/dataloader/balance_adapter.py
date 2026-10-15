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
"""Bounded iterable bridge: native index order, producer plans, read-only workers."""

import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass

import numpy as np
import mindspore.dataset as ds

from mindformers.dataset.balance_plan import StepPlanner


_INDEX_BLOCK_SIZE = 4096
_MAX_UNCOMMITTED_STEPS = 2


@dataclass(frozen=True)
class BalanceSpec:
    """Resolved static topology; the original scheduler supplies each step's GBS."""

    data_parallel: int
    shard_id: int
    local_batch_size: int
    global_batch_size: int
    drop_remainder: bool
    scheduler: object = None


class IndexSource:
    """Expose only source indices to the framework's native sampler."""

    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return (np.asarray(index, dtype=np.int64),)


class FrameworkOrderProvider:
    """Consume native global order, retaining baseline DP strides and cyclic padding.

    Recreate the index-only iterator for each data epoch, as the Trainer does
    for the unbalanced dataset. Recovery seeks within the target epoch.
    The provider retains one small index block and the prefix for DP padding;
    framework-owned full index arrays may still exist.
    """

    def __init__(self, size: int, degree: int, shuffle: bool, seed: int):
        if size <= 0 or degree <= 0:
            raise ValueError('Balance source and DP degree must be nonempty/positive.')
        self.size, self.degree, self.seed = size, degree, seed
        self.shuffle = True if shuffle is None else shuffle
        self.rows_per_rank = (size + degree - 1) // degree
        self._slots_per_epoch = self.rows_per_rank * degree
        self._iterator = None
        self._dataset = None
        self._epoch = 0
        self._position = 0
        self._buffer = np.empty(0, dtype=np.int64)
        self._buffer_offset = 0
        self._head = np.empty(min(size, self._slots_per_epoch - size), dtype=np.int64)
        self._cancelled = threading.Event()

    def _new_dataset(self):
        sampler = ds.DistributedSampler(1, 0, shuffle=True)
        sampler.seed = self.seed
        return ds.GeneratorDataset(
            IndexSource(self.size), ['sample_index'], sampler=sampler,
            num_parallel_workers=1, python_multiprocessing=False,
        ).batch(_INDEX_BLOCK_SIZE, drop_remainder=False)

    def _start(self, epoch: int, position: int = 0) -> None:
        """Open the framework iterator at the requested epoch and index block."""
        # The native iterator can seek by batch. Keep at most one block to skip
        # in Python, even when resuming deep inside a large source epoch.
        block = min(position, self.size - 1) // _INDEX_BLOCK_SIZE
        if block and self._head.size:
            # DP padding at the end of an epoch repeats its first few indices.
            # A seek past the first block must recover that prefix separately.
            prefix_dataset = self._new_dataset()
            prefix_iterator = prefix_dataset.create_tuple_iterator(num_epochs=1, output_numpy=True)
            try:
                filled = 0
                while filled < self._head.size:
                    values = np.asarray(next(prefix_iterator)[0], dtype=np.int64).reshape(-1)
                    count = min(self._head.size - filled, values.size)
                    self._head[filled:filled + count] = values[:count]
                    filled += count
            finally:
                prefix_iterator.stop()
        self._dataset = self._new_dataset()
        if block:
            self._dataset.set_init_step(block)
        self._iterator = self._dataset.create_tuple_iterator(num_epochs=1, output_numpy=True)
        self._epoch = epoch
        self._position = block * _INDEX_BLOCK_SIZE
        self._buffer = np.empty(0, dtype=np.int64)
        self._buffer_offset = 0

    def _read(self, count: int) -> np.ndarray:
        """Read bounded logical slots, including the original cyclic DP padding."""
        result = np.empty(count, dtype=np.int64)
        written = 0
        while written < count:
            if self._cancelled.is_set():
                raise RuntimeError('Framework index reading was cancelled.')
            if self._position < self.size:
                if self._buffer_offset == self._buffer.size:
                    try:
                        self._buffer = np.asarray(next(self._iterator)[0], dtype=np.int64).reshape(-1)
                    except StopIteration as exc:
                        raise RuntimeError('Framework index stream ended before the source epoch.') from exc
                    self._buffer_offset = 0
                    if not self._buffer.size:
                        raise RuntimeError('Framework index stream returned an empty block.')
                amount = min(count - written, self._buffer.size - self._buffer_offset,
                             self.size - self._position)
                values = self._buffer[self._buffer_offset:self._buffer_offset + amount]
                head_count = min(amount, max(0, self._head.size - self._position))
                if head_count:
                    self._head[self._position:self._position + head_count] = values[:head_count]
                self._buffer_offset += amount
            else:
                amount = min(count - written, self._slots_per_epoch - self._position)
                offsets = np.arange(self._position - self.size,
                                    self._position - self.size + amount) % self._head.size
                values = self._head[offsets]
            if amount <= 0:
                raise RuntimeError('Framework index query crossed an unadvanced epoch.')
            result[written:written + amount] = values
            written += amount
            self._position += amount
        return result

    def _skip(self, count: int) -> None:
        while count:
            amount = min(count, _INDEX_BLOCK_SIZE)
            self._read(amount)
            count -= amount

    def indices(self, epoch: int, start: int, count: int) -> np.ndarray:
        """Return the same rank/slot matrix expected by the existing planner."""
        if epoch < 0 or start < 0 or count < 0 or start + count > self.rows_per_rank:
            raise ValueError('Invalid framework index window.')
        if not self.shuffle:
            return ((np.arange(start, start + count)[None, :] * self.degree
                     + np.arange(self.degree)[:, None]) % self.size).astype(np.int64)
        target = start * self.degree
        if self._iterator is not None and (epoch != self._epoch or target < self._position):
            # Trainer recreates its iterator at every epoch; a backward query
            # likewise replays only the requested epoch from its beginning.
            self.close()
        if self._iterator is None:
            self._start(epoch, target)
        self._skip(target - self._position)
        return self._read(count * self.degree).reshape(count, self.degree).T

    def cancel(self) -> None:
        """Interrupt index replay between framework reads."""
        self._cancelled.set()

    def close(self) -> None:
        """Release the framework iterator; no index-stream state is persisted."""
        try:
            if self._iterator is not None:
                self._iterator.stop()
        finally:
            self._iterator = None
            self._dataset = None
            self._buffer = np.empty(0, dtype=np.int64)
            self._buffer_offset = 0


class BalancePlanSource:
    """An iterable of already-batched rows with bounded immutable step forecasts.

    A dedicated producer is the only planner/metadata caller. Conversion threads
    receive final indices and cannot mutate plans or advance committed state.
    This source intentionally has no __getitem__, so Python Sampler.get_indices
    cannot scan/materialize an epoch. Framework epoch resets drain the same stream.
    """

    def __init__(self, source, column_names: list, spec: BalanceSpec, shuffle: bool, workers: int):
        if not column_names or not 0 <= spec.shard_id < spec.data_parallel:
            raise ValueError('Balance source requires training columns and a valid DP shard.')
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError('Balance num_parallel_workers must be a positive integer.')
        self.source, self.column_names, self.spec = source, list(column_names), spec
        self.workers = workers
        self.seed = ds.config.get_seed()
        source.configure_balance_metadata()
        provider = FrameworkOrderProvider(len(source), spec.data_parallel, shuffle, self.seed)
        self.planner = StepPlanner(provider, source.get_balance_loads, spec.data_parallel,
                                   spec.local_batch_size, spec.global_batch_size,
                                   spec.drop_remainder, spec.scheduler)
        self.committed = self.planner.initial_state()
        self.condition = threading.Condition()
        self.plans = {}
        self.rows = queue.Queue(maxsize=max(2, self.workers))
        self.stop = threading.Event()
        self.thread = None
        self.error = None
        self.active_spec = None

    def __len__(self):
        return self.planner.micros_per_epoch

    def __iter__(self):
        self._start()
        # Framework epochs are fixed-size transport blocks. Real data epochs
        # live in StepSpec/C, including a resume that begins inside a data epoch.
        while not self.stop.is_set():
            try:
                kind, value = self.rows.get(timeout=0.1)
            except queue.Empty:
                if self.error is not None:
                    raise RuntimeError('Balance producer failed.') from self.error
                continue
            if kind == 'epoch_end':
                return
            if kind == 'error':
                raise RuntimeError('Balance producer failed.') from value
            if kind == 'close':
                return
            yield value

    def _start(self) -> None:
        with self.condition:
            if self.stop.is_set():
                raise RuntimeError('Balance source is closed.')
            if self.thread is None:
                self.thread = threading.Thread(target=self._produce, name='balance-plan-producer', daemon=True)
                self.thread.start()

    def _put(self, kind: str, value) -> bool:
        while not self.stop.is_set():
            try:
                self.rows.put((kind, value), timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _collate_micro(self, samples: list) -> tuple:
        """Stack one micro-batch and offset its compressed EOD boundaries."""
        columns = []
        for column, name in enumerate(self.column_names):
            values = [sample[column] for sample in samples]
            if name == 'actual_seq_len':
                offset = 0
                adjusted = []
                for value in values:
                    adjusted.append(np.asarray(value) + offset)
                    offset = int(adjusted[-1][-1])
                values = adjusted
            columns.append(np.stack(values, axis=0))
        return tuple(columns)

    def _wait_for_sample(self, pending):
        """Wait for the next planned read while surfacing later read failures."""
        future = pending[0]
        while not self.stop.is_set():
            done, _ = wait((future,), timeout=0.1)
            for queued in pending:
                if queued.done() and queued.exception() is not None:
                    queued.result()
            if done:
                return future.result()
        return None

    def _read_plan(self, indices: tuple, executor):
        """Read in plan order with at most one pending future per worker."""
        sample_indices = iter(index for micro in indices for index in micro)
        pending = deque()

        def submit_next():
            index = next(sample_indices, None)
            if index is not None and not self.stop.is_set():
                pending.append(executor.submit(self.source.__getitem__, index))

        for _ in range(self.workers):
            submit_next()
        try:
            for micro in indices:
                samples = []
                for _ in micro:
                    if self.stop.is_set():
                        return
                    sample = self._wait_for_sample(pending)
                    if self.stop.is_set():
                        return
                    samples.append(sample)
                    pending.popleft()
                    submit_next()
                yield self._collate_micro(samples)
        finally:
            for future in pending:
                future.cancel()

    def _produce(self) -> None:
        """Forecast bounded steps and publish read rows or the first failure."""
        emitted = 0
        executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='balance-reader')
        try:
            while not self.stop.is_set():
                with self.condition:
                    while (self.planner.state['step_id'] - self.committed['step_id']
                           >= _MAX_UNCOMMITTED_STEPS and not self.stop.is_set()):
                        self.condition.wait(timeout=0.1)
                if self.stop.is_set():
                    return
                plan = self.planner.next_plan()
                with self.condition:
                    self.plans[plan.spec.step_id] = plan
                    self.condition.notify_all()
                for row in self._read_plan(plan.indices[self.spec.shard_id], executor):
                    if not self._put('row', row):
                        return
                    emitted += 1
                    if emitted % len(self) == 0 and not self._put('epoch_end', None):
                        return
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Every producer failure must wake the trainer, regardless of type.
            with self.condition:
                self.error = exc
                self.condition.notify_all()
            self._put('error', exc)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            self.planner.provider.close()

    def begin_step(self, consumed_samples: int, gbs: int, micros: int) -> None:
        """Verify a frozen forecast against the trainer before consuming this step."""
        self._start()
        deadline = time.monotonic() + 120
        with self.condition:
            step = self.committed['step_id']
            while step not in self.plans:
                if self.stop.is_set():
                    raise RuntimeError('Balance source closed during step preparation.')
                if self.error is not None:
                    raise RuntimeError('Balance producer failed before step preparation.') from self.error
                if time.monotonic() >= deadline:
                    raise TimeoutError('Timed out waiting for balance StepPlan.')
                self.condition.wait(timeout=0.1)
            plan = self.plans[step]
            expected = (consumed_samples, gbs, micros)
            actual = (plan.spec.consumed_samples, plan.spec.global_batch_size, plan.spec.micro_batch_num)
            if actual != expected:
                raise ValueError(f'Balance StepSpec mismatch: planned {actual}, trainer {expected}.')
            self.active_spec = plan.spec

    def commit_step(self) -> dict:
        """Commit exactly once after the trainer has completed a data-consumption step."""
        with self.condition:
            step = self.committed['step_id']
            if self.active_spec is None or self.active_spec.step_id != step:
                raise RuntimeError('Balance data step must be prepared before commit.')
            plan = self.plans.pop(step)
            self.committed = plan.next_state()
            self.active_spec = None
            self.condition.notify_all()
            return dict(self.committed)

    def restore(self, consumed_samples: int, global_step: int) -> None:
        """Rebuild the consumption cursor; plans and prefetched rows start empty."""
        if self.thread is not None or self.stop.is_set():
            raise RuntimeError('Balance restore requires an open source that has not started reading.')
        units = self.spec.data_parallel * self.spec.local_batch_size
        if consumed_samples < 0 or consumed_samples % units or global_step < 0:
            raise ValueError('Invalid saved balance consumption boundary.')
        state = self.planner.initial_state()
        state.update(step_id=global_step, consumed_samples=consumed_samples,
                     micro_cursor=consumed_samples // units)
        self.planner.restore(state)
        self.committed = dict(state)

    def close(self) -> None:
        """Release the transient producer, plans, and prefetched rows."""
        self.stop.set()
        self.planner.provider.cancel()
        with self.condition:
            self.condition.notify_all()
        try:
            self.rows.put_nowait(('close', None))
        except queue.Full:
            pass
        if self.thread is not None:
            self.thread.join(timeout=5)
        else:
            self.planner.provider.close()


def build_balanced_generator(source, column_names: list, spec: BalanceSpec,
                             shuffle: bool, workers: int):
    """Construct the shared bridge before GeneratorDataset sampler construction."""
    adapter = BalancePlanSource(source, column_names, spec, shuffle, workers)
    dataset = ds.GeneratorDataset(adapter, column_names=column_names, shuffle=False,
                                  num_parallel_workers=1, python_multiprocessing=False)
    dataset._balance_plan_source = adapter
    return dataset
