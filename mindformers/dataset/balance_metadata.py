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
"""Read-only balance metadata; deliberately independent of loaders and models."""

import hashlib
import json
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np


def checked_indices(indices, size: int) -> np.ndarray:
    """Validate logical indices without accepting negative wrapping or float casts."""
    values = np.asarray(indices)
    if values.ndim != 1:
        raise ValueError('Balance indices must be a one-dimensional integer array.')
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    if values.dtype.kind not in 'iu':
        raise ValueError('Balance indices must be a one-dimensional integer array.')
    if np.any(values < 0) or np.any(values >= size):
        raise IndexError(f'Balance index outside [0, {size}).')
    return values.astype(np.int64, copy=False)


def attention_load(boundaries, length: int = None) -> int:
    """Compute an exact int64 proxy from validated cumulative training boundaries."""
    values = np.asarray(boundaries)
    if values.ndim != 1 or not values.size or values.dtype.kind not in 'iu':
        raise ValueError('Compressed EOD boundaries must be a nonempty integer vector.')
    values = values.astype(np.int64)
    if values[0] < 0 or np.any(values[1:] < values[:-1]):
        raise ValueError('Compressed EOD boundaries must be nonnegative and monotonic.')
    if length is not None and (values[-1] != length or np.any(values > length)):
        raise ValueError('Compressed EOD boundaries must cover the complete training sequence.')
    # Python integer arithmetic avoids overflow before the bound check.
    total = sum(int(x) ** 2 for x in np.diff(np.concatenate(([0], values))))
    if total > np.iinfo(np.int64).max:
        raise OverflowError('Attention load exceeds int64.')
    return total


def eod_boundaries(positions, length: int, capacity: int) -> np.ndarray:
    """Match token EOD conversion, excluding a final EOD and padding with length."""
    values = np.asarray(positions, dtype=np.int64)
    values = values[(values > 0) & (values < length)]
    if len(values) > capacity:
        raise ValueError('Number of EOD positions exceeds compressed boundary capacity.')
    result = np.pad(values, (0, capacity - len(values)), constant_values=length)
    attention_load(result, length)
    return result


def fingerprint(description) -> str:
    """Hash a versioned, JSON-compatible source/transform description."""
    return hashlib.sha256(json.dumps(description, sort_keys=True, default=str).encode()).hexdigest()


def file_versions(prefixes) -> list:
    """Describe immutable indexed-file revisions, including inode and nanosecond timestamps."""
    result = []
    for prefix in prefixes:
        for suffix in ('.bin', '.idx'):
            path = Path(str(prefix) + suffix).resolve()
            stat = path.stat()
            result.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino))
    return result


class DocumentEodIndex:
    """Persist token-only document scans with atomic publication and bounded memory.

    Each document is built once on shared storage. POSIX advisory locks release on
    process death; waiters time out instead of participating in world barriers.
    No labels, masks, position ids or training samples are constructed here.
    """

    def __init__(self, dataset, eod: int, cache_dir: str, source_revision: str,
                 max_documents: int = 256, max_bytes: int = 64 * 1024 * 1024):
        if max_documents < 0 or max_bytes < 0:
            raise ValueError('Document metadata cache limits must be nonnegative.')
        self.dataset = dataset
        self.eod = eod
        self.directory = Path(cache_dir) / source_revision
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cache = OrderedDict()
        self.max_documents = max_documents
        self.max_bytes = max_bytes
        self.cache_bytes = 0
        self.token_reads = 0

    def get(self, index: int) -> np.ndarray:
        """Return read-only cumulative EOD positions for a physical document."""
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        path = self.directory / f'{index}.npz'
        if not path.exists():
            self._build(index, path)
        try:
            with np.load(path, allow_pickle=False) as record:
                positions = record['positions']
                digest = str(record['sha256'].item())
            if hashlib.sha256(positions.tobytes()).hexdigest() != digest:
                raise ValueError('Document EOD checksum mismatch.')
            if positions.ndim != 1 or positions.dtype != np.int64 or np.any(positions <= 0):
                raise ValueError('Invalid document EOD sidecar.')
            if np.any(positions[1:] <= positions[:-1]):
                raise ValueError('Unsorted document EOD sidecar.')
            if positions.size and positions[-1] > self.dataset.sequence_lengths[index]:
                raise ValueError('Document EOD position exceeds document length.')
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f'Invalid balance metadata cache {path}; rebuild this revision.') from exc
        positions.setflags(write=False)
        if self.max_documents and positions.nbytes <= self.max_bytes:
            self.cache[index] = positions
            self.cache_bytes += positions.nbytes
            while len(self.cache) > self.max_documents or self.cache_bytes > self.max_bytes:
                _, removed = self.cache.popitem(last=False)
                self.cache_bytes -= removed.nbytes
        return positions

    def _build(self, index: int, path: Path) -> None:
        """Build and atomically publish the EOD sidecar under a file lock."""
        # fcntl is a POSIX standard-library dependency; importing lazily keeps the
        # pure numeric helpers usable on platforms without advisory file locks.
        import fcntl

        with open(str(path) + '.lock', 'a', encoding='utf-8') as lock:
            deadline = time.monotonic() + 120
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f'Timed out waiting for balance metadata {path}.') from exc
                    time.sleep(0.05)
            if path.exists():
                return
            if hasattr(self.dataset, 'get'):
                # Bound temporary token comparisons even for long physical
                # documents. The index stores positions, never full masks.
                parts = []
                length = int(self.dataset.sequence_lengths[index])
                for offset in range(0, length, 1024 * 1024):
                    tokens = self.dataset.get(index, offset=offset, length=min(1024 * 1024, length - offset))
                    if isinstance(tokens, tuple):
                        tokens = tokens[0]
                    parts.append(np.flatnonzero(np.asarray(tokens) == self.eod).astype(np.int64) + offset + 1)
                positions = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
                del parts
            else:
                tokens = self.dataset[index]
                if isinstance(tokens, tuple):
                    tokens = tokens[0]
                positions = np.flatnonzero(np.asarray(tokens) == self.eod).astype(np.int64) + 1
            self.token_reads += 1
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.directory, suffix='.npz', delete=False) as output:
                    temp_path = output.name
                    np.savez(output, positions=positions,
                             sha256=np.asarray(hashlib.sha256(positions.tobytes()).hexdigest()))
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temp_path, path)
            finally:
                if temp_path is not None and os.path.exists(temp_path):
                    os.unlink(temp_path)
