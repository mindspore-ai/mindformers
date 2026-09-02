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
"""Load-balance index scheduler and wrapper for map-style dataloaders."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mindformers.tools.logger import logger


# Supported map-style dataloader types. Add a type name to extend.
BALANCE_SUPPORTED_LOADERS = (
    'OrderedIndexDataLoader',
    'HFDataLoader',
    'BlendedMegatronDatasetDataLoader',
)


@dataclass
class BalanceConfig:
    """Balance scheduler configuration.

    Args:
        global_batch_size (int): Samples per global step (rebalance window).
        micro_batch_num (int): Micro-batches per global step, per DP rank.
        data_parallel (int): Data-parallel degree.
        model_parallel (int): Model-parallel degree, ``tensor_parallel * context_parallel``
            (reserved; currently unused).
    """

    global_batch_size: int
    micro_batch_num: int
    data_parallel: int
    model_parallel: int


def assert_balance_supported(dataloader_type: str, balance_enabled: bool) -> None:
    """Raise if balancing is enabled for an unsupported dataloader type.

    Args:
        dataloader_type (str): Dataloader type name.
        balance_enabled (bool): Whether load balancing is enabled.

    Raises:
        ValueError: If balancing is enabled and ``dataloader_type`` is unsupported.
    """
    if balance_enabled and dataloader_type not in BALANCE_SUPPORTED_LOADERS:
        raise ValueError(
            f"balance_enabled is True but dataloader type '{dataloader_type}' is not supported. "
            f"Supported types: {list(BALANCE_SUPPORTED_LOADERS)}. Set balance_enabled to False "
            f"or switch to a supported dataloader."
        )


class BalanceScheduler:
    """Reorder sample indices to balance attention load across DP ranks.

    Args:
        config (BalanceConfig): Balance configuration.
        total_length (int): Total number of samples.
        get_actual_seq_len (Callable[[int], np.ndarray]): Callback returning the
            compressed EOD positions of the sample at a given index.
    """

    def __init__(self, config: BalanceConfig, total_length: int,
                 get_actual_seq_len: Callable[[int], np.ndarray]):
        self.config = config
        self.total_length = total_length
        self.get_actual_seq_len = get_actual_seq_len
        # Per-window reorder cache.
        self.balanced_indices = None
        self.balanced_indices_start_idx = None
        # Per-micro-batch size.
        self.micro_batch_size = (config.global_batch_size
                                 // (config.data_parallel * config.micro_batch_num))
        logger.info("BalanceScheduler enabled: global_batch_size=%d, data_parallel=%d, "
                    "model_parallel=%d, micro_batch_num=%d, micro_batch_size=%d.",
                    config.global_batch_size, config.data_parallel, config.model_parallel,
                    config.micro_batch_num, self.micro_batch_size)

    def map_index(self, index: int) -> int:
        """Map a global index to its load-balanced peer index.

        Args:
            index (int): Original global sample index.

        Returns:
            int: Rebalanced global index, clamped to ``[0, total_length - 1]``.
        """
        global_batch_size = int(self.config.global_batch_size)
        start_idx, local_idx = divmod(index, global_batch_size)
        start_idx *= global_batch_size

        # Real sample count in this window.
        window_size = min(global_batch_size, self.total_length - start_idx)

        # Re-encode local_idx into SNAKE-ordered group position. Tail windows skip this.
        is_tail = window_size < global_batch_size
        if not is_tail:
            rank = local_idx % self.config.data_parallel
            in_shard_pos = local_idx // self.config.data_parallel
            mb = in_shard_pos // self.micro_batch_size
            slot = in_shard_pos % self.micro_batch_size
            # Reverse rank order on odd steps.
            block_pos = rank if (mb % 2 == 0) else (self.config.data_parallel - 1 - rank)
            local_idx = (mb * self.config.data_parallel + block_pos) * self.micro_batch_size + slot

        if self.balanced_indices is not None and self.balanced_indices_start_idx == start_idx:
            return self._clamp(self.balanced_indices[local_idx] + start_idx)

        actual_seq_lens = [
            self.get_actual_seq_len(start_idx + i) for i in range(window_size)
        ]

        self.balanced_indices = _balance_attention_load(
            actual_seq_lens,
            self.config.data_parallel,
            self.config.micro_batch_num
        )
        self.balanced_indices_start_idx = start_idx
        return self._clamp(self.balanced_indices[local_idx] + start_idx)

    def _clamp(self, index: int) -> int:
        """Clamp an index into the valid range ``[0, total_length - 1]``."""
        return min(index, self.total_length - 1)


class BalancedMapDataset:
    """Generic wrapper that load-balances a map-style dataset by index reordering.

    Args:
        dataset: The underlying map-style dataset.
        balance_config (BalanceConfig): Balance configuration.
        seq_len_index (int): Index of the ``actual_seq_len`` column. Default ``-1``.
    """

    def __init__(self, dataset, balance_config: BalanceConfig, seq_len_index: int = -1):
        self._dataset = dataset
        self._seq_len_index = seq_len_index
        self.scheduler = BalanceScheduler(balance_config, len(dataset), self._get_actual_seq_len)

    def __len__(self) -> int:
        """Return the number of samples, delegating to the wrapped dataset."""
        return len(self._dataset)

    def __getitem__(self, idx):
        """Return the sample at a (possibly remapped) index. ``idx is None`` is passed through."""
        if idx is not None:
            idx = int(self.scheduler.map_index(int(idx)))
        return self._dataset[idx]

    @property
    def column_names(self):
        """Proxy the wrapped dataset's column names."""
        return getattr(self._dataset, 'column_names', None)

    def _get_actual_seq_len(self, idx: int) -> np.ndarray:
        """Return a sample's ``actual_seq_len`` for the load metric. Falls back to token length if not 1-D."""
        sample = self._dataset[idx]
        col = sample[self._seq_len_index]
        if np.ndim(col) == 1:
            return np.array(col, dtype=np.int64)
        return np.array([len(sample[0])], dtype=np.int64)


class DatasetBalancer:
    """Entry point that load-balances a dataset.

    Balancing is gated by the caller: only construct this when the dataloader
    type is in :data:`BALANCE_SUPPORTED_LOADERS`, ``balance_enabled`` is True,
    and ``create_compressed_eod_mask`` is True.

    Args:
        dataset: A dispatched dataloader whose ``source`` is a map-style dataset.
        parallelism: Parallelism config exposing ``data_parallel``,
            ``tensor_parallel``, ``context_parallel`` and
            ``pipeline_parallel_microbatch_size``.
        global_batch_size (int): Samples per global step (rebalance window).
    """

    def __init__(self, dataset, *, parallelism, global_batch_size):
        self._dataset = dataset
        self._parallelism = parallelism
        self._global_batch_size = global_batch_size

    def apply(self):
        """Wrap the dataset's source with a :class:`BalancedMapDataset` and return it."""
        # Model-parallel degree spans both tensor and context parallelism: together
        # they carve the non-data-parallel ranks within a DP group.
        model_parallel = self._parallelism.tensor_parallel * self._parallelism.context_parallel
        cfg = BalanceConfig(
            global_batch_size=self._global_batch_size,
            micro_batch_num=self._parallelism.pipeline_parallel_microbatch_size,
            data_parallel=self._parallelism.data_parallel,
            model_parallel=model_parallel,
        )
        self._dataset.source = BalancedMapDataset(self._dataset.source, cfg)
        return self._dataset


def _balance_attention_load(actual_seq_lens, data_parallel, micro_batch_num):
    """Balance samples by approximate attention load across DP micro-batches.

    Load per sequence is ``sum((len_i - len_{i-1})^2)`` over its compressed EOD positions.

    Args:
        actual_seq_lens: Compressed EOD positions per sample.
        data_parallel: Data-parallel degree.
        micro_batch_num: Micro-batches per global step, per DP rank.

    Returns:
        Reordered sample indices (int64), length ``len(actual_seq_lens)``.
    """
    attn_load = []
    for seq in actual_seq_lens:
        # Prepend 0.
        seq_with_zero = [0] + seq.tolist()
        cur_load = sum((seq_with_zero[i] - seq_with_zero[i - 1]) ** 2
                       for i in range(1, len(seq_with_zero)))
        attn_load.append(cur_load)
    attn_load = np.array(attn_load)

    balanced_group, group_sums = _greedy_balanced_group(
        attn_load,
        data_parallel * micro_batch_num
    )

    # Reorder groups by ascending group load.
    indices = np.argsort(np.array(group_sums))
    balanced_group = [idx for group_idx in indices for idx in balanced_group[group_idx]]
    return np.array(balanced_group, dtype=np.int64)


def _greedy_balanced_group(attn_load, k=None):
    """Greedily partition attention loads into ``k`` balanced groups.

    Args:
        attn_load: Per-sample attention load.
        k: Number of groups (must be positive).

    Returns:
        (groups, group_sums): groups is a list of k index lists; group_sums is the
        load sum per group.
    """
    if k is None or k <= 0:
        raise ValueError(f"k must be a positive integer, but got {k}.")
    max_group_size = (len(attn_load) + k - 1) // k

    pairs = [(int(attn_load[i]), i) for i in range(len(attn_load))]
    pairs.sort(key=lambda x: -x[0])

    groups = [[] for _ in range(k)]
    group_sums = np.zeros(k, dtype=np.int64)
    group_sizes = np.zeros(k, dtype=int)

    for value, idx in pairs:
        best_group = None
        best_sum = None

        for g in range(k):
            if group_sizes[g] < max_group_size:
                if best_group is None or group_sums[g] < best_sum:
                    best_group = g
                    best_sum = group_sums[g]

        groups[best_group].append(idx)
        group_sums[best_group] += value
        group_sizes[best_group] += 1

    return groups, group_sums
