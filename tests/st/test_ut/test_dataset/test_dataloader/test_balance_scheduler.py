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
"""test balance scheduler"""

import types

import numpy as np
import pytest

from mindformers.dataset.dataloader.balance_scheduler import (
    BALANCE_SUPPORTED_LOADERS,
    BalanceConfig,
    BalanceScheduler,
    BalancedMapDataset,
    DatasetBalancer,
    assert_balance_supported,
)
from mindformers.dataset.dataloader.hf_dataloader import HFDataset


def _seq_len_fn(values):
    """Build a callback returning a fixed actual_seq_len per index."""
    def fn(idx):
        return np.array([values[idx % len(values)]], dtype=np.int64)
    return fn


def _cfg(gbs, dp, mbn):
    """Build a BalanceConfig with model_parallel fixed to 1."""
    return BalanceConfig(gbs, mbn, dp, 1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balance_supported_whitelist_and_guard():
    """
    Feature: BALANCE_SUPPORTED_LOADERS and assert_balance_supported
    Description: whitelist contents and the raise / no-op truth table
    Expectation: exact three loaders; raises for unsupported+True, no-op otherwise
    """
    assert set(BALANCE_SUPPORTED_LOADERS) == {
        'OrderedIndexDataLoader', 'HFDataLoader', 'BlendedMegatronDatasetDataLoader'}
    with pytest.raises(ValueError, match="MindDataset"):
        assert_balance_supported('MindDataset', True)
    with pytest.raises(ValueError, match="not supported"):
        assert_balance_supported('SomeOtherLoader', True)
    for loader in BALANCE_SUPPORTED_LOADERS:
        assert_balance_supported(loader, True)
    assert_balance_supported('MindDataset', False)
    assert_balance_supported('UnknownLoader', False)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_map_index_boundary_partial_window_clamp():
    """
    Feature: BalanceScheduler.map_index
    Description: tail window smaller than global_batch_size
    Expectation: every index is clamped into [0, total_length - 1]
    """
    sched = BalanceScheduler(_cfg(8, 1, 1), total_length=5,
                             get_actual_seq_len=_seq_len_fn([10, 1, 10, 1, 1]))
    out = [sched.map_index(i) for i in range(5)]
    assert all(0 <= x <= 4 for x in out)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_map_index_snake_balances_per_rank_total():
    """
    Feature: BalanceScheduler.map_index
    Description: snake assignment over a full window
    Expectation: bijection holds and per-rank total load spread satisfies snake <= col <= row
    """
    def _rank_total_spread(bi, load_vals, dp, mbn, mbs, mode):
        totals = [0] * dp
        for mb in range(mbn):
            for r in range(dp):
                if mode == 'row':
                    j = r * mbn + mb
                elif mode == 'col':
                    j = mb * dp + r
                else:  # snake
                    j = mb * dp + (r if mb % 2 == 0 else dp - 1 - r)
                for s in range(mbs):
                    totals[r] += int(load_vals[int(bi[j * mbs + s])]) ** 2
        return max(totals) - min(totals)

    # (load_vals, gbs, dp, mbn); gbs == dp*mbn*mbs. Unbalanced so greedy cannot equalize.
    cases = [
        ([1, 2, 3, 4, 5, 6, 7, 8], 8, 4, 2),                    # mbs=1
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 12, 2, 3),    # mbs=2
        ([9, 8, 7, 6, 5, 4, 3, 2, 1], 8, 2, 4),                  # mbs=1
        ([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 12, 2, 3),        # mbs=2
    ]
    for load_vals, gbs, dp, mbn in cases:
        mbs = gbs // (dp * mbn)
        sched = BalanceScheduler(_cfg(gbs, dp, mbn), total_length=gbs,
                                 get_actual_seq_len=_seq_len_fn(load_vals))
        out = [int(sched.map_index(i)) for i in range(gbs)]
        assert sorted(out) == list(range(gbs)), f"not a bijection: {out}"
        sched.map_index(0)  # populate window cache
        row = _rank_total_spread(sched.balanced_indices, load_vals, dp, mbn, mbs, 'row')
        col = _rank_total_spread(sched.balanced_indices, load_vals, dp, mbn, mbs, 'col')
        snake = _rank_total_spread(sched.balanced_indices, load_vals, dp, mbn, mbs, 'snake')
        assert snake <= col <= row, \
            f"snake<=col<=row failed: snake={snake} col={col} row={row} (gbs={gbs},dp={dp},mbn={mbn})"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_map_index_mbs_gt_1_balances_per_microbatch_load():
    """
    Feature: BalanceScheduler.map_index
    Description: per-micro-batch load with micro_batch_size > 1
    Expectation: balanced load spread is no larger than the identity baseline
    """
    gbs, dp, mbn = 8, 2, 2
    mbs = gbs // (dp * mbn)
    assert mbs == 2
    # Load values: heavies first, lights last.
    raw = [1000, 1000, 1000, 1000, 1, 1, 1, 1]
    sched = BalanceScheduler(_cfg(gbs, dp, mbn), total_length=gbs,
                             get_actual_seq_len=lambda idx: np.array([raw[idx % len(raw)]], dtype=np.int64))

    def _per_microbatch_loads(map_fn):
        sums = {}
        for i in range(gbs):
            rank = i % dp
            mb = (i // dp) // mbs
            s = map_fn(i)
            sums[(rank, mb)] = sums.get((rank, mb), 0) + int(raw[s]) ** 2
        return list(sums.values())

    balanced = _per_microbatch_loads(sched.map_index)
    identity = _per_microbatch_loads(lambda i: i)
    assert max(balanced) - min(balanced) == 0, f"balanced not flat: {balanced}"
    assert max(identity) - min(identity) > 0
    assert max(balanced) - min(balanced) < max(identity) - min(identity)


class _FakeMapDataset:
    """A map-style dataset stub returning (tokens, labels, loss_mask, position_ids, actual_seq_len).

    When seq_len is an int the last column is a 3-D attention mask (create_compressed_eod_mask=False);
    when it is a list the last column is a 1-D actual_seq_len.
    """

    def __init__(self, num, seq_lens):
        self._num = num
        self._seq_lens = seq_lens
        self.column_names = ['input_ids', 'labels', 'loss_mask', 'position_ids', 'actual_seq_len']

    def __len__(self):
        return self._num

    def __getitem__(self, idx):
        if idx is None:
            idx = 0
        n = self._seq_lens if isinstance(self._seq_lens, int) else self._seq_lens[idx % len(self._seq_lens)]
        tokens = np.full(n, 1, dtype=np.int32)
        last = np.zeros((1, n, n), dtype=np.int32) if isinstance(self._seq_lens, int) \
            else np.array([n], dtype=np.int32)
        return (tokens, tokens, tokens, tokens, last)


class _FakeLoaderWithSource:
    """A stub mimicking a GeneratorDataset-based dataloader with a swappable source."""

    def __init__(self, source):
        self.source = source


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_balanced_map_dataset_branches():
    """
    Feature: BalancedMapDataset
    Description: remap+delegate, None passthrough and 3-D mask fallback
    Expectation: index remapped to a valid sample, None passed through, 3-D mask falls back to token length
    """
    gbs, dp, mbn = 4, 2, 2
    # remap + delegate + column_names proxy
    base = _FakeMapDataset(gbs, [1024, 16, 1024, 16])
    wrapped = BalancedMapDataset(base, _cfg(gbs, dp, mbn))
    assert len(wrapped) == gbs
    assert wrapped.column_names == base.column_names
    sample = wrapped[0]
    assert len(sample) == 5
    assert int(sample[-1][0]) in (16, 1024)
    # None idx passthrough (padding sequence)
    assert len(wrapped[None]) == 5
    # 3-D attention mask falls back to token length
    mask3d = BalancedMapDataset(_FakeMapDataset(4, 8), _cfg(4, 2, 2))
    assert mask3d._get_actual_seq_len(0).tolist() == [8]


def _make_hf_config():
    """Build a minimal HFDataset config stub."""
    cfg = types.SimpleNamespace()
    cfg.create_compressed_eod_mask = False
    cfg.compressed_eod_mask_length = 128
    cfg.create_attention_mask = False
    cfg.process = None
    return cfg


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_hf_dataset_no_balance_logic():
    """
    Feature: HFDataset
    Description: HFDataset carries no balance logic
    Expectation: no scheduler attribute and no remapping
    """
    samples = [{'input_ids': np.array([i] * 4, dtype=np.int32),
                'labels': np.array([i] * 4, dtype=np.int32)} for i in range(4)]
    ds = HFDataset(_make_hf_config(), samples)
    assert not hasattr(ds, 'scheduler')
    assert tuple(ds[2]) == (samples[2]['input_ids'], samples[2]['labels'])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dataset_balancer_wraps_source():
    """
    Feature: DatasetBalancer
    Description: apply wraps the loader's source with a BalancedMapDataset
    Expectation: source is swapped to a BalancedMapDataset that is a bijection over indices
    """
    base = _FakeMapDataset(4, [1024, 16, 1024, 16])
    loader = _FakeLoaderWithSource(base)
    parallelism = types.SimpleNamespace(
        data_parallel=2, tensor_parallel=1, context_parallel=2,
        pipeline_parallel_microbatch_size=2)
    out = DatasetBalancer(loader, parallelism=parallelism,
                          global_batch_size=4).apply()
    assert out is loader
    assert isinstance(out.source, BalancedMapDataset)
    # model_parallel = tensor_parallel * context_parallel
    assert out.source.scheduler.config.model_parallel == 2
    mapped = [int(out.source.scheduler.map_index(i)) for i in range(4)]
    assert sorted(mapped) == [0, 1, 2, 3]
