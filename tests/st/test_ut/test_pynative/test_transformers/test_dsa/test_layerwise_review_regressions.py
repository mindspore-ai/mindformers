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
"""Regressions for single-device, MTP and overlapping early backward."""
from threading import Thread
from types import SimpleNamespace

import pytest
from mindspore import Tensor, nn
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.platform.mindspore.activation_checkpoint.checkpoint_exclude_wrapper import (
    CheckpointExcludeWrapper,
)

from mindformers.pynative.config.config import RecomputeConfig, RecomputeCommConfig
from mindformers.pynative.distributed import activation_checkpoint as ac
from mindformers.pynative.transformers.experimental_attention_variant import indexer
from mindformers.pynative.transformers.multi_token_prediction import MultiTokenPredictionLayer


@pytest.mark.parametrize("sparse", [False, True])
def test_plain_indexer_can_use_early_backward_without_fsdp(sparse):
    """Single-device indexers have gradients but no FSDP scheduler to finalize."""
    with indexer.warmup_indexer_loss_sink(True):
        indexer.publish_indexer_fsdp_unit(nn.Cell(), needs_drain=not sparse)
        assert not indexer.DSA_WARMUP_LOSS_SINK.fsdp_units


def test_mtp_full_recompute_excludes_the_inner_indexer(monkeypatch):
    """Exercise the same outer MTP cell presented to apply_recompute."""
    def init_cell(self, *_args, **_kwargs):
        nn.Cell.__init__(self)

    monkeypatch.setattr(MultiTokenPredictionLayer, "__init__", init_cell)
    layer = MultiTokenPredictionLayer(config=None, submodules=None)
    layer.transformer_layer = nn.Cell()
    attention = nn.Cell()
    attention.layerwise_indexer_backward = True
    attention.sparse_loss = True
    attention.core_attention = nn.Cell()
    attention.core_attention.indexer = nn.Cell()
    attention.core_attention.indexer_loss = nn.Cell()
    layer.transformer_layer.self_attention = attention
    monkeypatch.setattr(ac, "_config_list", {})
    ac.apply_recompute(
        SimpleNamespace(layers=[layer], layer_start=0, layer_end=0),
        RecomputeConfig(mode="full", full_recompute_layer=["0"]),
        RecomputeCommConfig(enable=False, select_module=None),
    )
    assert isinstance(attention.core_attention.indexer, CheckpointExcludeWrapper)
    assert isinstance(attention.core_attention.indexer_loss, CheckpointExcludeWrapper)


def test_early_backward_cannot_restore_another_threads_old_root_state(monkeypatch):
    """A root callback in another thread must not be undone by the flush."""
    class Loss(Tensor):
        def backward(self, seed):
            del seed
            thread = Thread(target=lambda: setattr(HSDPSchedulerV2, "root_bp_state", False))
            thread.start()
            thread.join(timeout=5)
            assert not thread.is_alive()

    monkeypatch.setattr(HSDPSchedulerV2, "root_bp_state", True)
    monkeypatch.setattr(indexer, "sum_indexer_losses", lambda _: Loss([1.0]))
    monkeypatch.setattr(indexer.mint, "ones_like", lambda _: 1.0)
    monkeypatch.setattr(indexer.mint, "reshape", lambda *_: 1.0)
    with indexer.warmup_indexer_loss_sink(True):
        indexer.flush_warmup_indexer_backward([object()])
    assert not HSDPSchedulerV2.root_bp_state


def test_expected_fsdp_indexer_still_rejects_a_missing_scheduler():
    """Ordinary single-device modules must not hide a broken managed unit."""
    module = nn.Cell()
    module._dsa_layerwise_fsdp_managed = True
    with indexer.warmup_indexer_loss_sink(True):
        with pytest.raises(RuntimeError, match="hsdp_scheduler"):
            indexer.publish_indexer_fsdp_unit(module, needs_drain=False)
