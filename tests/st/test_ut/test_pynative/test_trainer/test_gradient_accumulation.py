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
"""Tests for PyNative trainer gradient-accumulation communication scheduling."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from hyper_parallel.core.fully_shard.api import HSDPModule

from mindformers.pynative.trainer.trainer import Trainer


class _RecordingHSDPModel(HSDPModule):
    """Minimal HSDP model double recording replicate all-reduce toggles."""

    def __init__(self):
        self.all_reduce_calls = []

    def set_requires_all_reduce(self, requires_all_reduce: bool, *, recurse: bool = True) -> None:
        del recurse
        self.all_reduce_calls.append(requires_all_reduce)


def test_set_requires_all_reduce_only_updates_hsdp_models():
    """The helper should update each HSDP model and ignore plain objects."""
    trainer = Trainer.__new__(Trainer)
    first_model = _RecordingHSDPModel()
    second_model = _RecordingHSDPModel()
    trainer.model = [first_model, object(), second_model]

    trainer._set_requires_all_reduce(False)
    trainer._set_requires_all_reduce(True)

    assert first_model.all_reduce_calls == [False, True]
    assert second_model.all_reduce_calls == [False, True]


def test_training_step_enables_all_reduce_only_on_last_micro_step():
    """Every optimizer step should issue one replicate all-reduce schedule."""
    trainer = Trainer.__new__(Trainer)
    trainer.num_accumulation_steps = 4
    trainer.model = []
    trainer.enable_parallel = False
    trainer.monitor = MagicMock()
    trainer.monitor.should_record.return_value = False
    trainer._next_batch = MagicMock(side_effect=[object(), object(), object(), object()])
    trainer._forward_backward = MagicMock(side_effect=[1.0, 2.0, 3.0, 4.0])
    trainer._optimizer_update = MagicMock(return_value=0.0)
    trainer._set_requires_all_reduce = MagicMock()

    loss, grad_norm = trainer.training_step(step=0)

    assert trainer._set_requires_all_reduce.call_args_list == [
        call(False), call(False), call(False), call(True)
    ]
    assert trainer._optimizer_update.call_count == 1
    assert loss == 10.0
    assert grad_norm == 0.0


@pytest.mark.parametrize("frozen_trunk", [False, True])
def test_per_token_backward_respects_early_warmup_backward(monkeypatch, frozen_trunk):
    """Frozen warm-up already consumed the indexer graph during compute_loss."""
    from mindformers.pynative.trainer import trainer as trainer_module

    backward_calls = []

    class _Loss:
        def backward(self, seed):
            backward_calls.append(seed)

    numerator, denominator, seed = _Loss(), object(), object()
    trainer = Trainer.__new__(Trainer)
    trainer.compute_loss = MagicMock(return_value=(numerator, denominator))
    trainer.enable_parallel = False
    trainer.config = SimpleNamespace(parallelism=SimpleNamespace())
    trainer.num_accumulation_steps = 2
    trainer.calculate_per_token_loss = True
    trainer._dsa_frozen_trunk_layerwise = frozen_trunk
    monkeypatch.setattr(trainer_module, "get_loss_sense", lambda **kwargs: seed)

    assert trainer._forward_backward(object(), {}) == (numerator, denominator)
    assert backward_calls == ([] if frozen_trunk else [seed])
