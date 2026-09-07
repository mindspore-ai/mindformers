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
"""Tests for PyNative trainer loss extraction."""

from types import SimpleNamespace

import mindspore as ms
import numpy as np
import pytest
from hyper_parallel import PipelineStage

from mindformers.pynative.distributed.pipeline_parallel import ScaledLossPipelineStage
from mindformers.pynative.trainer import Trainer


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_compute_loss_keeps_the_per_token_pair():
    """Per-token mode hands the (loss_sum, token_count) pair through unchanged.

    Dividing here would normalise by this micro-batch's own token count, which is exactly
    what the switch exists to avoid. The single global division happens once per optimizer
    step in ``_finalize_per_token_grads``.
    """
    trainer = Trainer.__new__(Trainer)
    trainer.compute_loss_func = None
    trainer.calculate_per_token_loss = True
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(calculate_per_token_loss=True)
    )

    outputs = (ms.Tensor(12.0, ms.float32), ms.Tensor([3.0], ms.float32))
    numerator, denominator = trainer.compute_loss([lambda **_: outputs], {})

    np.testing.assert_allclose(numerator.asnumpy(), 12.0, rtol=0, atol=0)
    np.testing.assert_allclose(denominator.asnumpy(), [3.0], rtol=0, atol=0)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_compute_pp_loss_accumulates_per_token_outputs(monkeypatch):
    """PP sums the micro-batches' loss and token count, and zeroes the count's seed."""
    trainer = Trainer.__new__(Trainer)
    trainer.compute_loss_func = None
    trainer.calculate_per_token_loss = True
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(calculate_per_token_loss=True)
    )
    trainer.schedule = SimpleNamespace(
        run=lambda **_: [
            (ms.Tensor(12.0, ms.float32), ms.Tensor([3.0], ms.float32)),
            (ms.Tensor(10.0, ms.float32), ms.Tensor([5.0], ms.float32)),
        ]
    )

    # 12+10 = 22 and 3+5 = 8; the ratio is formed once per step, not per micro-batch.
    numerator, denominator = trainer.compute_pp_loss({})
    np.testing.assert_allclose(numerator.asnumpy(), 22.0, rtol=0, atol=0)
    np.testing.assert_allclose(denominator.asnumpy(), [8.0], rtol=0, atol=0)

    stage = object.__new__(ScaledLossPipelineStage)
    stage.loss_scale = 0.5
    stage.calculate_per_token_loss = True
    numerator = ms.Tensor(12.0, ms.float32)
    denominator = ms.Tensor([3.0], ms.float32)
    monkeypatch.setattr(
        PipelineStage,
        "get_last_stage_sens",
        lambda _, outputs: [ms.ops.ones_like(output) for output in outputs],
    )

    numerator_sens, denominator_sens = stage.get_last_stage_sens(
        (numerator, denominator)
    )
    # loss_scale only; no per-micro-batch division by the token count.
    np.testing.assert_allclose(numerator_sens.asnumpy(), 0.5, rtol=1e-6)
    np.testing.assert_allclose(denominator_sens.asnumpy(), 0.0, rtol=0, atol=0)

    stage.calculate_per_token_loss = False
    default_sens = stage.get_last_stage_sens((numerator,))
    np.testing.assert_allclose(default_sens[0].asnumpy(), 0.5, rtol=0, atol=0)
