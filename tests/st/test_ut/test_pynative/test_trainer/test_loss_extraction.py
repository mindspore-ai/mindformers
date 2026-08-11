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
def test_compute_loss_normalizes_explicit_per_token_output():
    """Per-token mode reports loss_sum divided by token_count."""
    trainer = Trainer.__new__(Trainer)
    trainer.compute_loss_func = None
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(calculate_per_token_loss=True)
    )

    outputs = (ms.Tensor(12.0, ms.float32), ms.Tensor([3.0], ms.float32))
    loss = trainer.compute_loss([lambda **_: outputs], {})

    np.testing.assert_allclose(loss.asnumpy(), 4.0, rtol=0, atol=0)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_compute_pp_loss_normalizes_per_token_outputs(monkeypatch):
    """PP reports normalized micro-batch losses and scales the loss-sum seed."""
    trainer = Trainer.__new__(Trainer)
    trainer.compute_loss_func = None
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(calculate_per_token_loss=True)
    )
    trainer.schedule = SimpleNamespace(
        run=lambda **_: [
            (ms.Tensor(12.0, ms.float32), ms.Tensor([3.0], ms.float32)),
            (ms.Tensor(10.0, ms.float32), ms.Tensor([5.0], ms.float32)),
        ]
    )

    loss = trainer.compute_pp_loss({})
    np.testing.assert_allclose(loss.asnumpy(), 3.0, rtol=0, atol=0)

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
    np.testing.assert_allclose(numerator_sens.asnumpy(), 1.0 / 6.0, rtol=1e-6)
    np.testing.assert_allclose(denominator_sens.asnumpy(), 0.0, rtol=0, atol=0)

    stage.calculate_per_token_loss = False
    default_sens = stage.get_last_stage_sens((numerator,))
    np.testing.assert_allclose(default_sens[0].asnumpy(), 0.5, rtol=0, atol=0)
