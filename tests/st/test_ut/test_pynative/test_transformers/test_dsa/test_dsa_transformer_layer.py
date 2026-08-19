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
"""Tests for the DSA-specific Transformer layer backward boundary."""
from types import SimpleNamespace

import numpy as np
import pytest

import mindspore as ms
from mindspore import Parameter, Tensor, nn, mint, ops
from hyper_parallel.platform.mindspore.activation_checkpoint import CheckpointWrapper

from mindformers.pynative.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from mindformers.pynative.config.config import RecomputeCommConfig, RecomputeConfig
from mindformers.pynative.distributed import activation_checkpoint
from mindformers.pynative.distributed.activation_checkpoint import apply_recompute
from mindformers.pynative.transformers.experimental_attention_variant.dsa_transformer_layer import (
    DSAHyperConnectionTransformerLayer,
    DSATransformerLayer,
    _DSAWarmupLayerBoundary,
)
from mindformers.pynative.transformers.experimental_attention_variant.indexer import (
    _IndexerLossAutoScaler,
)
from mindformers.pynative.transformers.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
)


@pytest.fixture(autouse=True)
def _reset_recompute_whitelist():
    """Keep the activation-checkpoint whitelist isolated between tests."""
    activation_checkpoint._config_list = {}
    yield
    activation_checkpoint._config_list = {}


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_dsa_selects_dedicated_transformer_layer_classes():
    """Only DSA selects the dedicated plain and mHC Transformer layer classes."""
    dsa_spec = get_gpt_layer_local_spec(
        multi_latent_attention=True,
        attention_variant="dsa",
    )
    dsa_hc_spec = get_gpt_layer_local_spec(
        multi_latent_attention=True,
        attention_variant="dsa",
        enable_hyper_connections=True,
    )
    mla_spec = get_gpt_layer_local_spec(
        multi_latent_attention=True,
        attention_variant="mla",
    )

    assert dsa_spec.module is DSATransformerLayer
    assert dsa_hc_spec.module is DSAHyperConnectionTransformerLayer
    assert mla_spec.module is TransformerLayer
    assert issubclass(DSATransformerLayer, TransformerLayer)
    assert issubclass(DSAHyperConnectionTransformerLayer, HyperConnectionTransformerLayer)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_dsa_warmup_boundary_skips_trunk_and_trains_all_local_indexers():
    """Two chained layers cut trunk grads while retaining both local Indexer roots."""
    ms.set_device("CPU")

    def network(x, trunk0, trunk1, indexer0, indexer1):
        output0 = x * trunk0
        loss0 = mint.sum(ops.stop_gradient(x) * indexer0)
        hidden0 = _DSAWarmupLayerBoundary.apply(
            ops.stop_gradient(output0), x, loss0, False
        )

        output1 = hidden0 * trunk1
        loss1 = mint.sum(ops.stop_gradient(hidden0) * indexer1)
        hidden1 = _DSAWarmupLayerBoundary.apply(
            ops.stop_gradient(output1), hidden0, loss1, True
        )
        return mint.sum(hidden1)

    previous_scale = _IndexerLossAutoScaler.main_loss_backward_scale
    try:
        _IndexerLossAutoScaler.set_loss_scale(Tensor(0.5, ms.float32))
        inputs = (
            Tensor([2.0], ms.float32),
            Parameter(Tensor([3.0], ms.float32), name="trunk0"),
            Parameter(Tensor([5.0], ms.float32), name="trunk1"),
            Parameter(Tensor([7.0], ms.float32), name="indexer0"),
            Parameter(Tensor([11.0], ms.float32), name="indexer1"),
        )
        grads = ops.grad(network, grad_position=(0, 1, 2, 3, 4))(*inputs)
    finally:
        _IndexerLossAutoScaler.set_loss_scale(previous_scale)

    expected = ([0.0], [0.0], [0.0], [1.0], [3.0])
    for actual, target in zip(grads, expected):
        np.testing.assert_array_equal(actual.asnumpy(), target)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_dsa_warmup_layer_and_nested_mtp_are_not_checkpoint_wrapped():
    """Full recompute skips direct and MTP-nested DSA1 layer boundaries."""
    dsa_layer = nn.Cell()
    dsa_layer.disable_activation_recompute = True
    regular_layer = nn.Cell()
    model = SimpleNamespace(
        layers=[dsa_layer, regular_layer],
        layer_start=0,
        layer_end=1,
    )
    recompute = RecomputeConfig(mode="full", full_recompute_layer=["0-1"])
    recompute_comm = RecomputeCommConfig(enable=False, select_module=None)

    apply_recompute(model, recompute, recompute_comm)

    assert not isinstance(model.layers[0], CheckpointWrapper)
    assert isinstance(model.layers[1], CheckpointWrapper)

    nested_dsa_layer = nn.Cell()
    nested_dsa_layer.disable_activation_recompute = True
    mtp_layer = nn.Cell()
    mtp_layer.transformer_layer = nested_dsa_layer
    mtp_model = SimpleNamespace(layers=[mtp_layer], layer_start=0, layer_end=0)
    recompute = RecomputeConfig(mode="full", full_recompute_layer=["0"])

    apply_recompute(mtp_model, recompute, recompute_comm)

    assert not isinstance(mtp_model.layers[0], CheckpointWrapper)
