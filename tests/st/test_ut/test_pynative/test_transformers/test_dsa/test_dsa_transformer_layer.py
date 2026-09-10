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
    DSA_WARMUP_LOSS_SINK,
    _IndexerLossAutoScaler,
    flush_warmup_indexer_backward,
    sum_indexer_losses,
    warmup_indexer_loss_sink,
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
def test_dsa_warmup_boundary_tolerates_a_layer_with_no_indexer_term():
    """A Shared layer under ``leader`` supervision contributes no Indexer loss.

    ``TransformerBlock`` seeds ``attention_loss`` with the Python float ``0.``, and under
    ``dsa_index_share_loss: leader`` a Shared layer never replaces it -- it owns no indexer
    and publishes no term. The boundary must recognise that there is no gradient to seed
    instead of calling ``ones_like`` on a float, while the Full layer next to it still
    trains normally.
    """
    ms.set_device("CPU")

    def network(x, trunk0, trunk1, indexer1):
        # Layer 0 is Shared: no indexer, so the block's float default reaches the boundary.
        output0 = x * trunk0
        hidden0 = _DSAWarmupLayerBoundary.apply(
            ops.stop_gradient(output0), x, 0.0, False
        )

        # Layer 1 is Full and keeps its own Indexer loss.
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
            Parameter(Tensor([11.0], ms.float32), name="indexer1"),
        )
        grads = ops.grad(network, grad_position=(0, 1, 2, 3))(*inputs)
    finally:
        _IndexerLossAutoScaler.set_loss_scale(previous_scale)

    # Trunks stay frozen; only the Full layer's indexer gets a gradient, and it is the same
    # value the all-layers-supervised case gives it (hidden0 = 2 * 3 = 6, scaled by 0.5).
    expected = ([0.0], [0.0], [0.0], [3.0])
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_warmup_loss_sink_is_published_and_restored():
    """The sink is visible inside the block and gone again afterwards, even on a raise."""
    assert DSA_WARMUP_LOSS_SINK.sink is None

    with warmup_indexer_loss_sink(True) as sink:
        assert sink is not None
        assert DSA_WARMUP_LOSS_SINK.sink is sink
    assert DSA_WARMUP_LOSS_SINK.sink is None

    # Inactive publishes None, i.e. the layers keep the step-end boundary path.
    with warmup_indexer_loss_sink(False) as sink:
        assert sink is None
        assert DSA_WARMUP_LOSS_SINK.sink is None

    # A raising forward must not leave a stale list behind for the next call.
    with pytest.raises(ValueError):
        with warmup_indexer_loss_sink(True):
            raise ValueError("forward blew up")
    assert DSA_WARMUP_LOSS_SINK.sink is None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_warmup_loss_sink_nests_without_leaking():
    """A nested block (MTP inside a decoder forward) restores the outer sink, not None."""
    with warmup_indexer_loss_sink(True) as outer:
        with warmup_indexer_loss_sink(True) as inner:
            assert inner is not outer
            assert DSA_WARMUP_LOSS_SINK.sink is inner
        assert DSA_WARMUP_LOSS_SINK.sink is outer
    assert DSA_WARMUP_LOSS_SINK.sink is None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_flush_warmup_indexer_backward_sums_the_group_once():
    """A group's terms are summed into one backward, and the sink is emptied.

    Uses a stand-in for the loss term because ``Tensor.backward`` is provided by the
    device-side pynative autograd and is absent in a CPU-only unit test.  What matters here
    is the contract IndexShare depends on -- one traversal per group, not one per layer.
    """
    backwards = []

    class _Term:
        """Minimal stand-in: adds like a loss and records its backward calls."""

        def __init__(self, value):
            self.value = value

        def __add__(self, other):
            return _Term(self.value + other.value)

        def backward(self, seed):
            backwards.append((self.value, seed))

    sink = [_Term(3.0), _Term(5.0), _Term(7.0)]
    total = sum_indexer_losses(sink)
    assert total.value == 15.0, "every term of the group must enter one traversal"

    # An empty sink is a no-op: a Shared-only group boundary must not fire a backward.
    flush_warmup_indexer_backward([])
    assert not backwards

    # A group whose every layer contributed the block's float default is the same no-op.
    # This is the per-layer-backward twin of the boundary's ``has_indexer_loss`` check: under
    # ``dsa_index_share_loss: leader`` a Shared layer publishes no term, and a float carries
    # no graph to start a backward from.
    float_only = [0.0, 0.0]
    flush_warmup_indexer_backward(float_only)
    assert not backwards, "a group of graph-less zeros must not fire a backward"
    assert not float_only, "the sink is emptied either way, or the terms leak into the next group"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_warmup_indexer_recompute_honours_full_recompute_layer():
    """A DSA warm-up layer outside ``full_recompute_layer`` keeps its indexer unwrapped."""
    def _dsa_layer_with_indexer():
        layer = nn.Cell()
        layer.disable_activation_recompute = True
        layer.self_attention = nn.Cell()
        layer.self_attention.core_attention = nn.Cell()
        layer.self_attention.core_attention.indexer = nn.Cell()
        return layer

    selected, skipped = _dsa_layer_with_indexer(), _dsa_layer_with_indexer()
    model = SimpleNamespace(layers=[selected, skipped], layer_start=0, layer_end=1)
    # Only layer 0 is selected; layer 1 must stay untouched.
    apply_recompute(model,
                    RecomputeConfig(mode="full", full_recompute_layer=["0"]),
                    RecomputeCommConfig(enable=False, select_module=None))

    assert isinstance(selected.self_attention.core_attention.indexer, CheckpointWrapper)
    assert not isinstance(skipped.self_attention.core_attention.indexer, CheckpointWrapper)
