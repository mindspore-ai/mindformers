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
"""Tests for the process-wide switch that asks for TND-ordered softmax statistics."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from mindspore import Tensor
from mindspore import dtype as mstype

from mindformers.pynative.layers import tnd_softmax_out as tnd_module
from mindformers.pynative.layers.flash_attention import FlashAttention
from mindformers.pynative.transformers.experimental_attention_variant import dsa as dsa_module
from mindformers.parallel_core.transformer_config import TransformerConfig


@pytest.fixture
def layout_state(monkeypatch):
    """Isolate the process-wide contract without changing the real HP kernel mode."""
    monkeypatch.setattr(tnd_module, '_TND_SOFTMAX_OUT_REQUEST', {'value': None, 'requester': ''})
    monkeypatch.setattr(tnd_module, 'set_tnd_softmax_out', lambda value: None)
    monkeypatch.setattr(tnd_module, 'npu_flash_attention_varlen_v4', object())


def _dense(native, layout='TND', converter=None):
    converter = dsa_module.DSASoftmaxConverter(layout) if converter is None else converter
    return dsa_module.DSADenseFlashAttention(layout, 4, 0.5, 3,
        converter, tnd_softmax_out=native)


def _ordinary():
    config = TransformerConfig(num_layers=1, hidden_size=512, num_attention_heads=4,
                               seq_length=2048, use_eod_attn_mask_compression=True,
                               track_max_attention_logit=True)
    return FlashAttention(config, layer_number=1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("absent", ["set_tnd_softmax_out", "npu_flash_attention_varlen_v4"])
@pytest.mark.usefixtures('layout_state')
def test_request_is_refused_without_the_v4_symbols(monkeypatch, absent):
    """Asking for V4-ordered statistics on a hyper-parallel without V4 fails by name.

    Both symbols are needed and for different reasons -- the switch makes hyper-parallel's
    dispatched operator use V4, and the operator itself is what the attention cells call
    where nothing dispatches (``cp == 1``) -- so either one missing must be refused while the
    model is being built. Falling back to the stock kernel would hand the caller v1's
    NTD-ordered statistics under a ``(T, N, G)`` declaration: shapes match, nothing raises,
    and the per-head reduction silently maxes across every head.
    """
    monkeypatch.setattr(tnd_module, absent, None)
    monkeypatch.setattr(tnd_module, "_TND_SOFTMAX_OUT_REQUEST", {"value": None, "requester": ""})
    with pytest.raises(ValueError, match=rf"not supported by this hyper-parallel build.*{absent}"):
        tnd_module.request_tnd_softmax_out(True, "a test")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_not_requesting_needs_no_v4_symbols():
    """Leaving the switch off must still run on a hyper-parallel without the V4 change."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tnd_module, "set_tnd_softmax_out", None)
        patch.setattr(tnd_module, "npu_flash_attention_varlen_v4", None)
        patch.setattr(tnd_module, "_TND_SOFTMAX_OUT_REQUEST", {"value": None, "requester": ""})
        tnd_module.request_tnd_softmax_out(False, "a test")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.usefixtures('layout_state')
def test_conflicting_requests_name_both_sides():
    """Two cells asking for opposite orders must fail, naming who asked for what.

    The switch is process-wide while the decision is per cell, so a disagreement would
    otherwise leave one consumer reading statistics in the order the other one asked for --
    element counts match either way, so nothing would raise.
    """
    tnd_module.request_tnd_softmax_out(True, "first cell")
    with pytest.raises(ValueError, match=r"first cell.*second cell"):
        tnd_module.request_tnd_softmax_out(False, "second cell")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dense_first', [True, False])
@pytest.mark.usefixtures('layout_state')
def test_mixed_dense_and_fa_conflicting_consumers(dense_first):
    """A dense NTD consumer and tracked TND-native FA fail before silently mixing layouts."""
    first, second = (lambda: _dense(False), _ordinary) if dense_first else (_ordinary, lambda: _dense(False))
    first()
    with pytest.raises(ValueError, match='must agree') as error:
        second()
    assert 'dsa_dense_fa_tnd_softmax' in str(error.value) and 'Muon qk_clip' in str(error.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dense_first', [True, False])
@pytest.mark.usefixtures('layout_state')
def test_mixed_dense_and_fa_agree_on_native_layout(dense_first):
    """Both construction orders use one layout when dense TND-native mode is enabled."""
    factories = (lambda: _dense(True), _ordinary) if dense_first else (_ordinary, lambda: _dense(True))
    cells = [build() for build in factories]
    assert all(cell.tnd_softmax_out for cell in cells)
    assert tnd_module._TND_SOFTMAX_OUT_REQUEST['value'] is True


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.usefixtures('layout_state')
def test_bsnd_dense_does_not_override_tnd_consumers():
    """BSND statistics do not participate in the process-wide TND ordering contract."""
    _ordinary()
    cell = _dense(False, 'BSND')
    assert not cell.tnd_softmax_out
    assert tnd_module._TND_SOFTMAX_OUT_REQUEST['value'] is True


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_native_and_legacy_stats_have_same_canonical_values():
    """Native TND skips document repacking while legacy NTD still reconstructs each document."""
    values = np.arange(15, dtype=np.float32).reshape(5, 3, 1)
    lengths = Tensor([2, 5], mstype.int32)
    converter = dsa_module.DSASoftmaxConverter('TND')
    packed = np.concatenate([values[:2].transpose(1, 0, 2).ravel(), values[2:].transpose(1, 0, 2).ravel()])
    legacy, _ = converter.to_cp_layout(Tensor(packed.reshape(values.shape)),
                                        Tensor(packed.reshape(values.shape)), lengths)
    converter.set_cp_rank(7)  # Native statistics do not depend on the chunk's global offset.
    native, _ = converter.to_cp_layout(Tensor(values), Tensor(values), lengths, tnd_native=True)
    np.testing.assert_array_equal(legacy.asnumpy(), values)
    np.testing.assert_array_equal(native.asnumpy(), values)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dispatched', [False, True])
@pytest.mark.usefixtures('layout_state')
def test_dense_native_uses_local_or_cp_dispatch(monkeypatch, dispatched):
    """CP tensors retain the primitive boundary; plain tensors call V4 directly."""
    class DistributedQuery:
        pass

    values = Tensor(np.arange(20, dtype=np.float32).reshape(5, 4, 1))
    output = Tensor(np.zeros((5, 4, 2), np.float32))
    primitive = Mock(return_value=(values, values, None, output))
    local = Mock(return_value=(output, values, values))
    monkeypatch.setattr(dsa_module, 'DTensor', DistributedQuery)
    monkeypatch.setattr(dsa_module, 'FlashAttentionScore', lambda **kwargs: primitive)
    monkeypatch.setattr(dsa_module, 'local_v4_varlen_flash_attention', local)
    monkeypatch.setattr(dsa_module, 'DSASoftmaxConverter', Mock(
        side_effect=AssertionError('Dispatch-only test must not construct the real converter')))
    conversion = Mock(return_value=(values, values))
    cell = _dense(True, converter=SimpleNamespace(to_cp_layout=conversion))
    query = DistributedQuery() if dispatched else output
    lengths = Tensor([2, 5], mstype.int32)
    result, maximum, _ = cell.construct(query, output, output, actual_seq_qlen=lengths)
    assert result is output
    np.testing.assert_array_equal(maximum.asnumpy(), values.asnumpy())
    assert primitive.called is dispatched and local.called is not dispatched
    conversion.assert_called_once()
    assert conversion.call_args.kwargs == {'tnd_native': True}
    assert conversion.call_args.args[2] is lengths


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_dense_option_survives_model_config_conversion():
    """The HF-style converter and common mapping both retain the new option."""
    from mindformers.models.xing4 import Xing4V41Config
    from mindformers.parallel_core.transformer_config_utils import COMMON_CONFIG_MAPPING
    assert not Xing4V41Config().convert_to_transformer_config().dsa_dense_fa_tnd_softmax
    assert Xing4V41Config(dsa_dense_fa_tnd_softmax=True).convert_to_transformer_config().dsa_dense_fa_tnd_softmax
    assert COMMON_CONFIG_MAPPING['dsa_dense_fa_tnd_softmax'] == 'dsa_dense_fa_tnd_softmax'
