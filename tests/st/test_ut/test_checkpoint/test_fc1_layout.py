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
"""The fused gate|up layout of ``linear_fc1`` follows the model config.

``use_interleaved_weight_layout_mlp`` decides whether the layer spec builds
``MLPInterleaved`` or ``MLP`` (likewise ``SharedExpertMLPInterleaved`` /
``SharedExpertMLP``), which read the fused weight differently. A converter that
pins ``interleaved`` in its declaration silently disagrees with the model as
soon as that flag is flipped, and the error survives a round trip: converting
out and back with the same wrong convention reproduces the input exactly, so
only a check against the model's own layout catches it.
"""
import numpy as np
import pytest

from mindformers.checkpoint.converter.convert_op import ConcatConvertOp

FC1 = "decoder.layers.0.mlp.linear_fc1.weight"
SHARED_FC1 = "decoder.layers.0.mlp.shared_experts.linear_fc1.weight"
EH_PROJ = "mtp.layers.0.eh_proj.weight"


class _Config:
    def __init__(self, interleaved):
        self.use_interleaved_weight_layout_mlp = interleaved


def _fc1_op(mf_name=FC1, **kwargs):
    return ConcatConvertOp(hf_names=["l.w1.weight", "l.w3.weight"],
                           mf_names=[mf_name], dim=0, **kwargs)


@pytest.mark.parametrize("mf_name", [FC1, SHARED_FC1])
@pytest.mark.parametrize("interleaved", [True, False])
def test_fc1_layout_follows_config(mf_name, interleaved):
    """Both the dense MLP and the shared-expert MLP take the layout from config."""
    op = _fc1_op(mf_name)
    op.set_model_config(_Config(interleaved))
    assert op.interleaved is interleaved


def test_non_fc1_concat_keeps_its_declaration():
    """MTP's eh_proj is not a gated MLP; the flag must not reach it."""
    op = ConcatConvertOp(hf_names=["mtp.0.e_proj.weight", "mtp.0.h_proj.weight"],
                         mf_names=[EH_PROJ], dim=1, interleaved=False)
    op.set_model_config(_Config(True))
    assert op.interleaved is False


def test_config_without_the_flag_leaves_layout_alone():
    """An older config that lacks the field must not silently flip the layout."""
    op = _fc1_op()
    op.set_model_config(object())
    assert op.interleaved is True


@pytest.mark.parametrize("interleaved", [True, False])
def test_round_trip_matches_the_configured_layout(interleaved):
    """The bytes produced match what the configured MLP variant would read."""
    gate = np.arange(6, dtype=np.float32).reshape(3, 2)
    up = np.arange(100, 106, dtype=np.float32).reshape(3, 2)

    op = _fc1_op()
    op.set_model_config(_Config(interleaved))
    # pylint: disable=protected-access
    fused = op._hf_to_mf([gate, up])[0]

    if interleaved:
        # MLPInterleaved reshapes to (..., ffn // 2, 2): rows alternate.
        expected = np.empty((6, 2), dtype=np.float32)
        expected[0::2], expected[1::2] = gate, up
    else:
        # MLP splits the fused output in half: the halves stay contiguous.
        expected = np.concatenate([gate, up], axis=0)
    np.testing.assert_array_equal(fused, expected)

    # pylint: disable=protected-access
    back_gate, back_up = op._mf_to_hf([fused])
    np.testing.assert_array_equal(back_gate, gate)
    np.testing.assert_array_equal(back_up, up)
