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
"""Check that FlashAttention input and output shapes match their declared layouts."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor

from mindformers.pynative.layers.flash_attention import FlashAttention
from mindformers.parallel_core.transformer_config import TransformerConfig


def _build(input_layout, num_heads=2, head_dim=3):
    # Unequal head count and dimension expose accidental swaps of the last axes.
    config = TransformerConfig(
        multi_latent_attention=False,
        hidden_size=num_heads * head_dim,
        num_attention_heads=num_heads,
        num_layers=1,
    )
    config.input_layout = input_layout
    return FlashAttention(config=config, layer_number=1)


class _Recorder:
    """Record kernel inputs and return an output with the requested shape."""

    def __init__(self, out_shape):
        self.seen = None
        self._out_shape = out_shape

    def __call__(self, query, key, value, *args, **kwargs):
        del key, value, args
        self.seen = query.shape
        out = Tensor(np.zeros(self._out_shape, np.float16))
        softmax_val = Tensor(np.zeros((1, 1, 1, 1), np.float32))
        return softmax_val, None, None, out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_bsnd_hands_the_operator_four_dimensions():
    """Ordinary attention in DSA models must keep BSND inputs four-dimensional."""
    seq, bsz, heads, dim = 4, 3, 2, 3
    fa = _build("BSND", heads, dim)
    fa.flash_attention = _Recorder((bsz, seq, heads, dim))

    q = Tensor(np.zeros((seq, bsz, heads, dim), np.float16))
    output = fa(q, q, q, None)

    assert fa.flash_attention.seen == (bsz, seq, heads, dim), (
        "BSND kernels require four-dimensional (B, S, N, D) inputs"
    )
    # The caller merges heads, so restore SBND here.
    assert output.shape == (seq, bsz, heads, dim)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize(
    "layout, expect_in, expect_out",
    [
        ("BSH", (3, 4, 6), (4, 3, 6)),         # (S,B,H) -> (B,S,H) -> (S,B,H)
        ("BNSD", (3, 2, 4, 3), (3, 2, 4, 3)),  # _merge_heads handles the BNSD output.
    ],
)
def test_existing_layouts_are_unchanged(layout, expect_in, expect_out):
    """The BSND branch must preserve existing BSH and BNSD behavior."""
    seq, bsz, heads, dim = 4, 3, 2, 3
    fa = _build(layout, heads, dim)
    fa.flash_attention = _Recorder(expect_out)

    if layout == "BSH":
        q = Tensor(np.zeros((seq, bsz, heads * dim), np.float16))
    else:
        q = Tensor(np.zeros((seq, bsz, heads, dim), np.float16))
    fa(q, q, q, None)
    assert fa.flash_attention.seen == expect_in


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("method,degree", [("colossal", 1), ("ulysses", 4), ("hybrid", 2)])
def test_cp_style_hands_bsnd_four_dimensions(method, degree):
    """The CP pre-hook must prepare BSND inputs before the attention cell runs."""
    from mindformers.pynative.distributed.context_parallel import (
        _prepare_attention_qkv,
        _restore_attention_output,
    )

    seq, bsz, heads, dim = 4, 3, 8, 3
    q = Tensor(np.zeros((seq, bsz, heads, dim), np.float16))

    from mindformers.pynative.distributed.style import build_hp_cp_style

    prepared = _prepare_attention_qkv(method, (q, q, q), input_layout="BSND")
    style = build_hp_cp_style(method, 4, degree, input_layout="BSND").hp_style
    assert prepared[0].shape[style.seq_dim] == seq
    assert prepared[0].shape[style.head_dim] == heads
    assert prepared[0].shape == (bsz, seq, heads, dim), "BSND must retain four dimensions and move batch to the front"

    # Restore the BSND kernel output to model-native SBH.
    fa_out = Tensor(np.zeros((bsz, seq, heads, dim), np.float16))
    restored = _restore_attention_output(method, fa_out, input_layout="BSND")
    assert restored.shape == (seq, bsz, heads * dim)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_cp_style_other_layouts_unchanged():
    """Preserve the existing colossal BSH, Ulysses BNSD and TND paths."""
    from mindformers.pynative.distributed.context_parallel import _prepare_attention_qkv

    seq, bsz, heads, dim = 4, 3, 2, 3
    q = Tensor(np.zeros((seq, bsz, heads, dim), np.float16))

    assert _prepare_attention_qkv("colossal", (q, q, q))[0].shape == (bsz, seq, heads * dim)
    assert _prepare_attention_qkv("ulysses", (q, q, q))[0].shape == (bsz, heads, seq, dim)
    # TND passes through unchanged.
    assert _prepare_attention_qkv("colossal", (q, q, q), input_layout="TND")[0].shape == q.shape
