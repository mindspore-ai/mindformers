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
"""Test module for testing fused RoPE in ApplyRotaryPosEmb used for mindformers."""
import pytest
import mindspore as ms
import numpy as np
from tests.utils.double_benchmark import DoubleBenchmarkComparator, DoubleBenchmarkStandard
from mindformers.pynative.base_models.common.embeddings.rope_utils import ApplyRotaryPosEmb
from mindformers.parallel_core.transformer_config import TransformerConfig

from .data_gen_utils import get_init_params

FUSED_ROPE_TEST_CASES = [
    (True, True),
    (True, False),
    (False, True),
    (False, False),
]


def deinterleave_mla_output(output, rot_dim):
    """Convert optimized fused MLA interleaved output back to the unfused layout for comparison."""
    output = output.copy()
    rotary = output[..., :rot_dim]
    output[..., :rot_dim] = np.concatenate((rotary[..., 0::2], rotary[..., 1::2]), axis=-1)
    return output


class TestFusedRoPE:
    """A test class for testing FusedRoPE kernel"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        init_params = get_init_params()
        freqs = init_params.get("freqs").copy()
        half = freqs.shape[-1] // 2
        # Match real non-interleaved RoPE/Yarn freqs: [theta..., theta...].
        freqs[..., half:] = freqs[..., :half]
        self.input_t = ms.Tensor(init_params.get("t"), dtype=ms.float32)
        self.input_freqs = ms.Tensor(freqs, dtype=ms.float32)
        self.mscale = 1.0
        self.freqs = (self.input_freqs, self.mscale)

    def run_test(self, multi_latent_attention, rotary_interleaved):
        """Helper function to run test"""
        no_fused_rope_config = TransformerConfig(num_attention_heads=1,
                                                 num_layers=1,
                                                 apply_rope_fusion=False,
                                                 rotary_dtype='fp32',
                                                 multi_latent_attention=multi_latent_attention,
                                                 rotary_interleaved=rotary_interleaved)
        fused_rope_config = TransformerConfig(num_attention_heads=1,
                                              num_layers=1,
                                              apply_rope_fusion=True,
                                              rotary_dtype='fp32',
                                              multi_latent_attention=multi_latent_attention,
                                              rotary_interleaved=rotary_interleaved)
        fused_rope_output = ApplyRotaryPosEmb(fused_rope_config)(
            self.input_t, self.freqs, rotary_interleaved, multi_latent_attention)
        no_fused_rope_output = ApplyRotaryPosEmb(no_fused_rope_config)(
            self.input_t, self.freqs, rotary_interleaved, multi_latent_attention)
        actual = fused_rope_output.asnumpy()
        golden = no_fused_rope_output.asnumpy()
        if multi_latent_attention and not rotary_interleaved:
            actual = deinterleave_mla_output(actual, self.input_freqs.shape[-1])
        standard = DoubleBenchmarkStandard(dtype="float32")
        assert DoubleBenchmarkComparator.check_pass_or_not(actual, golden, golden, standard), (
            f"FusedRoPE test failed with multi_latent_attention={multi_latent_attention}, "
            f"rotary_interleaved={rotary_interleaved}.\n"
            f"FusedRoPE output:\n{actual}\n\n"
            f"RoPE output:\n{golden}\n\n"
        )

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize("multi_latent_attention, rotary_interleaved", FUSED_ROPE_TEST_CASES)
    def test_fused_rope(self, multi_latent_attention, rotary_interleaved):
        """
        Feature: ApplyRotaryPosEmb
        Description: Test Fused RoPE,
        Exception: AssertionError
        """
        self.run_test(multi_latent_attention, rotary_interleaved)
