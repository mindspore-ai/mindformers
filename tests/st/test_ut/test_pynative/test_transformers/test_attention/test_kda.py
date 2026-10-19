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
"""Configuration and dependency boundaries for fused KDA training."""
import pytest

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.transformers import kda


def make_config(**overrides):
    """Build a small hidden-size configuration with kernel-compatible heads."""
    values = {
        "hidden_size": 16, "num_layers": 1, "num_attention_heads": 1,
        "experimental_attention_variant": "kda", "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_num_key_heads": 1, "linear_num_value_heads": 1,
        "compute_dtype": "bfloat16", "params_dtype": "float32",
    }
    return TransformerConfig(**(values | overrides))


@pytest.mark.parametrize("overrides,message", [
    ({"linear_key_head_dim": 4, "linear_value_head_dim": 4}, "128"),
    ({"kda_chunk_size": 128}, "kda_chunk_size=64"),
    ({"compute_dtype": "float32"}, "compute_dtype"),
    ({"kda_backend": "primitive"}, "fused training backend"),
])
def test_fused_kda_rejects_unsupported_training_config(overrides, message):
    """Reject unsupported backward configurations before launching a kernel."""
    with pytest.raises(ValueError, match=message):
        make_config(**overrides)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_fused_kda_accepts_supported_compute_dtype(dtype):
    """Both low-precision input formats are accepted by fused KDA."""
    assert make_config(compute_dtype=dtype).kda_backend == "fused"


def test_non_kda_configuration_does_not_apply_kernel_restrictions():
    """Ordinary attention can retain FP32 and unrelated KDA field defaults."""
    TransformerConfig(hidden_size=16, num_layers=1, num_attention_heads=1,
                      compute_dtype="float32")


def test_missing_fused_dependency_reports_actionable_error(monkeypatch):
    """Missing HP support is reported before a fused KDA layer is built."""
    monkeypatch.setattr(kda, "npu_chunk_kda", None)
    with pytest.raises(ImportError, match="Hyper-Parallel PR #895"):
        kda.KimiDeltaAttention(make_config(), kda.KimiDeltaAttentionSubmodules())


def test_safe_gate_initializes_a_log_to_zero(monkeypatch):
    """Safe-gate KDA uses the reference initialization for A_log."""
    from types import SimpleNamespace

    class DummyAttention:
        """Minimal holder for testing the parameter initialization method."""

        _reset_a_log = kda.KimiDeltaAttention._reset_a_log

        def __init__(self):
            self.config = SimpleNamespace(kda_safe_gate=True)
            setattr(self, "A_log", SimpleNamespace(value="nonzero"))

            def copy_(value):
                self.A_log.value = value

            self.A_log.copy_ = copy_

    monkeypatch.setattr(kda.ops, "zeros_like", lambda value: ("zeros_like", value))
    attention = DummyAttention()

    attention._reset_a_log()

    assert attention.A_log.value == ("zeros_like", attention.A_log)


def test_beta_sigmoid_is_computed_in_fp32(monkeypatch):
    """The sigmoid input and result use FP32 even with BF16 projections."""
    import mindspore as ms

    calls = []

    def cast(value, dtype):
        calls.append(("cast", value, dtype))
        return ("fp32", value)

    def sigmoid(value):
        calls.append(("sigmoid", value))
        return "sigmoid_result"

    monkeypatch.setattr(kda.ops, "cast", cast)
    monkeypatch.setattr(kda.ops, "sigmoid", sigmoid)
    result = kda.KimiDeltaAttention._compute_beta("bf16_beta")

    assert result == "sigmoid_result"
    assert calls == [
        ("cast", "bf16_beta", ms.float32),
        ("sigmoid", ("fp32", "bf16_beta")),
    ]


def test_gpt_layer_spec_uses_kda_module():
    """The GPT entry point resolves KDA from its dedicated module."""
    from mindformers.pynative.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    spec = get_gpt_layer_local_spec(attention_variant="kda")
    assert spec.submodules.self_attention.module is kda.KimiDeltaAttention
    assert spec.submodules.self_attention.submodules.in_proj is not None
