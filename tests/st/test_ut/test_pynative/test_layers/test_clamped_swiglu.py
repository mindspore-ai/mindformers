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
"""Tests for ClampedSwiGlu Cell and activation_func_clamp_value config validation."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, context


ms.set_context(mode=context.PYNATIVE_MODE)


def _silu_np(x: np.ndarray) -> np.ndarray:
    """Numpy SiLU in float64."""
    return x / (1.0 + np.exp(-x))


def _clamped_swiglu_ref(x_np: np.ndarray, clamp_value: float) -> np.ndarray:
    """Numpy reference for Megatron clamped_swiglu, computed in float64.

    First half along last axis is the gate, second half is the linear part.
    Returns an array cast back to the input dtype (matching Megatron's
    ``res.to(dtype)``).
    """
    orig_dtype = x_np.dtype
    x64 = x_np.astype(np.float64)
    half = x64.shape[-1] // 2
    gate = x64[..., :half]
    linear = x64[..., half:]
    gate = np.minimum(gate, clamp_value)
    linear = np.clip(linear, -clamp_value, clamp_value)
    out = _silu_np(gate) * linear
    return out.astype(orig_dtype)


class TestClampedSwiGlu:
    """UT for ClampedSwiGlu cell (numeric path runs on Ascend NPU)."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_numeric_matches_megatron(self):
        """
        Feature: ClampedSwiGlu
        Description: Fixed fp32 input, compare against numpy clamped_swiglu reference.
        Exception: AssertionError
        """
        from mindformers.pynative.layers.activation import ClampedSwiGlu

        rng = np.random.default_rng(1234)
        clamp_value = 0.5
        x_np = (rng.standard_normal((4, 3, 16)).astype(np.float32)) * 2.0
        expected = _clamped_swiglu_ref(x_np, clamp_value)

        act = ClampedSwiGlu(clamp_value)
        out = act(Tensor(x_np, dtype=ms.float32)).asnumpy()

        assert out.shape == expected.shape
        np.testing.assert_allclose(out, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_clamp_takes_effect(self):
        """
        Feature: ClampedSwiGlu
        Description: Inputs far beyond clamp_value get clamped: gate <= v (so silu(gate)
                     <= silu(v)) and linear in [-v, v]; verify via the equivalent ref and
                     by checking the bound is actually exercised.
        Exception: AssertionError
        """
        from mindformers.pynative.layers.activation import ClampedSwiGlu

        clamp_value = 1.0
        half = 8
        gate = np.full((2, half), 100.0, dtype=np.float32)
        linear = np.concatenate(
            [np.full((2, half // 2), 50.0, dtype=np.float32),
             np.full((2, half // 2), -50.0, dtype=np.float32)],
            axis=-1,
        )
        x_np = np.concatenate([gate, linear], axis=-1)

        act = ClampedSwiGlu(clamp_value)
        out = act(Tensor(x_np, dtype=ms.float32)).asnumpy()

        # gate clamped to v=1.0 => silu(1.0); linear clamped to +/-1.0.
        silu_v = 1.0 / (1.0 + np.exp(-1.0))
        expected_pos = silu_v * 1.0
        expected_neg = silu_v * (-1.0)
        np.testing.assert_allclose(out[:, : half // 2], expected_pos, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(out[:, half // 2:], expected_neg, atol=1e-4, rtol=1e-4)

        assert np.all(np.abs(out) <= 1.0 + 1e-4)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_dtype_preserved_bf16(self):
        """
        Feature: ClampedSwiGlu
        Description: bf16 input -> bf16 output (clamp computed in fp32 internally).
        Exception: AssertionError
        """
        from mindformers.pynative.layers.activation import ClampedSwiGlu

        rng = np.random.default_rng(7)
        x_np = rng.standard_normal((2, 10)).astype(np.float32)
        act = ClampedSwiGlu(0.8)
        out = act(Tensor(x_np, dtype=ms.bfloat16))
        assert out.dtype == ms.bfloat16

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_output_last_dim_is_half(self):
        """
        Feature: ClampedSwiGlu
        Description: Output last dim is half the input last dim; non-default split dim works.
        Exception: AssertionError
        """
        from mindformers.pynative.layers.activation import ClampedSwiGlu

        act = ClampedSwiGlu(0.5)
        x = Tensor(np.random.randn(3, 5, 12).astype(np.float32), dtype=ms.float32)
        out = act(x)
        assert out.shape == (3, 5, 6)

        # split along dim=0 (Megatron-style first-half/second-half along that axis).
        x2 = Tensor(np.random.randn(8, 4).astype(np.float32), dtype=ms.float32)
        out2 = act(x2, dim=0)
        assert out2.shape == (4, 4)


class TestClampedSwiGluConfigValidation:
    """Config-field validation for activation_func_clamp_value.

    Pure dataclass checks -> runnable off-Ascend (platform_x86_cpu).
    """

    @staticmethod
    def _swiglu_kwargs(**overrides):
        """Build a minimal SwiGLU TransformerConfig kwargs dict with overrides."""
        kwargs = {
            "num_layers": 2,
            "num_attention_heads": 4,
            "hidden_act": "swiglu",
            "bias_swiglu_fusion": True,
            "gated_linear_unit": True,
            # MoE path requires add_bias_linear=False (existing TransformerConfig
            # constraint, unrelated to clamp).
            "add_bias_linear": False,
        }
        kwargs.update(overrides)
        return kwargs

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_clamp_valid_with_moe(self):
        """clamp_value + SwiGLU + MoE (num_moe_experts set) is valid."""
        from mindformers.parallel_core.transformer_config import TransformerConfig

        config = TransformerConfig(
            **self._swiglu_kwargs(activation_func_clamp_value=0.5, num_moe_experts=8)
        )
        assert config.activation_func_clamp_value == 0.5

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_clamp_default_none(self):
        """Default activation_func_clamp_value is None (backward compatible)."""
        from mindformers.parallel_core.transformer_config import TransformerConfig

        config = TransformerConfig(**self._swiglu_kwargs(num_moe_experts=8))
        assert config.activation_func_clamp_value is None

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_clamp_swiglu_without_moe_raises(self):
        """clamp_value + SwiGLU but no MoE -> ValueError (MoE-only, aligns Megatron)."""
        from mindformers.parallel_core.transformer_config import TransformerConfig

        with pytest.raises(ValueError) as exc_info:
            TransformerConfig(
                **self._swiglu_kwargs(activation_func_clamp_value=0.5, num_moe_experts=None)
            )
        assert "MoE" in str(exc_info.value)


def _mlp_clamp_kwargs(**overrides):
    """Build a minimal TransformerConfig kwargs dict for MLP with ClampedSwiGlu.

    ``num_moe_experts`` is set to satisfy config validation (clamp requires MoE).
    The MLP itself is dense; ``num_moe_experts`` only bypasses the validation gate.
    """
    kwargs = {
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_layers": 1,
        "ffn_hidden_size": 32,
        "hidden_act": "fusedswiglu",
        "gated_linear_unit": True,
        "add_bias_linear": False,
        "num_moe_experts": 4,
        "compute_dtype": "float32",
        "params_dtype": "float32",
    }
    kwargs.update(overrides)
    return kwargs


class TestMLPClampedSwiGluSelection:
    """MLP activation selection: ClampedSwiGlu vs FusedSwiGlu (runs on Ascend NPU
    because MLP.__init__ allocates device memory via Linear -> mint.empty)."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_mlp_uses_clamped_swiglu_when_configured(self):
        """
        Feature: MLP ClampedSwiGlu selection
        Description: MLP with activation_func_clamp_value set -> activation_func is
                     ClampedSwiGlu and use_clamped_swiglu flag is True.
        Exception: AssertionError
        """
        from mindformers.parallel_core.transformer_config import TransformerConfig
        from mindformers.pynative.transformers.mlp import MLP, MLPSubmodules
        from mindformers.pynative.layers.linear import Linear
        from mindformers.pynative.layers.activation import ClampedSwiGlu

        clamp_value = 0.5
        config = TransformerConfig(
            **_mlp_clamp_kwargs(activation_func_clamp_value=clamp_value)
        )
        mlp = MLP(config, MLPSubmodules(linear_fc1=Linear, linear_fc2=Linear))

        assert mlp.use_clamped_swiglu is True
        assert isinstance(mlp.activation_func, ClampedSwiGlu)
        assert mlp.activation_func.clamp_value == clamp_value

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_mlp_uses_fused_swiglu_without_clamp(self):
        """
        Feature: MLP ClampedSwiGlu selection
        Description: MLP without activation_func_clamp_value (default None) ->
                     activation_func is FusedSwiGlu, NOT ClampedSwiGlu.
        Exception: AssertionError
        """
        from mindformers.parallel_core.transformer_config import TransformerConfig
        from mindformers.pynative.transformers.mlp import MLP, MLPSubmodules
        from mindformers.pynative.layers.linear import Linear
        from mindformers.pynative.layers.activation import ClampedSwiGlu, FusedSwiGlu

        config = TransformerConfig(**_mlp_clamp_kwargs())
        assert config.activation_func_clamp_value is None
        mlp = MLP(config, MLPSubmodules(linear_fc1=Linear, linear_fc2=Linear))

        assert mlp.use_clamped_swiglu is False
        assert not isinstance(mlp.activation_func, ClampedSwiGlu)
        assert isinstance(mlp.activation_func, FusedSwiGlu)


class TestMLPClampedSwiGluNumeric:
    """MLP numeric correctness with ClampedSwiGlu activation (runs on Ascend NPU)."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_mlp_clamped_swiglu_output_shape(self):
        """
        Feature: MLP ClampedSwiGlu
        Description: MLP with ClampedSwiGlu produces output shape (seq, bs, hidden_size).
        Exception: AssertionError
        """
        from mindformers.parallel_core.transformer_config import TransformerConfig
        from mindformers.pynative.transformers.mlp import MLP, MLPSubmodules
        from mindformers.pynative.layers.linear import Linear

        clamp_value = 0.5
        config = TransformerConfig(
            **_mlp_clamp_kwargs(activation_func_clamp_value=clamp_value)
        )
        mlp = MLP(config, MLPSubmodules(linear_fc1=Linear, linear_fc2=Linear))

        seq, bs = 4, 2
        x = Tensor(np.random.randn(seq, bs, 16).astype(np.float32), dtype=ms.float32)
        out = mlp(x)
        assert out.shape == (seq, bs, 16)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_mlp_clamped_swiglu_numeric_precision(self):
        """
        Feature: MLP ClampedSwiGlu
        Description: End-to-end MLP output matches manual step-by-step computation
                     (linear_fc1 -> de-interleave -> ClampedSwiGlu -> linear_fc2).
        Exception: AssertionError
        """
        from mindformers.parallel_core.transformer_config import TransformerConfig
        from mindformers.pynative.transformers.mlp import MLP, MLPSubmodules
        from mindformers.pynative.layers.linear import Linear

        clamp_value = 0.5
        hidden_size = 8
        ffn_hidden_size = 16
        config = TransformerConfig(
            **_mlp_clamp_kwargs(
                activation_func_clamp_value=clamp_value,
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
            )
        )
        mlp = MLP(config, MLPSubmodules(linear_fc1=Linear, linear_fc2=Linear))

        rng = np.random.default_rng(42)
        fc1_w = rng.standard_normal((ffn_hidden_size * 2, hidden_size)).astype(np.float32) * 0.1
        fc2_w = rng.standard_normal((hidden_size, ffn_hidden_size)).astype(np.float32) * 0.1
        mlp.linear_fc1.weight.set_data(Tensor(fc1_w, dtype=ms.float32))
        mlp.linear_fc2.weight.set_data(Tensor(fc2_w, dtype=ms.float32))

        x_np = rng.standard_normal((2, 3, hidden_size)).astype(np.float32)
        x = Tensor(x_np, dtype=ms.float32)
        out = mlp(x).asnumpy()

        intermediate = x_np @ fc1_w.T
        seq, bs_i, ffn2 = intermediate.shape
        ffn = ffn2 // 2
        reshaped = intermediate.reshape(seq, bs_i, ffn, 2)
        gate_interleaved = reshaped[..., 0]
        linear_interleaved = reshaped[..., 1]
        de_interleaved = np.concatenate([gate_interleaved, linear_interleaved], axis=-1)
        act_out = _clamped_swiglu_ref(de_interleaved, clamp_value)
        expected = act_out @ fc2_w.T

        assert out.shape == expected.shape
        np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)


def _grouped_mlp_kwargs(**overrides):
    """Build minimal MoE TransformerConfig kwargs for GroupedMLP.
    hidden_act='fusedswiglu' required (ClampedSwiGlu switch keys off it);
    add_bias_linear=False (GroupedMLP rejects bias).
    """
    kwargs = {
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_layers": 2,
        "hidden_act": "fusedswiglu",
        "gated_linear_unit": True,
        "num_moe_experts": 4,
        "moe_router_topk": 2,
        "moe_ffn_hidden_size": 32,
        "add_bias_linear": False,
        "compute_dtype": "float32",
        "params_dtype": "float32",
    }
    kwargs.update(overrides)
    return kwargs


def _manual_grouped_mlp_forward(hidden_states_np, w1_np, w2_np, clamp_value,
                                 num_tokens_per_expert, num_experts):
    """Manual step-by-step GroupedMLP forward with ClampedSwiGlu (numpy).

    Mirrors GroupedMLP.experts_forward:
      1. GroupedMatmul fc1: split input by expert, matmul each with its w1 slice
      2. ClampedSwiGlu activation on fc1_output
      3. GroupedMatmul fc2: split intermediate by expert, matmul each with its w2 slice
    """
    ffn_hidden_size = w2_np.shape[1]

    token_counts = np.diff(np.concatenate([[0], num_tokens_per_expert]))

    fc1_parts = []
    offset = 0
    for e in range(num_experts):
        n = int(token_counts[e])
        x_e = hidden_states_np[offset:offset + n]
        fc1_e = x_e @ w1_np[e]
        fc1_parts.append(fc1_e)
        offset += n
    fc1_output = np.concatenate(fc1_parts, axis=0)

    act_output = _clamped_swiglu_ref(fc1_output, clamp_value)
    intermediate = act_output[:, :ffn_hidden_size]

    fc2_parts = []
    offset = 0
    for e in range(num_experts):
        n = int(token_counts[e])
        inter_e = intermediate[offset:offset + n]
        fc2_e = inter_e @ w2_np[e]
        fc2_parts.append(fc2_e)
        offset += n
    fc2_output = np.concatenate(fc2_parts, axis=0)

    return fc2_output


class TestGroupedMLPClampedSwiGluSelection:
    """GroupedMLP activation selection: ClampedSwiGlu vs FusedSwiGlu (runs on
    Ascend NPU because GroupedMLP.__init__ allocates device memory via mint.empty)."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_grouped_mlp_uses_clamped_swiglu_when_configured(self):
        """
        Feature: GroupedMLP ClampedSwiGlu selection
        Description: fusedswiglu MoE + activation_func_clamp_value set -> activation_func
                     is ClampedSwiGlu carrying the configured clamp_value (experts.py:71-73).
        Exception: AssertionError
        """
        from mindformers.parallel_core.transformer_config import TransformerConfig
        from mindformers.pynative.transformers.moe.experts import GroupedMLP
        from mindformers.pynative.layers.activation import ClampedSwiGlu

        clamp_value = 0.5
        config = TransformerConfig(
            **_grouped_mlp_kwargs(activation_func_clamp_value=clamp_value)
        )
        grouped_mlp = GroupedMLP(config)

        assert isinstance(grouped_mlp.activation_func, ClampedSwiGlu)
        assert grouped_mlp.activation_func.clamp_value == clamp_value

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_grouped_mlp_uses_plain_swiglu_without_clamp(self):
        """
        Feature: GroupedMLP ClampedSwiGlu selection
        Description: fusedswiglu MoE with clamp_value None (default) -> activation_func
                     is the plain FusedSwiGlu, NOT ClampedSwiGlu (experts.py:74-75 else branch).
        Exception: AssertionError
        """
        from mindformers.parallel_core.transformer_config import TransformerConfig
        from mindformers.pynative.transformers.moe.experts import GroupedMLP
        from mindformers.pynative.layers.activation import ClampedSwiGlu, FusedSwiGlu

        config = TransformerConfig(**_grouped_mlp_kwargs())
        assert config.activation_func_clamp_value is None
        grouped_mlp = GroupedMLP(config)

        assert not isinstance(grouped_mlp.activation_func, ClampedSwiGlu)
        assert isinstance(grouped_mlp.activation_func, FusedSwiGlu)


class TestGroupedMLPExpertsForwardClampedSwiGlu:
    """GroupedMLP.experts_forward numeric correctness with ClampedSwiGlu.
    Calls experts_forward directly (bypassing permute/unpermute) to isolate the
    fc1 → ClampedSwiGlu → fc2 path. Runs on Ascend NPU.
    """

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_experts_forward_clamped_swiglu_numeric_precision(self):
        """
        Feature: GroupedMLP experts_forward ClampedSwiGlu
        Description: experts_forward output matches manual step-by-step computation
                     (GroupedMatmul fc1 -> ClampedSwiGlu -> GroupedMatmul fc2).
        Exception: AssertionError
        """
        from mindformers.parallel_core.transformer_config import TransformerConfig
        from mindformers.pynative.transformers.moe.experts import GroupedMLP
        from mindformers.pynative.layers.activation import ClampedSwiGlu

        hidden_size = 8
        ffn_hidden_size = 16
        num_experts = 2
        clamp_value = 0.5

        config = TransformerConfig(
            **_grouped_mlp_kwargs(
                activation_func_clamp_value=clamp_value,
                hidden_size=hidden_size,
                moe_ffn_hidden_size=ffn_hidden_size,
                num_moe_experts=num_experts,
            )
        )
        grouped_mlp = GroupedMLP(config)

        assert isinstance(grouped_mlp.activation_func, ClampedSwiGlu)

        rng = np.random.default_rng(42)
        w1_np = rng.standard_normal(
            (num_experts, hidden_size, ffn_hidden_size * 2)).astype(np.float32) * 0.1
        w2_np = rng.standard_normal(
            (num_experts, ffn_hidden_size, hidden_size)).astype(np.float32) * 0.1
        grouped_mlp.weight1.set_data(Tensor(w1_np, dtype=ms.float32))
        grouped_mlp.weight2.set_data(Tensor(w2_np, dtype=ms.float32))

        token_counts = [3, 2]
        total_tokens = sum(token_counts)
        tokens_per_expert_cumsum = np.cumsum(token_counts).tolist()

        x_np = rng.standard_normal((total_tokens, hidden_size)).astype(np.float32)
        x = Tensor(x_np, dtype=ms.float32)
        tokens_per_expert = Tensor(tokens_per_expert_cumsum, dtype=ms.int64)

        out = grouped_mlp.experts_forward(x, tokens_per_expert).asnumpy()

        expected = _manual_grouped_mlp_forward(
            x_np, w1_np, w2_np, clamp_value, tokens_per_expert_cumsum, num_experts
        )

        assert out.shape == expected.shape
        np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_experts_forward_output_shape(self):
        """
        Feature: GroupedMLP experts_forward ClampedSwiGlu
        Description: experts_forward output shape is (total_tokens, hidden_size).
        Exception: AssertionError
        """
        from mindformers.parallel_core.transformer_config import TransformerConfig
        from mindformers.pynative.transformers.moe.experts import GroupedMLP

        hidden_size = 16
        ffn_hidden_size = 32
        num_experts = 4
        clamp_value = 0.5

        config = TransformerConfig(
            **_grouped_mlp_kwargs(
                activation_func_clamp_value=clamp_value,
                hidden_size=hidden_size,
                moe_ffn_hidden_size=ffn_hidden_size,
                num_moe_experts=num_experts,
            )
        )
        grouped_mlp = GroupedMLP(config)

        token_counts = [2, 3, 1, 2]
        total_tokens = sum(token_counts)
        tokens_per_expert_cumsum = np.cumsum(token_counts).tolist()

        x = Tensor(np.random.randn(total_tokens, hidden_size).astype(np.float32), dtype=ms.float32)
        tokens_per_expert = Tensor(tokens_per_expert_cumsum, dtype=ms.int64)

        out = grouped_mlp.experts_forward(x, tokens_per_expert)
        assert out.shape == (total_tokens, hidden_size)
