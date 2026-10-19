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
"""Minimal single-card Kimi Delta Attention for Pynative training."""

from dataclasses import dataclass
from typing import Union

import mindspore as ms
from mindspore import Parameter, Tensor, mint, nn, ops

try:
    from hyper_parallel.custom_ops.experimental import npu_chunk_kda
except ImportError:
    npu_chunk_kda = None

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module


@dataclass
class KimiDeltaAttentionSubmodules:
    """Submodules required by the minimal KDA attention layer."""

    in_proj: Union[ModuleSpec, type] = None
    beta_proj: Union[ModuleSpec, type] = None
    f_proj: Union[ModuleSpec, type] = None
    f_a_proj: Union[ModuleSpec, type] = None
    f_b_proj: Union[ModuleSpec, type] = None
    g_proj: Union[ModuleSpec, type] = None
    g_a_proj: Union[ModuleSpec, type] = None
    g_b_proj: Union[ModuleSpec, type] = None
    out_norm: Union[ModuleSpec, type] = None
    out_proj: Union[ModuleSpec, type] = None


class ShortConv(nn.Cell):
    """Stateless causal depthwise Conv1d fallback for KDA training.

    The implementation follows Megatron's ``F.conv1d`` fallback: input is
    converted from ``[B, T, C]`` to ``[B, C, T]``, symmetrically padded by
    ``kernel_size - 1``, cropped back to the first ``T`` positions to preserve
    causality, and activated with SiLU.  It intentionally has no recurrent
    convolution state because inference is outside this module's scope.
    """

    def __init__(self, channels, kernel_size, params_dtype=ms.float32, has_bias=False):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.has_bias = has_bias
        self.weight = Parameter(
            mint.empty((channels, 1, kernel_size), dtype=params_dtype),
            name="weight",
        )
        self.bias = (
            Parameter(mint.empty((channels,), dtype=params_dtype), name="bias")
            if has_bias
            else None
        )
    def reset_parameter(self):
        """Match the default depthwise Conv1d fan-in initialization."""
        bound = self.kernel_size ** -0.5
        self.weight.uniform_(-bound, bound)
        if self.bias is not None:
            self.bias.uniform_(-bound, bound)

    def construct(self, x):
        """Apply causal depthwise convolution to a ``[B, T, C]`` tensor."""
        if x.ndim != 3:
            raise ValueError(f"ShortConv input must have shape [B, T, C], got {x.shape}.")
        if x.shape[-1] != self.channels:
            raise ValueError(
                f"ShortConv expected {self.channels} channels, got {x.shape[-1]}."
            )

        seq_len = x.shape[1]
        conv_input = ops.transpose(x, (0, 2, 1))
        weight = ops.cast(self.weight, conv_input.dtype)
        bias = ops.cast(self.bias, conv_input.dtype) if self.bias is not None else None
        conv_out = mint.nn.functional.conv1d(
            conv_input,
            weight,
            bias=bias,
            stride=1,
            padding=self.kernel_size - 1,
            dilation=1,
            groups=self.channels,
        )
        conv_out = mint.nn.functional.silu(conv_out[..., :seq_len])
        return ops.transpose(conv_out, (0, 2, 1))


class KimiDeltaAttention(nn.Cell):
    """Minimal direct-projection KDA module for single-card training.

    This implementation includes the native MindSpore ShortConv fallback and
    intentionally excludes packed sequences, inference state/cache handling,
    and all TP/PP/CP execution paths. The default fused recurrence requires
    Hyper-Parallel's differentiable Chunk KDA and the FLA-NPU OPP, with
    128-dimensional heads, 64-token chunks, and FP16/BF16 computation.
    """

    attention_variant = "kda"

    def __init__(
            self,
            config: TransformerConfig,
            submodules: KimiDeltaAttentionSubmodules,
            layer_number: int = 1,
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.hidden_size = config.hidden_size
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.qk_dim = self.num_key_heads * self.key_head_dim
        self.v_dim = self.num_value_heads * self.value_head_dim
        self.gate_dim = self.num_value_heads * self.key_head_dim
        self.chunk_size = config.kda_chunk_size
        if npu_chunk_kda is None:
            raise ImportError(
                "Fused KDA requires hyper_parallel.custom_ops.experimental.npu_chunk_kda "
                "(Hyper-Parallel PR #895) and the FLA-NPU ChunkKdaFwd/ChunkKdaBwd OPP. "
            )
        self.conv_dim = 2 * self.qk_dim + self.v_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        params_dtype = config.params_dtype
        if isinstance(params_dtype, str):
            params_dtype = getattr(ms, params_dtype)
        self.conv1d = ShortConv(
            channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            params_dtype=params_dtype,
            has_bias=False,
        )

        self.use_legacy_fused_projections = (
            config.kda_f_lora_rank is None and config.kda_gate_lora_rank is None
        )
        if self.use_legacy_fused_projections:
            in_proj_dim = 2 * self.qk_dim + 2 * self.v_dim + self.gate_dim
        else:
            # Low-rank/full-rank F and output-gate paths are separate from Q/K/V.
            in_proj_dim = 2 * self.qk_dim + self.v_dim
        linear_kwargs = {
            "compute_dtype": config.compute_dtype,
            "params_dtype": config.params_dtype,
            "init_method": config.init_method,
            "bias": config.add_bias_linear,
        }
        self.in_proj = build_module(
            submodules.in_proj,
            input_size=self.hidden_size,
            output_size=in_proj_dim,
            **linear_kwargs,
        )
        # Beta remains a standalone head-wise projection in both layouts.
        self.beta_proj = build_module(
            submodules.beta_proj,
            input_size=self.hidden_size,
            output_size=self.num_value_heads,
            **linear_kwargs,
        )
        if not self.use_legacy_fused_projections:
            if config.kda_f_lora_rank is None:
                self.f_proj = build_module(
                    submodules.f_proj,
                    input_size=self.hidden_size,
                    output_size=self.gate_dim,
                    **{**linear_kwargs, "bias": False},
                )
            else:
                self.f_a_proj = build_module(
                    submodules.f_a_proj,
                    input_size=self.hidden_size,
                    output_size=config.kda_f_lora_rank,
                    **{**linear_kwargs, "bias": False},
                )
                self.f_b_proj = build_module(
                    submodules.f_b_proj,
                    input_size=config.kda_f_lora_rank,
                    output_size=self.gate_dim,
                    **{**linear_kwargs, "bias": False},
                )
            if config.kda_gate_lora_rank is None:
                self.g_proj = build_module(
                    submodules.g_proj,
                    input_size=self.hidden_size,
                    output_size=self.v_dim,
                    **{**linear_kwargs, "bias": False},
                )
            else:
                self.g_a_proj = build_module(
                    submodules.g_a_proj,
                    input_size=self.hidden_size,
                    output_size=config.kda_gate_lora_rank,
                    **{**linear_kwargs, "bias": False},
                )
                self.g_b_proj = build_module(
                    submodules.g_b_proj,
                    input_size=config.kda_gate_lora_rank,
                    output_size=self.v_dim,
                    **{**linear_kwargs, "bias": False},
                )
        self.out_norm = build_module(
            submodules.out_norm,
            dim=self.value_head_dim,
            eps=config.layernorm_epsilon,
            compute_dtype=config.layernorm_compute_dtype,
        )
        self.out_proj = build_module(
            submodules.out_proj,
            input_size=self.v_dim,
            output_size=self.hidden_size,
            compute_dtype=config.compute_dtype,
            params_dtype=config.params_dtype,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
        )

        # Preserve the reference checkpoint key when registering the decay parameter.
        a_log = Parameter(mint.zeros((self.num_value_heads,), dtype=ms.float32), name="A_log")
        setattr(self, "A_log", a_log)
        self.dt_bias = Parameter(
            mint.zeros((self.num_value_heads, self.key_head_dim), dtype=ms.float32),
            name="dt_bias",
        )
        self.in_proj_split_sections = (
            (self.qk_dim, self.qk_dim, self.v_dim, self.gate_dim, self.v_dim)
            if self.use_legacy_fused_projections
            else (self.qk_dim, self.qk_dim, self.v_dim)
        )

    @staticmethod
    def _l2_normalize(x):
        """Normalize each query/key head in FP32."""
        x = ops.cast(x, ms.float32)
        inv_norm = ops.rsqrt(ops.sum(x * x, dim=-1, keepdim=True) + Tensor(1e-6, ms.float32))
        return x * inv_norm

    def _compute_decay(self, raw_gate):
        """Convert raw projected gates to channel-wise log decays."""
        raw_gate = ops.cast(raw_gate, ms.float32)
        gate_input = raw_gate + self.dt_bias.reshape(1, 1, self.num_value_heads, self.key_head_dim)
        decay_rate = ops.exp(self.A_log).reshape(1, 1, self.num_value_heads, 1)
        if self.config.kda_safe_gate:
            return Tensor(self.config.kda_lower_bound, ms.float32) * ops.sigmoid(
                decay_rate * gate_input
            )
        return -decay_rate * ops.softplus(gate_input)

    @staticmethod
    def _compute_beta(raw_beta):
        """Apply the beta sigmoid in FP32 to preserve update precision."""
        return ops.sigmoid(ops.cast(raw_beta, ms.float32))

    def reset_parameter(self):
        """Initialize KDA-owned decay parameters after delayed materialization.

        Ordinary projections, convolution, and normalization parameters are reset by
        their own leaf cells during ``TrainModelMixin.init_states``.  Re-resetting
        them here would make the attention cell reset the same parameters twice.
        """
        self._reset_a_log()
        self._reset_dt_bias()

    def _reset_a_log(self):
        """Initialize ``A_log`` according to the selected KDA gate mode."""
        local_a_log = self.A_log.to_local() if hasattr(self.A_log, "to_local") else self.A_log
        if self.config.kda_safe_gate:
            if hasattr(local_a_log, "shape"):
                value = ops.zeros(local_a_log.shape, ms.float32)
            else:
                value = ops.zeros_like(self.A_log)
            KimiDeltaAttention._assign_parameter(self.A_log, value)
            return
        shape = local_a_log.shape
        a = ops.uniform(
            shape,
            Tensor(1.0, ms.float32),
            Tensor(16.0, ms.float32),
            dtype=ms.float32,
        )
        KimiDeltaAttention._assign_parameter(self.A_log, ops.log(a))

    @staticmethod
    def _assign_parameter(parameter, value):
        """Assign a local initialization to a plain Parameter or Hyper-Parallel DTensor."""
        if hasattr(parameter, "to_local") and hasattr(parameter, "set_data"):
            parameter.set_data(value)
        else:
            parameter.copy_(value)

    def _reset_dt_bias(self):
        """Initialize channel-wise ``dt_bias`` in inverse-softplus space.

        NVIDIA KDA samples the target positive step size log-uniformly from
        ``[1e-3, 1e-1]`` and stores its inverse-softplus value.  Consequently,
        when the projected gate is initially close to zero,
        ``softplus(dt_bias)`` recovers the sampled step size while allowing the
        parameter to remain unconstrained during optimization.
        """
        log_dt_min = Tensor(-6.907755278982137, ms.float32)
        log_dt_max = Tensor(-2.302585092994046, ms.float32)
        shape = getattr(self.dt_bias, "to_local", lambda: self.dt_bias)().shape
        random_unit = ops.uniform(
            shape,
            Tensor(0.0, ms.float32),
            Tensor(1.0, ms.float32),
            dtype=ms.float32,
        )
        dt = ops.exp(random_unit * (log_dt_max - log_dt_min) + log_dt_min)
        # log(exp(dt) - 1), written with expm1 for stable small-dt values.
        inverse_softplus = dt + ops.log(-ops.expm1(-dt))
        KimiDeltaAttention._assign_parameter(self.dt_bias, inverse_softplus)

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=None,
            prefix_keys_values=None,
            actual_seq_len=None,
            mscale=1.0,
            rotary_cos_sin=None,
    ):
        """Run the minimal SBH single-card training path."""
        del rotary_pos_emb, mscale, rotary_cos_sin
        if attention_mask is not None:
            raise NotImplementedError("KDA does not support attention_mask in the current SBH path.")
        if prefix_keys_values is not None:
            raise NotImplementedError("KDA inference prefix/KV cache is outside the minimal scope.")
        if actual_seq_len is not None:
            raise NotImplementedError("Packed or variable-length KDA input is outside the minimal scope.")

        seq_len, batch, _ = hidden_states.shape
        projected = self.in_proj(hidden_states)
        if self.use_legacy_fused_projections:
            q, k, v, raw_gate, output_gate = mint.split(
                projected, self.in_proj_split_sections, dim=-1
            )
        else:
            q, k, v = mint.split(projected, self.in_proj_split_sections, dim=-1)
            if self.config.kda_f_lora_rank is None:
                raw_gate = self.f_proj(hidden_states)
            else:
                raw_gate = self.f_b_proj(self.f_a_proj(hidden_states))
            if self.config.kda_gate_lora_rank is None:
                output_gate = self.g_proj(hidden_states)
            else:
                output_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        beta = self.beta_proj(hidden_states)

        qkv = ops.concat((q, k, v), axis=-1)
        qkv = self.conv1d(ops.transpose(qkv, (1, 0, 2)))
        q, k, v = mint.split(
            ops.transpose(qkv, (1, 0, 2)),
            (self.qk_dim, self.qk_dim, self.v_dim),
            dim=-1,
        )

        def to_bthd(tensor, num_heads, head_dim):
            tensor = tensor.reshape(seq_len, batch, num_heads, head_dim)
            return ops.transpose(tensor, (1, 0, 2, 3))

        q = self._l2_normalize(to_bthd(q, self.num_key_heads, self.key_head_dim))
        k = self._l2_normalize(to_bthd(k, self.num_key_heads, self.key_head_dim))
        v = to_bthd(v, self.num_value_heads, self.value_head_dim)
        raw_gate = to_bthd(raw_gate, self.num_value_heads, self.key_head_dim)
        output_gate = to_bthd(output_gate, self.num_value_heads, self.value_head_dim)
        beta = self._compute_beta(beta)
        beta = ops.transpose(
            beta.reshape(seq_len, batch, self.num_value_heads), (1, 0, 2)
        )

        decay = self._compute_decay(raw_gate)
        # Normalize in FP32, then use the kernel's FP16/BF16 Q/K/V contract.
        # Gate processing stays outside the kernel so A_log/dt_bias retain
        # the existing FP32 autograd path, including safe-gate semantics.
        output, _ = npu_chunk_kda(
            ops.cast(q, v.dtype), ops.cast(k, v.dtype), v,
            decay, beta,
            scale=self.key_head_dim ** -0.5,
            chunk_size=self.chunk_size,
            layout="BSND",
            output_final_state=False,
            use_gate_in_kernel=False,
            disable_recompute=True,
            state_v_first=False,
        )
        output = self.out_norm(output) * ops.sigmoid(ops.cast(output_gate, ms.float32))
        output = ops.transpose(output, (1, 0, 2, 3)).reshape(seq_len, batch, self.v_dim)
        return self.out_proj(ops.cast(output, hidden_states.dtype))
