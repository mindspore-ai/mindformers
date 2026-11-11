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
"""Fused Kimi Delta Attention with fixed-length context-parallel training."""

from dataclasses import dataclass
from typing import Union

import mindspore as ms
from mindspore import Parameter, Tensor, mint, nn, ops
from hyper_parallel import DTensor
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard

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

    def __init__(self, channels, kernel_size, params_dtype=ms.float32, has_bias=False, channel_groups=1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.has_bias = has_bias
        self.channel_groups = channel_groups
        if channel_groups < 1 or channels % channel_groups:
            raise ValueError("channel_groups must be positive and divide channels.")
        weight_shape = (channels, 1, kernel_size)
        if channel_groups > 1:
            weight_shape = (channel_groups, channels // channel_groups, 1, kernel_size)
        self.weight = Parameter(
            mint.empty(weight_shape, dtype=params_dtype),
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

    @staticmethod
    def compute(x, weight, kernel_size, bias=None):
        """Apply depthwise causal convolution to complete local-head sequences."""
        weight = ops.cast(weight.reshape(-1, 1, kernel_size), x.dtype)
        if x.ndim != 3 or x.shape[-1] != weight.shape[0]:
            raise ValueError("ShortConv requires [B,T,C] input and one kernel per channel.")
        bias = ops.cast(bias.reshape(-1), x.dtype) if bias is not None else None
        conv_out = mint.nn.functional.conv1d(
            ops.transpose(x, (0, 2, 1)), weight, bias=bias, stride=1,
            padding=kernel_size - 1, dilation=1, groups=x.shape[-1],
        )
        conv_out = mint.nn.functional.silu(conv_out[..., :x.shape[1]])
        return ops.transpose(conv_out, (0, 2, 1))

    def construct(self, x):
        """Standalone path; KDA passes this cell's weight to the CP-wrapped core."""
        weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        bias = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
        return self.compute(x, weight, self.kernel_size, bias)


class KimiDeltaCoreAttention(nn.Cell):
    """Local convolution/normalization/recurrence boundary inside context parallel."""

    attention_variant = "kda"

    def __init__(self, config):
        super().__init__()
        self.chunk_size = config.kda_chunk_size
        self.scale = config.linear_key_head_dim ** -0.5
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.safe_gate, self.lower_bound = config.kda_safe_gate, config.kda_lower_bound

    @staticmethod
    def _l2_normalize(x):
        """Normalize each query/key head in FP32."""
        x = ops.cast(x, ms.float32)
        return x * ops.rsqrt(ops.sum(x * x, dim=-1, keepdim=True) + Tensor(1e-6, ms.float32))

    def construct(self, query, key, value, decay, beta, conv_weight, a_log, dt_bias):
        """Run local convolution and recurrence after CP redistribution."""
        layout = (query.device_mesh, query.placements) if isinstance(query, DTensor) else None
        query, key, value, decay, beta = (
            tensor.to_local() if isinstance(tensor, DTensor) else tensor
            for tensor in (query, key, value, decay, beta)
        )
        conv_weight = conv_weight.to_local() if isinstance(conv_weight, DTensor) else conv_weight
        a_log = a_log.to_local() if isinstance(a_log, DTensor) else a_log
        dt_bias = dt_bias.to_local() if isinstance(dt_bias, DTensor) else dt_bias
        conv_weight = conv_weight.reshape(3, -1, 1, self.conv_kernel_size)
        state_context = getattr(self, "_context_parallel_state", None)
        head_mesh = state_context.head_mesh if state_context is not None else (layout[0] if layout else None)
        if head_mesh is not None:
            # FSDP has gathered this TP rank's weights; select only Ulysses-local heads.
            conv_weight = distribute_tensor(conv_weight, head_mesh, (Shard(1),)).to_local()
            a_log = distribute_tensor(a_log, head_mesh, (Shard(0),)).to_local()
            dt_bias = distribute_tensor(dt_bias, head_mesh, (Shard(0),)).to_local()
        query, key, value = self._convolve(query, key, value, conv_weight)
        output, _ = npu_chunk_kda(
            ops.cast(query, value.dtype), ops.cast(key, value.dtype), value,
            decay, beta, scale=self.scale, chunk_size=self.chunk_size,
            layout="BSND", output_final_state=False, use_gate_in_kernel=True,
            safe_gate=self.safe_gate, lower_bound=self.lower_bound,
            a_log=a_log, dt_bias=dt_bias,
            disable_recompute=True, state_v_first=False,
            context_parallel=state_context,
        )
        return DTensor.from_local(output, *layout) if layout is not None else output

    def _convolve(self, query, key, value, conv_weight):
        """Convolve local inputs with causal context and normalize Q/K."""
        batch, tokens, heads, dim = query.shape
        widths = (heads * dim, heads * dim, heads * value.shape[-1])
        projected = ops.concat(tuple(tensor.reshape(batch, tokens, -1) for tensor in (query, key, value)), axis=-1)
        state_context = getattr(self, "_context_parallel_state", None)
        if state_context is not None:
            projected = state_context.sequence_halo(projected, self.conv_kernel_size - 1)
        convolved = ShortConv.compute(projected, conv_weight, self.conv_kernel_size)
        if state_context is not None:
            convolved = convolved[:, self.conv_kernel_size - 1:]
        query, key, value = mint.split(convolved, widths, dim=-1)
        query = self._l2_normalize(query.reshape(batch, tokens, heads, dim))
        key = self._l2_normalize(key.reshape(batch, tokens, heads, dim))
        value = value.reshape(batch, tokens, heads, -1)
        return query, key, value


class KimiDeltaAttention(nn.Cell):
    """Direct-projection KDA module for fixed-length training.

    This implementation includes the native MindSpore ShortConv fallback and
    intentionally excludes packed sequences and inference state/cache handling.
    Head-wise TP keeps complete local heads. Ulysses gathers complete sequences
    before convolution and normalization, then restores local sequences after the
    fused recurrence. State CP exchanges summaries and causal halos. The recurrence requires
    Hyper-Parallel's differentiable Chunk KDA and the FLA-NPU OPP, with
    128-dimensional heads, 64-token chunks, and BF16 safe-gate computation.
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
        self.core_attention = KimiDeltaCoreAttention(config)
        if npu_chunk_kda is None:
            raise ImportError(
                "Fused KDA requires Hyper-Parallel's npu_chunk_kda API and the matching FLA-NPU stage OPP."
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
            channel_groups=3,
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
            output_groups=5 if self.use_legacy_fused_projections else 3,
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

    def reshape_legacy_parameters(self, parameters):
        """Manually reshape layer-relative weights before load/TP; no loader/optimizer migration."""
        result = dict(parameters)
        for name in ("in_proj.weight", "in_proj.bias", "conv1d.weight"):
            if name not in result:
                continue
            module_name, parameter_name = name.split(".")
            target = getattr(getattr(self, module_name), parameter_name)
            if target is None:
                raise ValueError(f"Legacy KDA parameter {name} has no target.")
            source = result[name]
            if tuple(source.shape) == tuple(target.shape):
                continue
            flat_shape = (target.shape[0] * target.shape[1],) + tuple(target.shape[2:])
            if tuple(source.shape) != flat_shape:
                raise ValueError(f"Legacy KDA parameter {name} must have shape {flat_shape}, got {source.shape}.")
            value = source.reshape(target.shape)
            result[name] = (Parameter(value, name=source.name, requires_grad=source.requires_grad)
                            if isinstance(source, Parameter) else value)
        return result

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
        """The supported safe gate initializes ``A_log`` to zero."""
        self.A_log.zero_()

    def _reset_dt_bias(self):
        """Sample log-uniform steps in [1e-3,1e-1] using the parameter's DTensor RNG."""
        sample = self._uniform_parameter(self.dt_bias, -6.907755278982137, -2.302585092994046)
        local = self.dt_bias.to_local() if isinstance(self.dt_bias, DTensor) else self.dt_bias
        dt = mint.exp(sample)
        local.copy_(dt + mint.log(-mint.expm1(-dt)))

    @staticmethod
    def _uniform_parameter(parameter, low, high):
        """Sample these small vectors replicated, avoiding sub-four-element RNG shard offsets."""
        sample = mint.empty(parameter.shape, dtype=parameter.dtype)
        if isinstance(parameter, DTensor):
            sample = DTensor.from_local(sample, parameter.device_mesh, (Replicate(),) * parameter.device_mesh.ndim)
        sample.uniform_(low, high)
        if isinstance(parameter, DTensor):
            sample = sample.redistribute(parameter.device_mesh, parameter.placements).to_local()
        return sample

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
        """Run SBH training, optionally exchanging sequences and heads for CP."""
        del rotary_pos_emb, mscale, rotary_cos_sin
        if attention_mask is not None:
            raise NotImplementedError("KDA does not support attention_mask in the current SBH path.")
        if prefix_keys_values is not None:
            raise NotImplementedError("KDA inference prefix/KV cache is outside the minimal scope.")
        if actual_seq_len is not None:
            raise NotImplementedError("Packed or variable-length KDA input is outside the minimal scope.")

        projected = self.in_proj(hidden_states)
        seq_len, batch, _ = projected.shape
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

        def to_bthd(tensor, num_heads, head_dim):
            tensor = tensor.reshape(seq_len, batch, num_heads, head_dim)
            return ops.transpose(tensor, (1, 0, 2, 3))

        q = to_bthd(q, self.num_key_heads, self.key_head_dim)
        k = to_bthd(k, self.num_key_heads, self.key_head_dim)
        v = to_bthd(v, self.num_value_heads, self.value_head_dim)
        raw_gate = to_bthd(raw_gate, self.num_value_heads, self.key_head_dim)
        output_gate = to_bthd(output_gate, self.num_value_heads, self.value_head_dim)
        beta = self._compute_beta(beta)
        beta = ops.transpose(
            beta.reshape(seq_len, batch, self.num_value_heads), (1, 0, 2)
        )

        decay = ops.cast(raw_gate, ms.float32)
        # Ulysses encloses convolution, Q/K normalization and fused raw-gate processing.
        output = self.core_attention(
            q, k, v, decay, beta, self.conv1d.weight, self.A_log, self.dt_bias
        )
        output = self.out_norm(output) * ops.sigmoid(ops.cast(output_gate, ms.float32))
        output = ops.transpose(output, (1, 0, 2, 3)).reshape(seq_len, batch, self.v_dim)
        return self.out_proj(ops.cast(output, hidden_states.dtype))
