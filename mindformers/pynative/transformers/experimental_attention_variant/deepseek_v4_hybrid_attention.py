# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2026 Huawei Technologies Co., Ltd
#
# This file is derived from Megatron-LM and adapted for MindSpore.
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
"""DSv4 hybrid top-level attention"""
from dataclasses import dataclass
from typing import Union

from mindspore import nn, Tensor, mint, ops
from mindspore import dtype as mstype
from mindspore.common.parameter import Parameter

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.pynative.base_models.common.embeddings.rope_utils import ApplyRotaryPosEmb
from mindformers.pynative.base_models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from mindformers.pynative.base_models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
)
from mindformers.pynative.transformers.multi_latent_attention import MultiLatentAttention


@dataclass
class DSv4HybridSelfAttentionSubmodules:
    """Submodules for the DSv4HybridAttention layer."""

    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None
    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None


class DSv4HybridSelfAttention(MultiLatentAttention):
    """DeepSeek-V4 hybrid attention top-level entry (replaces ``MLASelfAttention``)."""

    def __init__(
            self,
            config: MLATransformerConfig,
            submodules: DSv4HybridSelfAttentionSubmodules,
            layer_number: int,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type="self",
        )

        # DSv4 invariant: q_head_dim collapses to v_head_dim.
        self.q_head_dim = config.v_head_dim

        self.softmax_scale = config.v_head_dim ** -0.5

        if config.csa_compress_ratios is not None:
            self.compress_ratio = config.csa_compress_ratios[layer_number]
        else:
            self.compress_ratio = 0

        # The compressed-KV branch (ratio > 1) uses a different rotary base than the window branch (rotary_base).
        rope_base = config.csa_compress_rotary_base if self.compress_ratio > 1 else config.rotary_base
        self.rotary_pos_emb = self._build_rotary_pos_emb(config, rope_base)
        self.apply_rotary_emb = ApplyRotaryPosEmb(config)

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_index,
            softmax_scale=self.softmax_scale,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=self.compress_ratio,
        )

        self.linear_q_down_proj = build_module(
            submodules.linear_q_down_proj,
            input_size=config.hidden_size,
            output_size=config.q_lora_rank,
            params_dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            init_method=config.init_method,
            bias=False,
        )

        self.q_layernorm = build_module(
            submodules.q_layernorm,
            dim=config.q_lora_rank,
            eps=config.layernorm_epsilon,
            compute_dtype=config.layernorm_compute_dtype,
        )

        self.linear_q_up_proj = build_module(
            submodules.linear_q_up_proj,
            input_size=config.q_lora_rank,
            output_size=config.num_attention_heads * self.q_head_dim,
            params_dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            init_method=config.init_method,
            bias=False,
        )

        self.linear_kv_proj = build_module(
            submodules.linear_kv_proj,
            input_size=config.hidden_size,
            output_size=config.v_head_dim,
            params_dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            init_method=config.init_method,
            bias=False,
        )

        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            dim=config.v_head_dim,
            eps=config.layernorm_epsilon,
            compute_dtype=config.layernorm_compute_dtype,
        )

        o_groups = config.o_groups
        o_lora_rank = config.o_lora_rank
        o_chunk = self.query_projection_size // o_groups
        self.linear_o_group_proj = Parameter(
            config.init_method((o_groups * o_lora_rank, o_chunk)),
            name="linear_o_group_proj",
        )

        self.linear_proj = build_module(
            submodules.linear_proj,
            input_size=o_groups * o_lora_rank,
            output_size=config.hidden_size,
            params_dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
        )

        self.split = mint.split
        self.cat = mint.cat
        self.unsqueeze = mint.unsqueeze
        self.bmm = mint.bmm
        self.rms_norm = ops.rms_norm
        self.q_rms_gamma = Parameter(
            mint.ones((self.q_head_dim,), dtype=mstype.float32),
            name="q_rms_gamma", requires_grad=False,
        )

    def reset_parameter(self):
        """Re-init ``linear_o_group_proj`` in-place after meta-device ``to_empty()``."""
        self.linear_o_group_proj.normal_(mean=0.0, std=0.01)
        self.q_rms_gamma.fill_(1.0)

    @staticmethod
    def _build_rotary_pos_emb(config: MLATransformerConfig, rope_base: float):
        """Build the per-layer RoPE module."""
        if config.rope_type == "rope":
            return RotaryEmbedding(
                config.qk_pos_emb_head_dim,
                rotary_percent=config.rotary_percent,
                rotary_interleaved=config.rotary_interleaved,
                seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
                rotary_base=rope_base,
                rope_scaling=config.use_rope_scaling,
            )
        if config.rope_type == "yarn":
            return YarnRotaryEmbedding(
                config.qk_pos_emb_head_dim,
                rotary_percent=config.rotary_percent,
                rotary_interleaved=config.rotary_interleaved,
                seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
                rotary_base=rope_base,
                scaling_factor=config.rotary_scaling_factor,
                original_max_position_embeddings=config.max_position_embeddings,
                beta_fast=config.beta_fast,
                beta_slow=config.beta_slow,
                mscale=config.mscale,
                mscale_all_dim=config.mscale_all_dim,
            )
        raise ValueError(
            f"Unsupported RoPE type: {config.rope_type}, supported types are "
            "'rope' and 'yarn'"
        )

    def _apply_forward_rope(self, t: Tensor, freqs: Tensor, mscale: float = 1.0) -> Tensor:
        """Forward-RoPE the trailing ``qk_pos_emb_head_dim`` lanes of ``t``."""
        pos_dim = self.config.qk_pos_emb_head_dim
        nope_dim = int(t.shape[-1]) - pos_dim
        t_nope, t_pe = self.split(t, [nope_dim, pos_dim], dim=-1)
        t_pe = self.apply_rotary_emb(
            t_pe,
            freqs,
            mscale,
            rotary_interleaved=self.config.rotary_interleaved,
            multi_latent_attention=self.config.multi_latent_attention,
            mla_output_remove_interleaving=True,
        )
        return self.cat([t_nope, t_pe], dim=-1)

    def _apply_inverse_rope(self, core_attn_out: Tensor, sq: int, bsz: int) -> Tensor:
        """Inverse-RoPE the trailing ``qk_pos_emb_head_dim`` lanes of each head."""
        if self.rotary_pos_emb is None:
            return core_attn_out
        n_heads = self.config.num_attention_heads
        pos_dim = self.config.qk_pos_emb_head_dim
        v_head_dim = self.config.v_head_dim
        nope_dim = v_head_dim - pos_dim
        # [sq, b, n_heads * v_head_dim] -> [sq, b, n_heads, v_head_dim].
        out_4d = self.reshape(core_attn_out, (sq, bsz, n_heads, v_head_dim))
        content_part, rot_part = self.split(out_4d, [nope_dim, pos_dim], dim=-1)
        # rotary_pos_emb(seq_len) -> (freqs, mscale); force mscale = 1.0.
        freqs, _ = self.rotary_pos_emb(sq)
        rot_part = self.apply_rotary_emb(
            rot_part,
            freqs,
            1.0,
            rotary_interleaved=self.config.rotary_interleaved,
            multi_latent_attention=self.config.multi_latent_attention,
            inverse=True,
            mla_output_remove_interleaving=True,
        )
        out_4d = self.cat([content_part, rot_part], dim=-1)
        return self.reshape(out_4d, (sq, bsz, n_heads * v_head_dim))

    # pylint: disable=arguments-differ,unused-argument
    def construct(self, x: Tensor, attention_mask=None, rotary_pos_emb=None,
                  prefix_keys_values=None, actual_seq_len=None, mscale=1.0) -> Tensor:
        """DeepSeek-V4 hybrid attention forward."""
        if prefix_keys_values is not None:
            raise NotImplementedError("prefix_keys_values is not supported for now.")

        ori_dtype = x.dtype
        sq, bsz, _ = self.shape(x)

        # ---- Q low-rank down -> norm -> up ---------------------------
        # q_compressed: [sq, b, q_lora_rank]
        q_compressed = self.linear_q_down_proj(x)
        q_compressed = self.q_layernorm(q_compressed)
        # q: [sq, b, num_attention_heads * v_head_dim]
        q = self.linear_q_up_proj(q_compressed)
        # q: [sq, b, num_attention_heads, v_head_dim]
        q = self.reshape(q, (sq, bsz, self.config.num_attention_heads, self.q_head_dim))
        # Q-head RMS normalization, aligns with Megatron
        eps = self.config.layernorm_epsilon
        q = self.cast(
            self.rms_norm(self.cast(q, mstype.float32), self.q_rms_gamma, eps)[0],
            self.compute_dtype,
        )

        # ---- KV shared single head -----------------------------------
        # kv: [sq, b, v_head_dim]
        kv = self.linear_kv_proj(x)
        kv = self.kv_layernorm(kv)
        # key / value: [sq, b, 1, v_head_dim] (shared single head).
        kv_4d = self.unsqueeze(kv, -2)
        kv_4d = self.cast(kv_4d, self.compute_dtype)

        # ---- Main Q / K pe-lane RoPE ---------------------------------
        if self.rotary_pos_emb is not None:
            freqs, _ = self.rotary_pos_emb(sq)
            q = self._apply_forward_rope(q, freqs, 1.0)
            kv_4d = self._apply_forward_rope(kv_4d, freqs, 1.0)
        key = kv_4d

        # ---- Core attention ------------------------------------------
        # core_out: [sq, b, num_attention_heads * v_head_dim]
        core_out = self.core_attention(
            q, key, x, q_compressed,
            actual_seq_len=actual_seq_len
        )

        # ---- Inverse-RoPE on the post-core-attention output ----------
        core_out = self._apply_inverse_rope(core_out, sq, bsz)

        # ---- Grouped wo_a --------------------------------------------
        o_groups = self.config.o_groups
        o_lora_rank = self.config.o_lora_rank
        d = self.query_projection_size // o_groups
        # core_grouped: [sq, b, o_groups, d]; wo_a_3d: [o_groups, o_lora_rank, d]
        core_grouped = self.reshape(core_out, (sq, bsz, o_groups, d))
        wo_a_3d = self.reshape(self.linear_o_group_proj, (o_groups, o_lora_rank, d))
        # einsum("...gd,grd->...gr") ==> [g, sq*b, d] @ [g, d, r] -> [g, sq*b, r] -> [sq, b, g, r].
        cg = self.permute(self.reshape(core_grouped, (sq * bsz, o_groups, d)), (1, 0, 2))
        wo = self.permute(wo_a_3d, (0, 2, 1))
        inter = self.bmm(self.cast(cg, mstype.float32), self.cast(wo, mstype.float32))
        # [g, sq*b, r] -> [sq*b, g, r] -> [sq, b, o_groups*o_lora_rank]
        intermediate = self.reshape(
            self.permute(inter, (1, 0, 2)), (sq, bsz, o_groups * o_lora_rank)
        )

        # ---- Final output linear -------------------------------------
        output = self.linear_proj(intermediate)
        output = self.cast(output, ori_dtype)
        return output
