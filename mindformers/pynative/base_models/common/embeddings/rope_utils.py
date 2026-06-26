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
"""
Apply rotary position embedding for transformer.
"""
__all__ = [
    "ApplyRotaryPosEmb"
]

from mindspore import nn, Tensor, ops, mint
from mindformers.parallel_core.transformer_config import TransformerConfig, MLATransformerConfig
from mindformers.pynative.base_models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from mindformers.pynative.base_models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding


def _get_rope(config: MLATransformerConfig):
    return RotaryEmbedding(
        config.qk_pos_emb_head_dim,
        rotary_percent=config.rotary_percent,
        rotary_interleaved=config.rotary_interleaved,
        seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
        rotary_base=config.rotary_base,
        rope_scaling=config.use_rope_scaling
    )


def _get_yarn(config: MLATransformerConfig):
    return YarnRotaryEmbedding(
        config.qk_pos_emb_head_dim,
        rotary_percent=config.rotary_percent,
        rotary_interleaved=config.rotary_interleaved,
        seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
        rotary_base=config.rotary_base,
        scaling_factor=config.rotary_scaling_factor,
        original_max_position_embeddings=config.max_position_embeddings,
        beta_fast=config.beta_fast,
        beta_slow=config.beta_slow,
        mscale=config.mscale,
        mscale_all_dim=config.mscale_all_dim,
    )


ROPE_FUNCTIONS = {
    'rope': _get_rope,
    'yarn': _get_yarn,
}


class ApplyRotaryPosEmb(nn.Cell):
    """Apply rotary positional embedding to input tensor T.

    Args:
        config (TransformerConfig): The transformer configuration
    """

    def __init__(self, config: TransformerConfig,):
        super().__init__()
        self.apply_rope_fusion = config.apply_rope_fusion
        self.rotary_dtype = config.rotary_dtype

        self.add = mint.add
        self.mul = mint.mul
        self.mul_mscale = mint.mul
        self.cos = mint.cos
        self.sin = mint.sin
        self.neg = mint.neg
        self.split = mint.split
        self.cat = mint.cat
        self.stack = mint.stack
        self.reshape = mint.reshape
        self.narrow = mint.narrow
        self.cast = ops.cast

        if self.apply_rope_fusion:
            self.rope = ops.auto_generate.gen_ops_prim.RotaryPositionEmbedding()

    def construct(self,
                  t: Tensor,
                  freqs: Tensor,
                  mscale: float = 1.0,
                  rotary_interleaved: bool = False,
                  multi_latent_attention: bool = False,
                  inverse: bool = False,
                  mla_output_remove_interleaving: bool = False) -> Tensor:
        """Apply rotary position embedding to input tensor.

        Args:
            t (Tensor): Input tensor is of shape [seq_length, ... , dim]
            freqs (Tensor): Rotary position embedding frequencies of shape [seq_length, ... , dim].
            mscale (float, optional): Magnitude scaling applied to the rotary angles before
                computing cos/sin (yarn interface). Default: 1.0
            rotary_interleaved (bool, optional): Whether to use interleaved rotary position embedding. Default: False
            multi_latent_attention (bool, optional): Whether to use multi latent attention. Default: False
            inverse (bool, optional): Whether to apply the inverse rotation (un-rotate).
                The forward rotation is ``t * cos + rotate_half(t) * sin``; the inverse negates the sine term
                (``t * cos - rotate_half(t) * sin``), i.e. rotation by ``-theta``. Used by DSv4
                hybrid attention's inverse-RoPE on the post-core-attention output. Default: False
            mla_output_remove_interleaving (bool, optional):
                Whether to re-interleave the rotated lanes back to the original (interleaved) layout after rotation,
                undoing the ``multi_latent_attention`` de-interleave. Default: False

        Returns:
            Tensor: Output tensor after applying rotary position embedding.
        """
        origin_dtype = t.dtype
        seq_len, bs, n_heads, head_dim = t.shape
        m_scale = mscale
        rot_dim = freqs.shape[-1]
        if head_dim == rot_dim:
            t_not_rotary = None
        else:
            t_rotary = self.narrow(t, dim=-1, start=0, length=rot_dim)
            t_not_rotary = self.narrow(t, dim=-1, start=rot_dim, length=head_dim - rot_dim)
            t = t_rotary

        deinterleave_mla = multi_latent_attention and not rotary_interleaved
        fused_interleaved_mla = (
            self.apply_rope_fusion and deinterleave_mla and not inverse and not mla_output_remove_interleaving
        )
        if deinterleave_mla and not fused_interleaved_mla:
            t_reshaped = self.reshape(t, (seq_len, bs, n_heads, rot_dim // 2, 2))
            t_1 = t_reshaped[..., 0]
            t_2 = t_reshaped[..., 1]
            t = self.cat((t_1, t_2), dim=-1)

        t = self.cast(t, self.rotary_dtype)
        if fused_interleaved_mla:
            freqs = self._to_interleaved_freqs(freqs)
        cos_ = self.cast(self.cos(self.mul_mscale(freqs, m_scale)), self.rotary_dtype)
        sin_ = self.cast(self.sin(self.mul_mscale(freqs, m_scale)), self.rotary_dtype)
        # Negating sin implements the conjugate (inverse) rotation by -theta used for the MLA
        # output inverse-RoPE; it composes with the forward RoPE (same cos, +sin) to the identity.
        if inverse:
            sin_ = self.neg(sin_)

        if self.apply_rope_fusion and not inverse:
            mode = 1 if rotary_interleaved or fused_interleaved_mla else 0
            output = self.rope(t, cos_, sin_, mode)
        else:
            t_rot = self._rotate_half(t, rotary_interleaved)
            output = self.add(self.mul(t, cos_), self.mul(t_rot, sin_))

        # Why this re-interleave branch exists:
        # DSv4 applies rope on V and O, so we need to uninterleave the tensor.
        # The existing MLA code is safe because the dot product is permutation-invariant.
        if deinterleave_mla and mla_output_remove_interleaving:
            half = output.shape[-1] // 2
            o_1 = self.narrow(output, dim=-1, start=0, length=half)
            o_2 = self.narrow(output, dim=-1, start=half, length=half)
            output = self.reshape(
                self.stack((o_1, o_2), dim=-1),
                (seq_len, bs, n_heads, -1),
            )

        if t_not_rotary is not None:
            output = self.cat((output, t_not_rotary), dim=-1)
        return self.cast(output, origin_dtype)

    def _to_interleaved_freqs(self, freqs: Tensor) -> Tensor:
        """Convert NeoX-style duplicated frequencies to GPT-J interleaved layout."""
        rot_dim = freqs.shape[-1]
        # NeoX freqs are ``cat(freqs_half, freqs_half)`` so the leading half holds the
        # unique angles. Interleave-duplicate each angle (``[a, b] -> [a, a, b, b]``)
        # using only ``split``/``reshape``/``cat``, all of which have registered
        # distributed (DTensor) layout-inference implementations (``narrow``, ``stack``
        # and ``repeat_interleave`` do not).
        freqs_half = self.split(freqs, rot_dim // 2, dim=-1)[0]
        lead_shape = freqs_half.shape[:-1]
        expanded = self.reshape(freqs_half, (*lead_shape, rot_dim // 2, 1))
        duplicated = self.cat((expanded, expanded), dim=-1)
        return self.reshape(duplicated, (*lead_shape, rot_dim))

    def _rotate_half(self, t: Tensor, rotary_interleaved: bool = False) -> Tensor:
        """Rotates half of the input tensor for rotary position embeddings.

        Args:
            t (Tensor): Input tensor of shape (seq_len, bs, n_heads, head_dim)
            rotary_interleaved (bool, optional): If True, processes interleaved features. Defaults to False.

        Returns:
            Rotated tensor with same shape as input.
        """
        seq_len, bs, n_heads, head_dim = t.shape
        if rotary_interleaved:
            t_reshaped = self.reshape(t, (seq_len, bs, n_heads, head_dim // 2, 2))
            t_1 = t_reshaped[..., 0]
            t_2 = t_reshaped[..., 1]
            t_rot = self.reshape(self.stack((self.neg(t_2), t_1), dim=-1), (seq_len, bs, n_heads, -1))
        else:
            split_size = t.shape[-1] // 2
            t_1, t_2 = self.split(t, split_size_or_sections=split_size, dim=-1)
            t_rot = self.cat((self.neg(t_2), t_1), dim=-1)
        return t_rot
