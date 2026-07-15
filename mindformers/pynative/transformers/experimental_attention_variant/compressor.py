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
"""CSA Compressor"""
from dataclasses import dataclass
from typing import Union, Optional

import mindspore.common.dtype as mstype
from mindspore import nn, ops, Tensor, mint, Parameter

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.pynative.layers.identity_op import IdentityOp
from mindformers.pynative.transformers.experimental_attention_variant.utils import Hadamard
from mindformers.pynative.base_models.common.embeddings.rope_utils import ApplyRotaryPosEmb

@dataclass
class CompressorSubmodules:
    """Submodule specs for CSA and HCA Compressor."""

    linear_wkv: Union[ModuleSpec, type] = None
    linear_wgate: Union[ModuleSpec, type] = None
    norm: Union[ModuleSpec, type] = None


class Compressor(nn.Cell):
    """Gated pooling compressor for DSv4 hybrid CSA / HCA sparse attention.

    Compresses a sequence of ``sq`` tokens into ``sq // compress_ratio``
    pooled KV tokens. Aligns 1:1 with Megatron's ``csa.py:Compressor``.

    Two ratio modes are supported by RFC §3.4.2.2:

      - ``compress_ratio == 4``: overlapping windows (``coff = 2``); each
        compressed position pools ``2 * ratio`` source tokens, half from the
        current group and half from the previous group.
      - ``compress_ratio == 128``: non-overlapping windows (``coff = 1``);
        each compressed position pools ``ratio`` source tokens.

    A learnable Absolute Position Embedding ``ape`` of shape
    ``(ratio, coff * head_dim)`` (kept in fp32) is added to the gate logits
    before softmax.

    Args:
        config: MLATransformerConfig. Used for ``hidden_size``,
            ``init_method``, ``layernorm_epsilon`` and dtype settings.
        submodules: CompressorSubmodules specifying ``linear_wkv``,
            ``linear_wgate`` and ``norm``.
        compress_ratio: Compression ratio (RFC currently restricts to
            ``{4, 128}``).
        head_dim: Compressed KV head dimension.
        rotate: Whether to Hadamard-rotate the output along ``head_dim``
            (used by the indexer branch). Default False.
        rotary_pos_emb: Optional RoPE module applied on the pooled KV.
            Not invoked when ``None`` (test path / no-rope mode).

    Inputs:
        x (Tensor[sq, b, hidden_size])

    Outputs:
        Tensor[sq // compress_ratio, b, head_dim] when ``sq >= compress_ratio``;
        ``None`` otherwise (per RFC).
    """

    def __init__(
            self,
            config: MLATransformerConfig,
            submodules: CompressorSubmodules,
            compress_ratio: int,
            head_dim: int,
            rotate: bool = False,
            rotary_pos_emb: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = compress_ratio == 4
        self.coff = 1 + int(self.overlap)
        self.rotate = rotate
        self.rotary_pos_emb = rotary_pos_emb
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim

        proj_out_dim = self.coff * head_dim

        self.linear_wkv = build_module(
            submodules.linear_wkv,
            input_size=config.hidden_size,
            output_size=proj_out_dim,
            params_dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            init_method=config.init_method,
            bias=False,
        )

        self.linear_wgate = build_module(
            submodules.linear_wgate,
            input_size=config.hidden_size,
            output_size=proj_out_dim,
            params_dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            init_method=config.init_method,
            bias=False,
        )

        self.ape = Parameter(
            mint.empty((compress_ratio, proj_out_dim), dtype=config.params_dtype),
            name="ape",
        )

        self.norm = build_module(
            submodules.norm,
            dim=head_dim,
            eps=config.layernorm_epsilon,
            compute_dtype=config.layernorm_compute_dtype,
        )

        self.hadamard = Hadamard(head_dim) if rotate else IdentityOp()

        # RoPE applicator for the compressed-KV pe lane (constructed in __init__
        # per the fine-grained-recompute convention; called in ``_apply_rope``).
        self.apply_rope = ApplyRotaryPosEmb(config)

        # Alias the non-trivial mint ops used in construct/forward per the
        # fine-grained-recompute convention (RFC §3.1 #9).
        self.reshape = mint.reshape
        self.softmax = mint.softmax
        self.unsqueeze = mint.unsqueeze
        self.chunk = mint.chunk
        self.split = mint.split
        self.roll = mint.roll
        self.cat = mint.cat
        self.squeeze = mint.squeeze
        self.permute = mint.permute

    def reset_parameter(self):
        """Re-init ``ape`` in-place after meta-device ``to_empty()`` (B2).

        Uses in-place ``normal_`` (the Linear / VocabEmbedding reset idiom) so the DTensor sharding is preserved.
        ``set_data`` with a full-shaped tensor would break FSDP's sharded layout
        (the ``reset_sharded_param`` shape check).
        """
        self.ape.normal_(mean=0.0, std=0.01)

    def _overlap_transform(self, tensor: Tensor, fill_value: float = 0.0) -> Tensor:
        """Apply the overlapping window transform used when ``compress_ratio == 4``.

        Input  shape: ``[n_groups, ratio, b, coff * head_dim]`` (coff == 2).
        Output shape: ``[n_groups, 2 * ratio, b, head_dim]``.

        The first ``ratio`` rows of each group are filled from the *previous*
        group's left-half (``[:, :, :, :head_dim]``); the last ``ratio`` rows
        of each group are filled from the *current* group's right-half
        (``[:, :, :, head_dim:]``). The very first group has no predecessor
        so its first ``ratio`` rows hold ``fill_value`` (0 for KV;
        ``-inf`` for the gate so softmax masks them out).
        """
        tensor_prev, tensor_next = self.chunk(tensor, 2, dim=-1)
        tensor_prev = self.roll(tensor_prev, 1, 0)
        tensor_prev[0] = fill_value
        out = self.cat([tensor_prev, tensor_next], 1)
        return out

    def construct(self, x: Tensor) -> Optional[Tensor]:
        """Compress the hidden states into a shorter KV sequence.

        Args:
            x: Tensor of shape ``[sq, b, hidden_size]``.

        Returns:
            ``[sq // compress_ratio, b, head_dim]`` if ``sq >= compress_ratio``,
            else ``None``.
        """
        sq, b, _ = x.shape
        ratio = self.compress_ratio

        if sq < ratio:
            return None

        kv = self.linear_wkv(x)
        score = self.linear_wgate(x)

        cutoff = (sq // ratio) * ratio
        if cutoff < sq:
            kv = kv[:cutoff]
            score = score[:cutoff]

        n_compressed = cutoff // ratio

        kv = self.reshape(kv, (n_compressed, ratio, b, -1))
        score = self.reshape(score, (n_compressed, ratio, b, -1))

        # Upcast to fp32 so the addition with fp32 ape and the subsequent
        # softmax stay numerically stable (matches Megatron implicit upcast).
        ape = self.ape.to_local() if hasattr(self.ape, "to_local") else self.ape
        score_f32 = score.astype(mstype.float32) + self.reshape(ape, (1, ratio, 1, -1))

        if self.overlap:
            kv = self._overlap_transform(kv, fill_value=0.0)
            score_f32 = self._overlap_transform(score_f32, fill_value=float("-inf"))

        weights = self.softmax(score_f32, dim=1)
        pooled = (kv.astype(mstype.float32) * weights).sum(dim=1)

        pooled = self.norm(pooled.astype(x.dtype))

        if self.rotary_pos_emb is not None:
            pooled = self._apply_rope(pooled)

        pooled = self.hadamard(pooled)
        pooled = self.unsqueeze(pooled, -2)
        return pooled

    def _apply_rope(self, kv: Tensor) -> Tensor:
        """Apply RoPE to the trailing ``qk_pos_emb_head_dim`` lanes of ``kv``."""
        n_compressed, _, d = kv.shape
        total_seq_len = n_compressed * self.compress_ratio
        freqs, _ = self.rotary_pos_emb(total_seq_len)
        if self.compress_ratio > 1:
            freqs = freqs[:total_seq_len:self.compress_ratio][:n_compressed]
        # Add a singleton head dim so apply_rotary_pos_emb sees [s, b, 1, hd].
        kv_4d = self.unsqueeze(kv, -2)
        kv_nope, kv_pe = self.split(kv_4d, [d - self.qk_pos_emb_head_dim, self.qk_pos_emb_head_dim], dim=-1)
        kv_pe = self.apply_rope(
            kv_pe, freqs, 1.0,
            rotary_interleaved=self.config.rotary_interleaved,
            multi_latent_attention=True,
            mla_output_remove_interleaving=True,
        )
        out = self.cat([kv_nope, kv_pe], dim=-1)
        return self.squeeze(out, -2)
