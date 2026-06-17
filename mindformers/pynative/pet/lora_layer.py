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
"""LoRA-wrapped Linear layer for PyNative mode.

``LinearWithLoRA`` subclasses :class:`mindformers.pynative.layers.linear.Linear`
so that ``isinstance(module, Linear)`` stays True (required by the PyNative tensor-
parallel sharding plan, which only recognises ``Linear``/``VocabEmbedding`` and
addresses the weight by the direct attribute name ``"weight"``).

The adapter computes ``y = x W^T + (alpha / r) * (x A^T) B^T`` where the base weight
``W`` is frozen and only ``A`` (``lora_a``, shape ``(r, in)``) and ``B`` (``lora_b``,
shape ``(out, r)``) are trained.

NOTE on initialisation: standard LoRA init — ``lora_a`` ~ N(0, sigma^2) random and
``lora_b`` = 0, so ``Delta W = 0`` at step 0 (an exact identity start). This is the default
(``lora_b_std`` = 0.0) and it trains correctly in PyNative once a pretrained checkpoint is
loaded (the normal fine-tuning case): with a non-zero base weight, B=0 still yields a non-zero
adapter gradient (verified — and graph and pynative agree bit-for-bit).

The one degenerate case to be aware of: if the *base* weight ``W`` is *also* zero (no
checkpoint loaded / a smoke test on an empty model), this Linear's total output
``base + adapter`` is exactly zero, and DeepSeek's downstream SwiGLU gate ``silu(x0) * x1`` has
an exactly-zero Jacobian at a zero input (``silu(0) = 0`` and ``silu'(0) * 0 = 0``; ReLU shares
this — plain silu/gelu/identity do not), so the adapter receives a *mathematically-correct*
zero gradient and never trains. This is gate calculus, **not** a framework (MindSpore autograd /
hyper_parallel) bug; loading a checkpoint (``W`` != 0) removes the condition. If you must train
on an un-checkpointed model, pass a tiny ``lora_b_std`` > 0 to break the exact zero.

Parameter naming ``lora_a``/``lora_b`` matches the static-graph implementation so the
``transform_ckpt_lora.py`` merge tool stays compatible.
"""
__all__ = ["LinearWithLoRA"]

import mindspore as ms
from mindspore import mint
from mindspore.common.parameter import Parameter

from mindformers.pynative.layers.linear import Linear
from mindformers.pynative.layers.dropout import Dropout
from mindformers.pynative.layers.identity_op import IdentityOp


class LinearWithLoRA(Linear):
    """A :class:`Linear` augmented with a low-rank adapter.

    Args:
        All positional/keyword args of :class:`Linear`, plus:
        lora_rank (int): Rank ``r`` of the adapter. Default: 8.
        lora_alpha (int): Scaling numerator; effective scale is ``alpha / r``. Default: 16.
        lora_dropout (float): Dropout probability on the adapter-branch input. Default: 0.0.
        lora_a_std (float): Std of the normal init for ``lora_a``. Default: 0.01.
        lora_b_std (float): Std of the normal init for ``lora_b``. Default: 0.0 — exact B=0,
            the standard LoRA identity start. See the module docstring for the un-checkpointed
            caveat (pass a small value only if training without a checkpoint).
    """

    def __init__(self,
                 *args,
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.0,
                 lora_a_std: float = 0.01,
                 lora_b_std: float = 0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = float(lora_alpha) / float(lora_rank)
        self.lora_a_std = lora_a_std
        self.lora_b_std = lora_b_std
        self.lora_dropout = Dropout(lora_dropout) if lora_dropout > 0.0 else IdentityOp()

        # Stored like the base weight as (out, in) and transposed at runtime:
        #   lora_a: (r, in)   -> A,  lora_b: (out, r) -> B
        self.lora_a = Parameter(
            mint.empty((lora_rank, self.input_size), dtype=self.params_dtype), name="lora_a")
        self.lora_b = Parameter(
            mint.empty((self.output_size, lora_rank), dtype=self.params_dtype), name="lora_b")

    def construct(self, input_: ms.Tensor, weight: ms.Tensor = None) -> ms.Tensor:
        """Forward: base linear plus the scaled low-rank adapter."""
        base_output = super().construct(input_, weight)

        ori_dtype = input_.dtype
        x = self.lora_dropout(input_)
        x = self.cast(x, self.compute_dtype)

        lora_a = self.cast(self.lora_a, self.compute_dtype)
        lora_b = self.cast(self.lora_b, self.compute_dtype)
        # (..., in) @ (in, r) -> (..., r) -> (..., out). Under tensor parallelism with a row-wise
        # base, ``tmp`` is a Partial DTensor (the first matmul contracts over the TP-sharded
        # in-dim); hyper_parallel (>= PR #867) propagates that Partial status through the second
        # matmul so the adapter gradient stays correct — no manual reduction needed here.
        tmp = self.matmul(x, self.transpose(lora_a, 1, 0))
        lora_out = self.matmul(tmp, self.transpose(lora_b, 1, 0))
        lora_out = self.cast(lora_out * self.scaling, ori_dtype)

        return self.cast(base_output, ori_dtype) + lora_out

    def reset_parameter(self) -> None:
        """Initialise adapter only (base weight is loaded from the pretrained ckpt).

        Standard LoRA init: ``lora_a`` ~ N(0, lora_a_std^2) random and ``lora_b`` = 0 (the
        default ``lora_b_std`` = 0.0), so ``Delta W = 0`` at step 0 — an exact identity start.
        B=0 trains fine once a checkpoint is loaded; see the module docstring for the
        un-checkpointed caveat. Uses in-place fills, matching the delayed-init idiom invoked by
        ``init_states`` after ``to_empty()``.
        """
        self.lora_a.normal_(mean=0.0, std=self.lora_a_std)
        if self.lora_b_std > 0.0:
            self.lora_b.normal_(mean=0.0, std=self.lora_b_std)
        else:
            self.lora_b.zero_()

    @classmethod
    def from_base(cls,
                  base: Linear,
                  lora_rank: int = 8,
                  lora_alpha: int = 16,
                  lora_dropout: float = 0.0,
                  lora_a_std: float = 0.01,
                  lora_b_std: float = 0.0) -> "LinearWithLoRA":
        """Build a ``LinearWithLoRA`` that reuses ``base``'s weight/bias Parameters.

        The base ``weight``/``bias`` Parameter objects (and names) are reused verbatim so
        the existing checkpoint mapping and parallelize plan keep working. Construct under
        a meta-device context so the new ``lora_a``/``lora_b`` are meta tensors, consistent
        with the meta base model before ``to_empty()``.
        """
        with ms.DeviceCtx("meta"):
            obj = cls(
                base.input_size,
                base.output_size,
                compute_dtype=base.compute_dtype,
                params_dtype=base.params_dtype,
                bias=base.has_bias,
                skip_weight_param_allocation=base.skip_weight_param_allocation,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_a_std=lora_a_std,
                lora_b_std=lora_b_std,
            )
        # Reuse the original base parameters (identity + names preserved).
        obj.weight = base.weight
        obj.has_bias = base.has_bias
        if base.has_bias:
            obj.bias = base.bias
            obj.bias_init = base.bias_init
        obj.init_method = base.init_method
        return obj
