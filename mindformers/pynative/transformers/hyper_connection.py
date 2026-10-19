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

import importlib
import math

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn, Parameter, Tensor, mint, ops
from hyper_parallel import DTensor

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.init_method import init_method_zero
from mindformers.pynative.layers.linear import Linear


def _load_fused_mhc_kernel(name):
    """Return the named fused mHC kernel, or None when this hyper-parallel build lacks it.

    The kernels are imported one at a time on purpose. They ship independently -- the main
    repo's r1.0.0 line carries npu_mhc_post and npu_mhc_pre_sinkhorn but not npu_mhc_pre_cmhc
    -- and a single combined import would discard every kernel present just because one is
    missing, silently sending every fused path back to the small-op one.
    """
    try:
        module = importlib.import_module("hyper_parallel.custom_ops.experimental")
    except ImportError:
        return None
    return getattr(module, name, None)


npu_mhc_pre_sinkhorn = _load_fused_mhc_kernel("npu_mhc_pre_sinkhorn")
npu_mhc_post = _load_fused_mhc_kernel("npu_mhc_post")
npu_mhc_pre_clamp_sinkhorn = _load_fused_mhc_kernel("npu_mhc_pre_clamp_sinkhorn")


def _require_fused_mhc_kernel(kernel, name, switch):
    """Raise a legible error when a fused mHC path is selected without its kernel."""
    if kernel is None:
        raise ImportError(
            f"{switch} requires the {name} kernel, which this hyper-parallel build does "
            f"not provide. Install a build that carries it, or turn {switch} off to use "
            f"the small-op mHC path instead."
        )

def _to_local(tensor):
    """Return the local value of a distributed mHC parameter."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class SinkhornKnopp(nn.Cell):
    """Doubly-stochastic projection via alternating row/column normalisation.

    Matches the MindSpeed DSV4 reference: row-softmax plus epsilon, one column
    normalisation, then ``iterations - 1`` alternating row/column updates.
    """

    _EPS = 1e-6

    def __init__(self, iterations: int, megatron_compatibility: bool = False):
        super().__init__()
        self.iterations = iterations
        self.megatron_compatibility = megatron_compatibility
        self.eps = self._EPS
        self.softmax = mint.softmax
        self.max = mint.max
        self.exp = mint.exp
        self.sum = mint.sum
        self.add = mint.add
        self.clamp = mint.clamp
        self.div = mint.div

    def construct(self, h):
        """Project raw logits onto the Birkhoff polytope.

        Args:
            h: [s, b, n, n] raw logits, float32

        Returns:
            Doubly-stochastic matrix [s, b, n, n], float32.
        """
        if not self.megatron_compatibility:
            m = self.add(self.softmax(h, dim=-1), self.eps)
            m = self.div(m, self.add(self.sum(m, dim=2, keepdim=True), self.eps))
            for _ in range(self.iterations - 1):
                # Row normalise along the last axis.
                m = self.div(m, self.add(self.sum(m, dim=3, keepdim=True), self.eps))
                # Column normalise along the residual-stream axis.
                m = self.div(m, self.add(self.sum(m, dim=2, keepdim=True), self.eps))
            return m

        h_max, _ = self.max(h, -1, True)
        m = self.exp(h - h_max)
        for _ in range(self.iterations):
            denominator = self.clamp(self.sum(m, dim=3, keepdim=True), min=self.eps)
            m = self.div(m, denominator)
            denominator = self.clamp(self.sum(m, dim=2, keepdim=True), min=self.eps)
            m = self.div(m, denominator)
        return m


class HyperConnectionOutputCell(nn.Cell):
    """Residual-stream update after a sublayer.

    Computes:
        new_streams = H_res^T @ x_streams + H_post * sublayer_out
                      [s,b,n,n]@[s,b,n,H] + [s,b,n,1]*[s,b,1,H]
                    = [s,b,n,H] + [s,b,n,H]
    """

    def __init__(
            self,
            n,
            hidden_size,
            dtype=mstype.bfloat16,
            compute_dtype=mstype.float32,
            megatron_compatibility=False,
    ):
        super().__init__()
        self.n = n
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.megatron_compatibility = megatron_compatibility
        self.reshape = mint.reshape
        self.matmul = mint.matmul
        self.bmm = mint.bmm
        self.mul = mint.mul
        self.add = mint.add
        self.transpose = mint.permute
        self.cast = ops.cast

    def construct(self, h_res, h_post, original_streams, sublayer_out):
        """Update residual streams.

        Args:
            h_res: [s, b, n, n] doubly-stochastic residual matrix
            h_post: [s, b, n, 1] per-stream output scale
            original_streams: [s, b, n*H] packed streams before the sublayer
            sublayer_out: [s, b, H] dropout output of the sublayer

        Returns:
            Updated packed streams with shape [s, b, n*H].
        """
        s = original_streams.shape[0]
        n, hidden_size = self.n, self.hidden_size

        if self.megatron_compatibility:
            h_res_batched = self.reshape(
                self.cast(h_res, self.compute_dtype), (-1, n, n)
            )
            x_streams = self.cast(original_streams, self.compute_dtype)
            x_streams_batched = self.reshape(x_streams, (-1, n, hidden_size))
            res_part = self.bmm(h_res_batched, x_streams_batched)
            res_part = self.reshape(res_part, (s, -1, n, hidden_size))

            sublayer_exp = self.reshape(sublayer_out, (s, -1, 1, hidden_size))
            post_part = self.mul(
                self.cast(h_post, self.compute_dtype),
                self.cast(sublayer_exp, self.compute_dtype),
            )
            output = self.cast(self.add(res_part, post_part), self.dtype)
            return self.reshape(output, (s, -1, n * hidden_size))

        # H_res^T @ streams: [s,b,n,n] @ [s,b,n,H] -> [s,b,n,H]
        x_streams = self.reshape(original_streams, (s, -1, n, hidden_size))
        x_streams = self.cast(x_streams, mstype.float32)
        h_res = self.cast(h_res, mstype.float32)
        # MindSpeed's reference computes
        #   sum_i h_res[i, j] * stream[i]
        # for output stream j, i.e. h_res^T @ streams.  This orientation is
        # observable once the learned Sinkhorn matrix is no longer symmetric.
        h_res_t = self.transpose(h_res, (0, 1, 3, 2))
        res_part = self.matmul(h_res_t, x_streams)

        # H_post * sublayer_out: [s,b,n,1] * [s,b,1,H] -> [s,b,n,H]
        sublayer_exp = self.reshape(sublayer_out, (s, -1, 1, hidden_size))
        sublayer_exp = self.cast(sublayer_exp, mstype.float32)
        h_post = self.cast(h_post, mstype.float32)
        post_part = self.mul(h_post, sublayer_exp)

        output = self.cast(self.add(res_part, post_part), self.dtype)
        return self.reshape(output, (s, -1, n * hidden_size))


class HyperConnectionHead(nn.Cell):
    """Learnable mHC head used to collapse the final residual streams.

    MindSpeed applies this head before the main decoder final RMSNorm and before
    each MTP final RMSNorm.  Keeping it learnable is important for multi-step
    loss alignment; a fixed arithmetic mean is only equivalent at a specially
    initialised checkpoint and diverges after the first optimiser update.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.dtype = config.compute_dtype
        self.norm_eps = config.mhc_layernorm_epsilon
        self.hc_eps = 1e-6

        self.hc_fn = Linear(
            input_size=self.n * self.hidden_size,
            output_size=self.n,
            compute_dtype=mstype.float32,
            params_dtype=mstype.float32,
            init_method=config.init_method,
            bias=False,
        )
        self.hc_base = Parameter(mint.empty((self.n,), dtype=mstype.float32), name="hc_base")
        self.hc_scale = Parameter(mint.empty((1,), dtype=mstype.float32), name="hc_scale")

        self.reshape = mint.reshape
        self.cast = ops.cast
        self.mean = mint.mean
        self.rsqrt = mint.rsqrt
        self.sigmoid = mint.sigmoid
        self.sum = mint.sum
        self.unsqueeze = mint.unsqueeze

    def construct(self, hidden_states):
        """Collapse packed ``[s, b, n*h]`` streams into ``[s, b, h]``."""
        s, _, _ = hidden_states.shape
        flat = self.cast(hidden_states, mstype.float32)
        rms_inv = self.rsqrt(self.mean(flat * flat, dim=-1, keepdim=True) + self.norm_eps)
        mixes = self.hc_fn(flat) * rms_inv
        pre = self.sigmoid(
            mixes * _to_local(self.hc_scale) + _to_local(self.hc_base)
        ) + self.hc_eps

        streams = self.reshape(flat, (s, -1, self.n, self.hidden_size))
        output = self.sum(self.unsqueeze(pre, -1) * streams, dim=2)
        return self.cast(output, self.dtype)

    def reset_parameter(self):
        """Reset the scalar and bias gates for delayed initialisation."""
        self.hc_base.zero_()
        self.hc_scale.zero_()


class HyperConnectionModule(nn.Cell):
    """Unified mHC module for PyNative mode.

    A single RMSNorm + single projection produces H_pre, H_post and H_res in one shot.
    Hidden states flow as packed residual streams with shape [s, b, n*H].

    Typical usage in TransformerLayer:

        aggregated, h_res, h_post = self.attn_hc(hidden_states)
        x = input_layernorm(aggregated)
        x = self_attention(x)
        x = dropout(x)
        hidden_states = self.attn_hc.output_cell(h_res, h_post, hidden_states, x)
    """

    def __init__(self, config: TransformerConfig, layer_number: int):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.n = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.dtype = config.compute_dtype
        self.params_dtype = (
            mstype.float32 if config.mhc_params_dtype is None
            else config.mhc_params_dtype
        )
        self.compute_dtype = (
            mstype.float32 if config.mhc_compute_dtype is None
            else config.mhc_compute_dtype
        )
        self.custom_precision = (
            config.mhc_params_dtype is not None
            or config.mhc_compute_dtype is not None
        )
        self.megatron_compatibility = config.mhc_megatron_compatibility
        # Configuration validation normalizes this to (min, max) or None.
        self.hc_sinkhorn_clamp = getattr(config, 'hc_sinkhorn_clamp', None)
        self.sinkhorn_iterations = config.mhc_sinkhorn_iterations
        self.norm_eps = config.mhc_layernorm_epsilon
        hidden_size = self.hidden_size
        n = self.n
        dim = n + n + n * n

        self.mapping_proj = Linear(
            input_size=n * hidden_size,
            output_size=dim,
            compute_dtype=self.compute_dtype,
            params_dtype=self.params_dtype,
            init_method=init_method_zero(self.params_dtype),
            bias=False,
        )
        self.alpha_pre = Parameter(mint.empty((1,), dtype=self.params_dtype), name='alpha_pre')
        self.alpha_post = Parameter(mint.empty((1,), dtype=self.params_dtype), name='alpha_post')
        self.alpha_res = Parameter(mint.empty((1,), dtype=self.params_dtype), name='alpha_res')

        self.bias = Parameter(mint.empty((dim,), dtype=self.params_dtype), name='bias')

        self.rms_weight = Parameter(
            mint.empty((n * hidden_size,), dtype=mstype.float32),
            name="rms_weight",
            requires_grad=False,
        )
        self.rms_norm = ops.rms_norm
        self._sinkhorn_op = SinkhornKnopp(
            self.sinkhorn_iterations,
            megatron_compatibility=self.megatron_compatibility,
        )
        self.output_cell = HyperConnectionOutputCell(
            n,
            self.hidden_size,
            self.dtype,
            compute_dtype=self.compute_dtype,
            megatron_compatibility=self.megatron_compatibility,
        )

        self.reshape = mint.reshape
        self.matmul = mint.matmul
        self.mm = mint.mm
        self.norm = mint.norm
        self.sigmoid = mint.sigmoid
        self.squeeze = mint.squeeze
        self.unsqueeze = mint.unsqueeze
        self.sum = mint.sum
        self.mul = mint.mul
        self.add = mint.add
        self.tile = mint.tile
        self.cast = ops.cast
        self.transpose = mint.permute
        self.concat = mint.concat
        self.split = mint.split

    def _clamp_logits(self, h_res):
        """Clamp residual logits before Sinkhorn; return them unchanged when disabled."""
        if self.hc_sinkhorn_clamp is None:
            return h_res
        lo, hi = self.hc_sinkhorn_clamp
        return mint.clamp(h_res, lo, hi)


    def construct(self, hidden_states):
        """Compute connection matrices and aggregate streams.

        Args:
            hidden_states: [s, b, n*H] packed residual streams

        Returns:
            aggregated: [s, b, H] weighted input for the sublayer
            h_res: [s, b, n, n] doubly-stochastic residual matrix
            h_post: [s, b, n, 1] per-stream output scale
        """
        s, _, n_hidden = hidden_states.shape
        n = self.n
        hidden_size = self.hidden_size

        if self.megatron_compatibility:
            x2d = self.cast(self.reshape(hidden_states, (-1, n_hidden)), self.compute_dtype)
            rms = self.norm(x2d, dim=-1, keepdim=True) / math.sqrt(n_hidden)
            r = 1.0 / (rms + self.norm_eps)

            proj_w_t = self.transpose(
                self.cast(_to_local(self.mapping_proj.weight), self.compute_dtype),
                (1, 0),
            )
            proj = self.mm(x2d, proj_w_t)
            alpha_ = self.concat(
                (
                    self.tile(_to_local(self.alpha_pre), (n,)),
                    self.tile(_to_local(self.alpha_post), (n,)),
                    self.tile(_to_local(self.alpha_res), (n * n,)),
                ),
                dim=-1,
            )
            alpha_ = self.cast(alpha_, self.compute_dtype)
            bias = self.cast(_to_local(self.bias), self.compute_dtype)
            h = self.add(self.mul(self.mul(r, proj), alpha_), bias)
            h = self.reshape(h, (s, -1, n + n + n * n))

            h_pre, h_post, h_res = self.split(h, [n, n, n * n], dim=2)
            h_pre = self.sigmoid(h_pre)
            h_post = self.mul(2.0, self.sigmoid(h_post))
            h_res = self.reshape(h_res, (s, -1, n, n))
            h_res = self._clamp_logits(h_res)
            h_res = self._sinkhorn_op(h_res)
            h_post = self.reshape(h_post, (s, -1, n, 1))

            x_streams = self.reshape(hidden_states, (s, -1, n, hidden_size))
            x_streams = self.cast(x_streams, self.compute_dtype)
            aggregate_terms = self.mul(self.unsqueeze(h_pre, -1), x_streams)
            aggregated = self.cast(self.sum(aggregate_terms, dim=2), self.dtype)
            return aggregated, h_res, h_post

        # 1. RMSNorm with fixed gamma=1.
        rms_weight = _to_local(self.rms_weight)
        if self.compute_dtype != mstype.float32:
            rms_weight = self.cast(rms_weight, self.compute_dtype)
        norm_x = self.rms_norm(
            self.cast(hidden_states, self.compute_dtype),
            rms_weight,
            self.norm_eps
        )[0]

        # 2. Project normalised streams -> [s, b, 1, n+n+n^2].
        # MindSpeed keeps the weightless RMSNorm and mHC projection in FP32,
        # even for a BF16 model.  Premature BF16 casts here measurably shift the
        # training loss and gradients.
        x4d = self.reshape(norm_x, (s, -1, 1, n_hidden))
        h = self.mapping_proj(x4d)
        h = self.cast(h, self.compute_dtype)

        alpha_ = self.concat(
            (
                self.tile(_to_local(self.alpha_pre), (n,)),
                self.tile(_to_local(self.alpha_post), (n,)),
                self.tile(_to_local(self.alpha_res), (n * n,)),
            ),
            dim=-1,
        )
        if self.custom_precision:
            alpha_ = self.cast(alpha_, self.compute_dtype)
            bias = self.cast(_to_local(self.bias), self.compute_dtype)
            h = self.add(self.mul(h, alpha_), bias)
        else:
            h = self.add(self.mul(h, alpha_), _to_local(self.bias))

        # 3. Split raw projection.
        h_pre, h_post, h_res = self.split(h, [n, n, n * n], dim=3)

        # 4. Apply activation and Sinkhorn projection.
        h_pre = self.add(self.sigmoid(h_pre), self._sinkhorn_op.eps)
        h_post = self.mul(2.0, self.sigmoid(h_post))
        h_res = self.reshape(h_res, (s, -1, n, n))
        h_res = self._clamp_logits(h_res)
        h_res = self._sinkhorn_op(h_res)
        h_post = self.reshape(h_post, (s, -1, n, 1))

        # 5. Aggregate streams: [s,b,1,n] @ [s,b,n,H] -> [s,b,H]
        x_streams = self.reshape(hidden_states, (s, -1, n, hidden_size))
        x_streams = self.cast(x_streams, self.compute_dtype)
        aggregated = self.cast(self.squeeze(self.matmul(h_pre, x_streams), 2), self.dtype)

        return aggregated, h_res, h_post

    def reset_parameter(self):
        """Reset MHC parameters for delayed initialization."""
        # Reset mapping projection weight with custom initialization (mean=0, std=1e-4)
        self.mapping_proj.weight.normal_(mean=0.0, std=1e-4)
        if self.mapping_proj.has_bias and self.mapping_proj.bias is not None:
            self.mapping_proj.bias.zero_()

        # Reset alpha parameters to init_gating_factor
        init_alpha = self.config.mhc_init_gating_factor
        self.alpha_pre.fill_(init_alpha)
        self.alpha_post.fill_(init_alpha)
        self.alpha_res.fill_(init_alpha)

        # Reset bias with original initialization values
        n = self.n
        pre_b = np.full((n,), -np.log(3), np.float32)
        post_b = np.zeros((n,), dtype=np.float32)
        res_b = ((np.eye(n, dtype=np.float32) - 1) * 5).reshape(n * n)
        bias = np.concatenate([pre_b, post_b, res_b], axis=0)
        self.bias.set_data(Tensor(bias, self.params_dtype))

        # Reset rms_weight to ones
        self.rms_weight.fill_(1.0)

class FusedHyperConnectionOutputCell(nn.Cell):
    """Residual-stream update after a sublayer using fused npu_mhc_post kernel.

    Computes, via a single Ascend custom kernel call:
        megatron_compatibility=False: new_streams = h_res^T @ x_streams + h_post * sublayer_out
        megatron_compatibility=True:  new_streams = h_res   @ x_streams + h_post * sublayer_out

    The kernel aggregates column-major -- ``out[k] = sum_m comb[m, k] * x[m]`` -- so passing
    h_res unchanged yields the h_res^T orientation of HyperConnectionOutputCell's default
    branch, and the Megatron orientation needs the inner (n, n) transposed first. Measured
    on-device against both numpy references (scripts/mhc_post_unit.py): the column-major
    reference matches to bf16 rounding while the row-major one is off by 45% relative.
    """

    def __init__(self, n, hidden_size, dtype=mstype.bfloat16, megatron_compatibility=False):
        super().__init__()
        self.n = n
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.megatron_compatibility = megatron_compatibility
        self.reshape = mint.reshape
        self.transpose = mint.permute
        self.squeeze = mint.squeeze
        self.cast = ops.cast

    def construct(self, h_res, h_post, original_streams, sublayer_out):
        """Update residual streams via fused kernel.

        Args:
            h_res: [s, b, n, n] doubly-stochastic residual matrix
            h_post: [s, b, n, 1] per-stream output scale
            original_streams: [s, b, n*H] packed streams before the sublayer
            sublayer_out: [s, b, H] dropout output of the sublayer

        Returns:
            Updated packed streams with shape [s, b, n*H].
        """
        s, b, _ = original_streams.shape
        n = self.n
        hidden_size = self.hidden_size

        x = self.reshape(original_streams, (s, b, n, hidden_size))
        x = self.cast(x, self.dtype)
        sublayer_out = self.cast(sublayer_out, self.dtype)
        if self.megatron_compatibility:
            # Undo the kernel's column-major aggregation to get h_res @ x.
            h_res = self.transpose(h_res, (0, 1, 3, 2))

        output = npu_mhc_post(x, h_res, sublayer_out, self.squeeze(h_post, -1))
        return self.reshape(output, (s, b, n * hidden_size))


class FusedHyperConnectionModule(HyperConnectionModule):
    """Fused mHC module using npu_mhc_pre_sinkhorn and npu_mhc_post custom kernels.

    Inherits from HyperConnectionModule so that parameter names, shapes and
    initialisation are identical — checkpoints can be loaded without any name
    mapping.  Only the forward path is overridden to call the fused kernels.

    Typical usage in TransformerLayer:

        aggregated, h_res, h_post = self.attn_hc(hidden_states)
        x = input_layernorm(aggregated)
        x = self_attention(x)
        x = dropout(x)
        hidden_states = self.attn_hc.output_cell(h_res, h_post, hidden_states, x)
    """

    def __init__(self, config: TransformerConfig, layer_number: int):
        super().__init__(config=config, layer_number=layer_number)
        # The clamp kernel also uses different Sinkhorn epsilon handling. Select
        # the configured kernel to preserve the reference checkpoint semantics.
        if self.hc_sinkhorn_clamp is not None:
            _require_fused_mhc_kernel(npu_mhc_pre_clamp_sinkhorn, "npu_mhc_pre_clamp_sinkhorn",
                                      "hc_sinkhorn_clamp")
        else:
            _require_fused_mhc_kernel(npu_mhc_pre_sinkhorn, "npu_mhc_pre_sinkhorn",
                                      "use_fused_mhc=True")
        _require_fused_mhc_kernel(npu_mhc_post, "npu_mhc_post", "use_fused_mhc=True")
        self.hc_eps = 1e-6
        self.output_cell = FusedHyperConnectionOutputCell(
            self.n, self.hidden_size, self.dtype,
            megatron_compatibility=self.megatron_compatibility,
        )

    def construct(self, hidden_states):
        """Compute connection matrices and aggregate streams via fused kernel.

        Args:
            hidden_states: [s, b, n*H] packed residual streams (SBH layout)

        Returns:
            aggregated: [s, b, H] weighted input for the sublayer
            h_res: [s, b, n, n] doubly-stochastic residual matrix
            h_post: [s, b, n, 1] per-stream output scale
        """
        s, b, _ = hidden_states.shape
        n = self.n
        hidden_size = self.hidden_size

        x = self.reshape(hidden_states, (s, b, n, hidden_size))
        x = self.cast(x, self.dtype)

        alpha = self.concat(
            (_to_local(self.alpha_pre), _to_local(self.alpha_post), _to_local(self.alpha_res)),
            dim=-1,
        )

        kernel_kwargs = {
            "hc_mult": n,
            "num_iters": self.sinkhorn_iterations,
            "hc_eps": self.hc_eps,
            "norm_eps": self.norm_eps,
        }
        if self.hc_sinkhorn_clamp is not None:
            lo, hi = self.hc_sinkhorn_clamp
            h_in, h_post, h_res_flat, *_ = npu_mhc_pre_clamp_sinkhorn(
                x, _to_local(self.mapping_proj.weight), alpha, _to_local(self.bias),
                clamp_min=lo, clamp_max=hi, **kernel_kwargs)
        else:
            h_in, h_post, h_res_flat, *_ = npu_mhc_pre_sinkhorn(
                x, _to_local(self.mapping_proj.weight), alpha, _to_local(self.bias),
                **kernel_kwargs)

        aggregated = h_in
        h_res = self.reshape(h_res_flat, (s, b, n, n))
        h_post = self.reshape(h_post, (s, b, n, 1))
        return aggregated, h_res, h_post
