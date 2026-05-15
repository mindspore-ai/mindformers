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

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn, Parameter, Tensor, mint, ops
from mindspore.common.initializer import initializer
try:
    from hyper_parallel.custom_ops.experimental import npu_mhc_pre_sinkhorn, npu_mhc_post
except ImportError:
    pass
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.init_method import init_method_zero
from mindformers.pynative.layers.linear import Linear


class SinkhornKnopp(nn.Cell):
    """Doubly-stochastic projection via alternating row/column normalisation.

    Uses exp-based initialisation (Megatron style) rather than softmax, so
    neither rows nor columns are pre-normalised before the first iteration.
    """

    _EPS = 1e-6

    def __init__(self, iterations: int):
        super().__init__()
        self.iterations = iterations
        self.eps = self._EPS
        self.max = mint.max
        self.exp = mint.exp
        self.sum = mint.sum
        self.add = mint.add
        self.div = mint.div

    def construct(self, h):
        """Project raw logits onto the Birkhoff polytope.

        Args:
            h: [s, b, n, n] raw logits, float32

        Returns:
            Doubly-stochastic matrix [s, b, n, n], float32.
        """
        h_max, _ = self.max(h, -1, True)
        m = self.exp(h - h_max)
        for _ in range(self.iterations):
            # Row normalise along the last axis.
            m = self.div(m, self.add(self.sum(m, dim=3, keepdim=True), self.eps))
            # Column normalise along the residual-stream axis.
            m = self.div(m, self.add(self.sum(m, dim=2, keepdim=True), self.eps))
        return m


class HyperConnectionOutputCell(nn.Cell):
    """Residual-stream update after a sublayer.

    Computes:
        new_streams = H_res @ x_streams + H_post * sublayer_out
                      [s,b,n,n]@[s,b,n,H] + [s,b,n,1]*[s,b,1,H]
                    = [s,b,n,H] + [s,b,n,H]
    """

    def __init__(self, n, hidden_size, dtype=mstype.bfloat16):
        super().__init__()
        self.n = n
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.reshape = mint.reshape
        self.matmul = mint.matmul
        self.mul = mint.mul
        self.add = mint.add
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

        # H_res @ streams: [s,b,n,n] @ [s,b,n,H] -> [s,b,n,H]
        x_streams = self.reshape(original_streams, (s, -1, n, hidden_size))
        x_streams = self.cast(x_streams, self.dtype)
        res_part = self.matmul(h_res, x_streams)

        # H_post * sublayer_out: [s,b,n,1] * [s,b,1,H] -> [s,b,n,H]
        sublayer_exp = self.reshape(sublayer_out, (s, -1, 1, hidden_size))
        post_part = self.mul(h_post, sublayer_exp)

        output = self.add(res_part, post_part)
        return self.reshape(output, (s, -1, n * hidden_size))


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
        self.sinkhorn_iterations = config.mhc_sinkhorn_iterations
        self.norm_eps = config.mhc_layernorm_epsilon
        hidden_size = self.hidden_size
        n = self.n
        dim = n + n + n * n

        self.mapping_proj = Linear(
            input_size=n * hidden_size,
            output_size=dim,
            compute_dtype=self.dtype,
            params_dtype=config.params_dtype,
            init_method=init_method_zero(config.params_dtype),
            bias=False,
        )
        self.alpha_pre = Parameter(mint.empty((1,), dtype=mstype.float32), name='alpha_pre')
        self.alpha_post = Parameter(mint.empty((1,), dtype=mstype.float32), name='alpha_post')
        self.alpha_res = Parameter(mint.empty((1,), dtype=mstype.float32), name='alpha_res')

        self.bias = Parameter(mint.empty((dim,), dtype=mstype.float32), name='bias')

        self.rms_weight = Parameter(
            mint.empty((n * hidden_size,), dtype=mstype.float32),
            name="rms_weight",
            requires_grad=False,
        )
        self.rms_norm = ops.rms_norm
        self._sinkhorn_op = SinkhornKnopp(self.sinkhorn_iterations)
        self.output_cell = HyperConnectionOutputCell(n, self.hidden_size, self.dtype)

        self.reshape = mint.reshape
        self.matmul = mint.matmul
        self.sigmoid = mint.sigmoid
        self.squeeze = mint.squeeze
        self.mul = mint.mul
        self.add = mint.add
        self.tile = mint.tile
        self.cast = ops.cast
        self.transpose = mint.permute
        self.concat = mint.concat
        self.split = mint.split

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

        # 1. RMSNorm with fixed gamma=1.
        norm_x = self.rms_norm(
            self.cast(hidden_states, mstype.float32),
            self.rms_weight,
            self.norm_eps
        )[0]

        # 2. Project normalised streams -> [s, b, 1, n+n+n^2].
        x4d = self.cast(self.reshape(norm_x, (s, -1, 1, n_hidden)), self.dtype)
        proj_w_t = self.transpose(self.cast(self.mapping_proj.weight, self.dtype), (1, 0))
        h = self.matmul(x4d, proj_w_t)
        h = self.cast(h, mstype.float32)

        alpha_ = self.concat(
            (
                self.tile(self.alpha_pre, (n,)),
                self.tile(self.alpha_post, (n,)),
                self.tile(self.alpha_res, (n * n,)),
            ),
            dim=-1,
        )
        h = self.add(self.mul(h, alpha_), self.bias)

        # 3. Split raw projection.
        h_pre, h_post, h_res = self.split(h, [n, n, n * n], dim=3)

        # 4. Apply activation and Sinkhorn projection.
        h_pre = self.sigmoid(h_pre)
        h_post = self.mul(2.0, self.sigmoid(h_post))
        h_res = self.reshape(h_res, (s, -1, n, n))
        h_res = self.cast(self._sinkhorn_op(h_res), self.dtype)

        h_pre = self.cast(h_pre, self.dtype)
        h_post = self.cast(self.reshape(h_post, (s, -1, n, 1)), self.dtype)

        # 5. Aggregate streams: [s,b,1,n] @ [s,b,n,H] -> [s,b,H]
        x_streams = self.reshape(hidden_states, (s, -1, n, hidden_size))
        x_streams = self.cast(x_streams, self.dtype)
        aggregated = self.squeeze(self.matmul(h_pre, x_streams), 2)

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
        self.bias.set_data(Tensor(bias, mstype.float32))

        # Reset rms_weight to ones
        self.rms_weight.fill_(1.0)

class FusedHyperConnectionOutputCell(nn.Cell):
    """Residual-stream update after a sublayer using fused npu_mhc_post kernel.

    Computes:
        new_streams = h_res^T @ x_streams + h_post * sublayer_out
    via a single Ascend custom kernel call.
    """

    def __init__(self, n, hidden_size, dtype=mstype.bfloat16):
        super().__init__()
        self.n = n
        self.hidden_size = hidden_size
        self.dtype = dtype
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
        self.hc_eps = 1e-6
        self.output_cell = FusedHyperConnectionOutputCell(self.n, self.hidden_size, self.dtype)

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

        alpha = self.concat((self.alpha_pre, self.alpha_post, self.alpha_res), dim=-1)

        h_in, h_post, h_res_flat, *_ = npu_mhc_pre_sinkhorn(
            x, self.mapping_proj.weight, alpha, self.bias,
            hc_mult=n,
            num_iters=self.sinkhorn_iterations,
            hc_eps=self.hc_eps,
            norm_eps=self.norm_eps,
        )

        aggregated = h_in
        h_res = self.reshape(h_res_flat, (s, b, n, n))
        h_post = self.reshape(h_post, (s, b, n, 1))
        return aggregated, h_res, h_post
