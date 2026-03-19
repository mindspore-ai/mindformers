# Copyright 2025 Huawei Technologies Co., Ltd
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
# pylint: disable=R0902,R0913,R0914,R0917
"""Unified mHC (Manifold-Constrained Hyper-Connections) — Megatron-style.

Key improvements over the three-separate-module design
(PrePostHC × 4 + ResHC × 2):

1. **Single projection per HC block** (n*H → n + n + n²).
   One RMSNorm call + one BatchMatMul produces H_pre, H_post and H_res
   jointly, compared with three independent fi projections before.

2. **Packed 3D hidden states [s, b, n*H]** flow between layers.
   expand / contract happen only once at TransformerBlock boundaries,
   eliminating the per-layer reshape overhead.

3. **SinkhornKnopp marked with .recompute()**.
   Intermediate loop tensors are NOT kept during the forward pass;
   they are recomputed on demand during backward — the same memory saving
   that Megatron achieves with a custom autograd Function.

4. **Separate alpha_pre / alpha_post / alpha_res** for independent
   per-type learning rates.

5. **Legacy-proven bias init** (better than Megatron's all-zeros):
     pre  : -log(3)  → sigmoid ≈ 0.25, uniform initial stream mixing
     post : 0        → sigmoid ≈ 0.5
     res  : (I−1)×5  → strong diagonal prior, fewer Sinkhorn iterations

6. **HyperConnectionOutputCell** is a standalone Cell so that its ops
   (res_mm, mul_outer, add_output) are compiled into the MindSpore graph
   with correct shard annotations.  Calling a plain Python method from
   inside another Cell's construct() bypasses graph compilation and
   silently drops all shard strategies.
"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn, Parameter, Tensor
from mindspore.ops import auto_generate as aclnn_ops
from mindformers.parallel_core.training_graph.device_matrix import layout


class SinkhornKnopp(nn.Cell):
    """Doubly-stochastic projection via alternating row/column normalisation.

    Uses exp-based initialisation (Megatron style) rather than softmax, so
    neither rows nor columns are pre-normalised before the first iteration.

    This Cell should be called with `.recompute()` on the instance so that
    the n*iters intermediate tensors inside the loop are NOT saved during
    the forward pass but recomputed during backward — trading compute for
    activation memory, exactly as Megatron's custom autograd Function does.
    """

    # Same threshold as Megatron's SinkhornKnopp.eps = 1e-6
    _EPS = 1e-6

    def __init__(self, iters: int):
        super().__init__()
        self.iters   = iters
        self.cast    = aclnn_ops.Cast()
        self.exp     = aclnn_ops.Exp()
        self.sub     = aclnn_ops.SubExt()
        self.div     = aclnn_ops.Div()
        self.add_eps = aclnn_ops.AddExt()
        self.max_op  = aclnn_ops.ReduceMax()
        self.expand_max = aclnn_ops.ExpandDims()
        self.row_sum = aclnn_ops.SumExt()
        self.col_sum = aclnn_ops.SumExt()
        # Epsilon tensor for clamp-equivalent: prevents div-by-zero when a row
        # or column of M collapses to zero (e.g. bfloat16 underflow on Ascend).
        # Megatron uses .clamp(min=1e-6); we add eps before dividing.
        # Shape [1,1,1,1] (not 0-D scalar) so the shard spec matches the 4-D
        # operand from row_sum / col_sum — a 0-D shard would cause a shape
        # mismatch in semi-auto parallel graph compilation.
        self.eps = Tensor(np.full((1, 1, 1, 1), self._EPS, np.float32), mstype.float32)

        self.shard()

    def shard(self):
        """Set shard strategies for all ops on [s, b, n, n] tensors."""
        shard_4d  = layout("tp", "dp", "None", "None")
        shard_nnn = layout("None", "None", "None", "None")
        self.exp.shard((shard_4d,))
        self.sub.shard((shard_4d, shard_4d))
        self.div.shard((shard_4d, shard_4d))
        self.add_eps.shard((shard_4d, shard_nnn))  # [s,b,n,1] + [1,1,1,1] → broadcast ok
        shard_3d  = layout("tp", "dp", "None")
        self.max_op.shard(in_strategy=(shard_4d,), out_strategy=(shard_3d,))
        self.max_op.add_prim_attr("self_define_shard", True)
        self.expand_max.shard((shard_3d,))
        self.row_sum.shard(in_strategy=(shard_4d,), out_strategy=(shard_4d,))
        self.row_sum.add_prim_attr("self_define_shard", True)
        self.col_sum.shard(in_strategy=(shard_4d,), out_strategy=(shard_4d,))
        self.col_sum.add_prim_attr("self_define_shard", True)

    def construct(self, h):
        """Project raw logits onto the Birkhoff polytope.

        Args:
            h: [s, b, n, n]  raw logits, float32
        Returns:
            Doubly-stochastic matrix [s, b, n, n], float32.
        """
        h = self.cast(h, mstype.float32)
        h_max = self.expand_max(self.max_op(h, -1), -1)            # numerically stable exp
        m = self.exp(self.sub(h, h_max))                         # [s, b, n, n]
        for _ in range(self.iters):
            # row normalise  (axis=3): clamp-equivalent via +eps, same as Megatron
            m = self.div(m, self.add_eps(self.row_sum(m, 3, True), self.eps))
            # col normalise  (axis=2)
            m = self.div(m, self.add_eps(self.col_sum(m, 2, True), self.eps))
        return m


class HyperConnectionOutputCell(nn.Cell):
    """Residual-stream update after a sublayer (attention or FFN).

    Factored into its own Cell so that MindSpore's graph compiler picks up
    the construct() method and applies all .shard() annotations correctly.
    Calling a plain Python helper method from inside another Cell's
    construct() is silently treated as Python-level execution — the graph
    compiler does not compile it and all shard strategies are ignored.

    Computes:
        new_streams = H_res @ x_streams  +  H_post ⊗ sublayer_out
                      [s,b,n,n]@[s,b,n,H]   [s,b,n,1]*[s,b,1,H]
                    = [s,b,n,H]              = [s,b,n,H]
    """

    def __init__(self, rate, hidden_size, dtype=mstype.bfloat16):
        super().__init__()
        self.rate        = rate
        self.hidden_size = hidden_size
        self.dtype       = dtype

        self.cast             = aclnn_ops.Cast()
        self.reshape_streams  = aclnn_ops.Reshape()
        self.reshape_sublayer = aclnn_ops.Reshape()
        self.reshape_output   = aclnn_ops.Reshape()
        self.res_mm           = aclnn_ops.BatchMatMul()
        self.mul_outer        = aclnn_ops.Mul()
        self.add_output       = aclnn_ops.AddExt()

        self.shard()

    def shard(self):
        """Set shard strategies for output cell ops on [s, b, n, H] tensors."""
        shard_4d = layout("tp", "dp", "None", "None")
        self.res_mm.shard((shard_4d, shard_4d))
        self.mul_outer.shard((shard_4d, shard_4d))
        self.add_output.shard((shard_4d, shard_4d))

    def construct(self, h_res, h_post, original_streams, sublayer_out):
        """
        Args:
            h_res:            [s, b, n, n]  doubly-stochastic matrix
            h_post:           [s, b, n, 1]  per-stream output scale
            original_streams: [s, b, n*H]   packed streams before the sublayer
            sublayer_out:     [s, b, H]      dropout output of sublayer
        Returns:
            [s, b, n*H]  updated packed streams
        """
        s          = original_streams.shape[0]
        n, h_size  = self.rate, self.hidden_size
        orig_dtype = original_streams.dtype

        # H_res @ streams:  [s,b,n,n] @ [s,b,n,H] → [s,b,n,H]
        x_streams = self.reshape_streams(
            self.cast(original_streams, self.dtype), (s, -1, n, h_size))
        res_part  = self.res_mm(h_res, x_streams)

        # H_post ⊗ sublayer_out:  [s,b,n,1] * [s,b,1,H] → [s,b,n,H]
        sublayer_exp = self.reshape_sublayer(
            self.cast(sublayer_out, self.dtype), (s, -1, 1, h_size))
        post_part = self.mul_outer(h_post, sublayer_exp)

        output = self.add_output(res_part, post_part)
        return self.cast(
            self.reshape_output(output, (s, -1, n * h_size)), orig_dtype)


class HyperConnectionModule(nn.Cell):
    """Megatron-style unified mHC module.

    A single RMSNorm + single BatchMatMul projection produces H_pre, H_post
    and H_res in one shot.  Hidden states flow as [s, b, n*H] (packed).

    Typical usage in TransformerLayer::

        # Attention block
        aggregated, h_res, h_post = self.attn_hc(hidden_states)
        x = input_layernorm(aggregated)                    # [s, b, H]
        x = self_attention(x)
        x = dropout(x)
        hidden_states = self.attn_hc.output_cell(
            h_res, h_post, hidden_states, x)              # [s, b, n*H]

        # FFN block
        aggregated, h_res, h_post = self.ffn_hc(hidden_states)
        x = pre_mlp_layernorm(aggregated)
        x = mlp(x)
        x = dropout(x)
        hidden_states = self.ffn_hc.output_cell(
            h_res, h_post, hidden_states, x)              # [s, b, n*H]

    output_cell is a HyperConnectionOutputCell instance whose construct()
    holds the post-sublayer update logic.  It must be a proper Cell (not a
    plain method) so MindSpore compiles it and honours all .shard() calls.
    """

    def __init__(self, rate, hidden_size, config,
                 sinkhorn_iters=20, init_gating_factor=0.01,
                 dtype=mstype.bfloat16,
                 shared_rms_norm=None, expand_post=2.0, **kwargs):
        super().__init__()
        _ = config, kwargs  # reserved for future config-driven init
        self.rate        = rate
        self.hidden_size = hidden_size
        self.dtype       = dtype
        n   = rate
        h_sz = hidden_size
        dim = n + n + n * n        # pre + post + res output dims

        # ── Single projection  [1, 1, n*H, dim] ─────────────────────────────
        fi = np.random.normal(0, 1e-4, (1, 1, n * h_sz, dim)).astype(np.float32)
        self.mapping_weight = Parameter(
            Tensor(fi, mstype.float32), parallel_optimizer=False)

        # ── Per-type alphas  [1, 1, 1, 1] ────────────────────────────────────
        def _alpha():
            return Parameter(
                Tensor(np.full((1, 1, 1, 1), init_gating_factor, np.float32), mstype.float32),
                parallel_optimizer=False)
        self.alpha_pre  = _alpha()
        self.alpha_post = _alpha()
        self.alpha_res  = _alpha()

        # ── Bias — legacy-proven initialisation ──────────────────────────────
        # Correct order: sigmoid(alpha * proj + bias), NOT sigmoid(alpha * (proj + bias)).
        # The second form would dilute the bias by alpha at init, almost
        # nullifying the -log(3) pre-bias and destroying the (I-1)*5 res prior.
        # Three separate Parameter tensors so each can be added AFTER alpha mul.
        #   pre:  -log(3)  → sigmoid(α*proj - log3) ≈ 0.25 at init
        #   post:  0       → sigmoid(α*proj) ≈ 0.5 at init
        #   res:  (I−1)×5  → Sinkhorn input is large diagonal → ≈ identity matrix
        pre_b  = np.full((1, 1, 1, n),     -np.log(3),               np.float32)
        post_b = np.zeros((1, 1, 1, n),                               dtype=np.float32)
        res_b  = ((np.eye(n, dtype=np.float32) - 1) * 5).reshape(1, 1, 1, n * n)
        self.bias_pre  = Parameter(Tensor(pre_b,  mstype.float32), parallel_optimizer=False)
        self.bias_post = Parameter(Tensor(post_b, mstype.float32), parallel_optimizer=False)
        self.bias_res  = Parameter(Tensor(res_b,  mstype.float32), parallel_optimizer=False)

        # expand_post scales H_post sigmoid (default 2.0 → range [0, 2])
        expand_arr = np.array(  # pylint: disable=too-many-function-args
            [expand_post], np.float32).reshape(1, 1, 1, 1)
        self.expand_post_val = Tensor(expand_arr, mstype.float32)

        # ── Norm (shared across pre / post / res in this block) ──────────────
        self.rms_norm = shared_rms_norm

        # ── Sinkhorn — recomputed during backward (no intermediate activations)
        self.sinkhorn = SinkhornKnopp(sinkhorn_iters)
        self.sinkhorn.recompute()

        # ── Post-sublayer update cell (Fix: must be a Cell, not a plain method)
        self.output_cell = HyperConnectionOutputCell(rate, hidden_size, dtype)

        # ── Primitive ops ─────────────────────────────────────────────────────
        self.cast            = aclnn_ops.Cast()
        self.reshape         = aclnn_ops.Reshape()
        self.sigmoid         = aclnn_ops.Sigmoid()
        # mapping_mm: [s,b,1,n*H] @ [1,1,n*H,dim] → [s,b,1,dim]
        # Weight batch dims are size-1 (broadcast); self_define_shard lets us
        # assert the output layout explicitly and bypasses the batch-shape
        # consistency check that BatchMatMul would otherwise fail on.
        self.mapping_mm      = aclnn_ops.BatchMatMul()
        self.pre_mm          = aclnn_ops.BatchMatMul()  # [s,b,1,n] @ [s,b,n,H] → aggregate
        self.squeeze         = aclnn_ops.Squeeze()
        self.mul_alpha_pre   = aclnn_ops.Mul()
        self.mul_alpha_post  = aclnn_ops.Mul()
        self.mul_alpha_res   = aclnn_ops.Mul()
        self.mul_expand_post = aclnn_ops.Mul()
        self.add_bias_pre    = aclnn_ops.AddExt()
        self.add_bias_post   = aclnn_ops.AddExt()
        self.add_bias_res    = aclnn_ops.AddExt()

        self.shard()

    def shard(self):
        """Set shard strategies for all HyperConnectionModule ops."""
        shard_4d  = layout("tp", "dp", "None", "None")
        shard_nnn = layout("None", "None", "None", "None")

        self.sigmoid.shard((shard_4d,))
        # Use self_define_shard so the broadcast batch dims (weight [1,1,...])
        # do not trigger a shape-consistency error in semi-auto parallel.
        self.mapping_mm.shard(in_strategy=(shard_4d, shard_nnn), out_strategy=(shard_4d,))
        self.mapping_mm.add_prim_attr("self_define_shard", True)
        self.pre_mm.shard((shard_4d, shard_4d))
        self.squeeze.shard((shard_4d,))
        self.mul_alpha_pre.shard((shard_nnn, shard_4d))
        self.mul_alpha_post.shard((shard_nnn, shard_4d))
        self.mul_alpha_res.shard((shard_nnn, shard_4d))
        self.mul_expand_post.shard((shard_nnn, shard_4d))
        self.add_bias_pre.shard((shard_4d, shard_nnn))
        self.add_bias_post.shard((shard_4d, shard_nnn))
        self.add_bias_res.shard((shard_4d, shard_nnn))

    # ─────────────────────────────────────────────────────────────────────────
    def construct(self, hidden_states):
        """Compute connection matrices and aggregate streams.

        Args:
            hidden_states: [s, b, n*H]  packed residual streams (bfloat16)
        Returns:
            aggregated: [s, b, H]       weighted input for the sublayer
            h_res:      [s, b, n, n]    doubly-stochastic residual matrix
            h_post:     [s, b, n, 1]    per-stream output scale
        """
        s          = hidden_states.shape[0]
        n, h_sz    = self.rate, self.hidden_size
        orig_dtype = hidden_states.dtype   # bfloat16 (packed streams)

        # 1. Norm on packed representation  [s, b, n*H]
        x        = self.cast(hidden_states, self.dtype)
        norm_x   = self.rms_norm(x)                                # [s, b, n*H]
        norm_x4d = self.reshape(norm_x, (s, -1, 1, n * h_sz))     # [s, b, 1, n*H]

        # 2. Single projection  →  [s, b, 1, n+n+n²]
        # Cast to float32 for numerical stability of bias/alpha ops.
        h = self.mapping_mm(norm_x4d,
                            self.cast(self.mapping_weight, self.dtype))
        h = self.cast(h, mstype.float32)                           # [s, b, 1, dim] float32

        # 3. Split raw projection (bias NOT yet added — must be added AFTER alpha mul)
        h_pre  = h[:, :, :, :n]                                    # [s, b, 1, n]
        h_post = h[:, :, :, n:2 * n]                               # [s, b, 1, n]
        h_res  = h[:, :, :, 2 * n:]                                # [s, b, 1, n²]

        # 4. Activations: sigmoid(alpha * proj + bias)
        #    Correct order: mul alpha FIRST, add bias SECOND.
        #    Reversed order sigmoid(alpha*(proj+bias)) would scale bias by alpha,
        #    destroying the -log(3) pre-prior and the (I-1)*5 res diagonal prior.
        h_pre  = self.sigmoid(
            self.add_bias_pre(self.mul_alpha_pre(self.alpha_pre, h_pre),
                              self.bias_pre))                       # float32
        h_post = self.mul_expand_post(
            self.expand_post_val,
            self.sigmoid(
                self.add_bias_post(self.mul_alpha_post(self.alpha_post, h_post),
                                   self.bias_post)))                # float32
        h_res  = self.reshape(
            self.add_bias_res(self.mul_alpha_res(self.alpha_res, h_res),
                              self.bias_res), (s, -1, n, n))
        h_res  = self.cast(self.sinkhorn(h_res), self.dtype)        # [s, b, n, n]  bfloat16

        # Cast h_pre / h_post to self.dtype (computation dtype) so all three
        # tensors that go into BatchMatMul / Mul share the same dtype as
        # x_streams.  Using orig_dtype instead would mismatch when the caller
        # passes float32 hidden_states (e.g. during fp32 warmup or inference).
        h_pre  = self.cast(h_pre, self.dtype)                      # [s, b, 1, n]  bfloat16
        h_post = self.cast(
            self.reshape(h_post, (s, -1, n, 1)), self.dtype)       # [s, b, n, 1]  bfloat16

        # 5. Aggregate streams:  [s,b,1,n] @ [s,b,n,H] → [s,b,1,H] → [s,b,H]
        x_streams  = self.reshape(
            self.cast(hidden_states, self.dtype), (s, -1, n, h_sz))  # [s, b, n, H]  bfloat16
        aggregated = self.squeeze(self.pre_mm(h_pre, x_streams)) # [s, b, H]
        aggregated = self.cast(aggregated, orig_dtype)

        return aggregated, h_res, h_post
