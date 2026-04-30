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
"""DSA utils."""
from mindspore import nn, ops
from mindspore.ops import auto_generate as aclnn_ops
import mindspore.ops.functional as F
from mindspore import dtype as mstype

from mindformers.parallel_core.training_graph.device_matrix import layout


def adjust_tnd_input(offset_id, q, k, actual_seq_qlen, actual_seq_klen):
    """adjust tnd input for dsa operators"""
    slice_tq = q.shape[0]
    slice_tk = k.shape[0]
    offset_q = slice_tq * offset_id
    new_actual_seq_qlen = F.cast(F.clamp(actual_seq_qlen - offset_q, 0, slice_tq), mstype.int32)
    new_actual_seq_klen = actual_seq_klen - F.relu(actual_seq_qlen - offset_q) + new_actual_seq_qlen
    prev_seq_klen = F.roll(new_actual_seq_klen, 1)
    prev_seq_klen[0] = 0
    new_actual_seq_klen = F.cumsum(F.relu(new_actual_seq_klen - prev_seq_klen), 0)
    new_actual_seq_klen[-1] = slice_tk
    return new_actual_seq_qlen, new_actual_seq_klen


def adjust_bsnd_input(offset_id, q, x):
    """adjust bsnd input for dsa operators"""
    slice_sq = q.shape[1]
    offset_q = slice_sq * (offset_id + 1)
    b, _, n, d = x.shape
    x = F.strided_slice(x, (0, 0, 0, 0), (b, offset_q, n, d), (1, 1, 1, 1))
    return x


class WeightAbsorb(nn.Cell):
    """weight absorb for MHA -> MQA"""
    def __init__(self):
        super().__init__()
        self.bmm = aclnn_ops.BatchMatMul()
        self.bs_transpose = aclnn_ops.Transpose()
        self.transpose = aclnn_ops.Transpose()
        self.reshape = aclnn_ops.Reshape()
        infer_dtype = lambda *args: args[0]
        self.absorb = ops.Morph(
            self._absorb_func,
            self.absorb_infer_shape,
            infer_dtype
        ).add_prim_attr("self_define_shard", True)
        self.shard()

    def absorb_infer_shape(self, *args):
        s, b, n, _ = args[0]
        d2 = args[1][-1]
        return s, b, n, d2

    def _absorb_func(self, x, x_absorb):
        s, b, n, d1 = x.shape
        d2 = x_absorb.shape[-1]
        x = self.reshape(x, ((s * b), n, d1))
        x = self.transpose(x, (1, 0, 2))
        out = self.bmm(x, x_absorb)
        out = self.reshape(self.transpose(out, (1, 0, 2)), (s, b, n, d2))
        return out

    def construct(self, x, absorb_weight, for_attn=False):
        if for_attn:
            x = self.bs_transpose(x, (1, 0, 2, 3))
        return self.absorb(x, absorb_weight)

    def shard(self):
        self.bs_transpose.shard((layout("dp", "cp", "tp", "None"),))
        self.absorb.shard(
            (layout("cp", "dp", "tp", "None"), layout("tp", "None", "None")),
            (layout("cp", "dp", "tp", "None"),)
        )
