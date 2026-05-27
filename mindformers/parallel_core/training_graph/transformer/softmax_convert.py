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
"""Convert Softmax_out Layout"""
from mindspore import nn, ops
from mindspore.ops import auto_generate as aclnn_ops
from mindspore import dtype as mstype

from mindformers.parallel_core.training_graph.communication import get_dp_cp_tp_id
from mindformers.parallel_core.training_graph.device_matrix import layout
from mindformers.parallel_core.transformer_config import TransformerConfig


class SoftmaxConverter(nn.Cell):
    """convert softmax_max and softmax_sum layout of FlashAttentionScore in TND layout"""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        if not config.input_layout != "TND":
            raise ValueError("Only softmax_out in TND layout need to be converted!")
        self.seq_length = config.seq_length
        self.head_num = config.num_attention_heads
        self.is_dryrun = config.is_dryrun
        self.tp = config.tensor_model_parallel_size
        self.cp = config.context_parallel_size
        self.cp_id = get_dp_cp_tp_id(config)[1]

        self.cast = aclnn_ops.Cast()
        self.reshape = aclnn_ops.Reshape()
        self.slice = aclnn_ops.StridedSlice().add_prim_attr("self_define_shard", True)
        self.clamp = aclnn_ops.ClampScalar()
        self.roll = aclnn_ops.Roll(1)
        self.repeat_scalar = aclnn_ops.RepeatInterleaveInt()
        self.arange = aclnn_ops.Arange()
        self.mod = aclnn_ops.FmodScalar()
        self.repeat_interleave = aclnn_ops.RepeatInterleaveTensor()
        self.equal = aclnn_ops.Equal()
        self.select = aclnn_ops.MaskedSelect()
        self.stack = aclnn_ops.StackExt()
        self.softmax_transpose = aclnn_ops.Transpose()
        infer_dtype = lambda *args: (args[0], args[1])
        self.convert_softmax = ops.Morph(
            self._convert_softmax,
            self.infer_shape, infer_dtype
        ).add_prim_attr("self_define_shard", True)
        self.shard()

    def infer_shape(self, *args):
        t, n, _ = args[0]
        out_shape = [n, t, 1]
        return out_shape, out_shape

    def _convert_softmax(self, softmax_max, softmax_sum, actual_seq_len):
        """convert softmax layout"""
        t, n, _ = softmax_max.shape
        softmax_max = self.reshape(softmax_max, (-1,))
        softmax_sum = self.reshape(softmax_sum, (-1,))
        partial_num_head = self.head_num // self.tp
        offset_q = t * self.cp_id
        actual_seq_qlen = self.cast(self.clamp(actual_seq_len - offset_q, 0, t), mstype.int32)
        prev_seq_qlen = self.roll(actual_seq_qlen)
        prev_seq_qlen[0] = 0
        interleave_seq_qlen = actual_seq_qlen - prev_seq_qlen
        interleave_seq_qlen = self.repeat_scalar(interleave_seq_qlen, partial_num_head)
        base = self.mod(self.arange(0, interleave_seq_qlen.shape[0], 1), partial_num_head)
        interleave_seq_qlen = self.repeat_interleave(base, interleave_seq_qlen)
        softmax_maxs = []
        softmax_sums = []
        for i in range(partial_num_head):
            head_mask = self.equal(interleave_seq_qlen, i)
            softmax_maxs.append(self.select(softmax_max, head_mask))
            softmax_sums.append(self.select(softmax_sum, head_mask))
        softmax_max = self.reshape(self.stack(softmax_maxs), (n, t, 1))
        softmax_sum = self.reshape(self.stack(softmax_sums), (n, t, 1))
        return softmax_max, softmax_sum

    def construct(self, softmax_max, softmax_sum, actual_seq_len):
        if self.is_dryrun:
            softmax_max = self.softmax_transpose(softmax_max, (1, 0, 2))
            softmax_sum = self.softmax_transpose(softmax_sum, (1, 0, 2))
            return softmax_max, softmax_sum
        if softmax_max.shape[-1] != 1:
            softmax_shape = softmax_max.shape[:-1]
            softmax_max = self.slice(softmax_max, (0, 0, 0), (*softmax_shape, 1), (1, 1, 1))
            softmax_sum = self.slice(softmax_sum, (0, 0, 0), (*softmax_shape, 1), (1, 1, 1))
        return self.convert_softmax(softmax_max, softmax_sum, actual_seq_len)

    def shard(self):
        self.slice.shard((layout("dp_cp", "tp", "None"),), (layout("dp_cp", "tp", "None"),))
        self.convert_softmax.shard(
            (layout("dp_cp", "tp", "None"), layout("dp_cp", "tp", "None"), layout("dp")),
            (layout("tp", "dp_cp", "None"), layout("tp", "dp_cp", "None"))
        )
        self.softmax_transpose.shard((layout("dp_cp", "tp", "None"),))
