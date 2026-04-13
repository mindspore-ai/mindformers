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
import mindspore.ops.functional as F
from mindspore import dtype as mstype


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
    return new_actual_seq_klen, new_actual_seq_klen


def adjust_bsnd_input(offset_id, q, x):
    """adjust bsnd input for dsa operators"""
    slice_sq = q.shape[1]
    offset_q = slice_sq * (offset_id + 1)
    b, _, n, d = x.shape
    x = F.strided_slice(x, (0, 0, 0, 0), (b, offset_q, n, d), (1, 1, 1, 1))
    return x
