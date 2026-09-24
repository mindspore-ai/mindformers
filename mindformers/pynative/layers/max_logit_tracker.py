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
"""Per-head max attention logit tracking, shared by every attention cell that feeds qk_clip.

``GPTModel`` finds the tracked layers by ``hasattr(core_attention, "max_logits_val")`` and
resets them through ``core_attention.reset_parameter()``, so an attention cell that wants
Muon's qk_clip only has to mix this in and say which axis of its statistics carries the head.

That axis is the one thing every caller must get right, and it differs per kernel and
layout: ordinary FlashAttention returns ``(T, N, G)`` / ``(B, N, S, G)``, DSA's dense
statistics come back from :class:`DSASoftmaxConverter` head-first, and DSA's sparse
statistics carry the head last. Hence the API takes ``head_dim`` and derives the reduction
itself -- reducing over the wrong axes raises nothing, it just silently maxes across heads,
which is exactly the failure this module exists to avoid.
"""
from mindspore import mint, ops
from mindspore import dtype as mstype

from mindformers.pynative.dtensor_compat import local_shard, to_local

__all__ = ["MaxAttentionLogitTracker"]


class MaxAttentionLogitTracker:
    """Mixin holding the per-head running max of the attention logits, for Muon's qk_clip."""

    def init_max_logits_tracking(self, enabled: bool):
        """Arm tracking for this cell. The buffer is sized from ``self.head_num``."""
        self.track_max_attention_logit = bool(enabled)
        if self.track_max_attention_logit:
            self._init_max_logits_val()

    def _init_max_logits_val(self, head_num=None):
        """(Re)create the per-head max-logit buffer, sized to this rank's head partition.

        ``self.head_num`` is read **live** and never cached: parallelization rewrites it to
        this rank's local head count after ``__init__``, and ``reset_parameter`` runs after
        that, so a cached copy would keep the buffer at the global count. A buffer wider than
        the local partition is not merely wasteful -- ``_clip_projection`` then computes
        ``shards > 1`` and indexes by the weight's split id, reading rows nobody filled.
        """
        self.register_buffer(
            "max_logits_val",
            # Recomputed from scratch every step, so it is not model state: non-persistent
            # keeps it out of the checkpoint instead of pinning a stale max into it.
            mint.zeros((int(head_num if head_num is not None else self.head_num),),
                       dtype=mstype.float32),
            persistent=False,
        )

    def update_max_logits(self, softmax_val, head_dim: int):
        """Reduce every non-head axis and accumulate this rank's per-head maxima.

        Size the buffer from the kernel output so TP-local statistics cannot leave
        unfilled rows that would silently disable clipping on other head partitions.
        """
        head_dim = head_dim % softmax_val.ndim
        reduce_dims = tuple(dim for dim in range(softmax_val.ndim) if dim != head_dim)
        max_logits = mint.amax(to_local(softmax_val), dim=reduce_dims, keepdim=False)
        local_heads = int(max_logits.shape[0])
        if int(self.max_logits_val.shape[0]) != local_heads:
            # The buffer must hold exactly this rank's heads: ``_clip_projection`` derives
            # ``shards`` from its width against the local weight rows, so a wider buffer makes
            # it index rows nobody fills. A cell whose ``head_num`` parallelization never
            # rewrote (DSA reads it straight from the config) would otherwise keep the global
            # count. Size it to what the kernel actually returns.
            self._init_max_logits_val(local_heads)
        tracked = self.max_logits_val.view(-1, local_heads)
        mine = local_shard(softmax_val, head_dim, int(tracked.shape[0]))
        # Support both plain tensors and the wrappers used by distributed training.
        tracked[mine] = mint.maximum(tracked[mine], ops.stop_gradient(max_logits))

    def reset_parameter(self):
        """Recreate the max-logit buffer: delayed init, and the per-step qk_clip reset."""
        if getattr(self, "track_max_attention_logit", False):
            self._init_max_logits_val()
