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
"""DTensor in-place op compatibility helpers.

``hyper_parallel``'s ``DTensor.copy_`` requires ``src`` to be a ``DTensor`` on
the same mesh / placement as the destination. A plain ``Tensor`` ``src`` raises
``TypeError``. Many call sites here copy a locally-computed plain ``Tensor``
(an ``amax`` result, an ``op_cast`` output, a momentum update, ...) into a
``DTensor`` parameter or buffer; those previously relied on ``copy_`` accepting
a plain ``src`` and writing it into the local shard.

``inplace_copy`` restores that behaviour by performing the copy on the local
shard. It accepts either a ``DTensor`` or a plain ``Tensor`` for both ``dst``
and ``src``, so the result is identical to the old local-to-local copy and
introduces no numerical change.
"""
from hyper_parallel import DTensor

__all__ = ["inplace_copy"]


def inplace_copy(dst, src):
    """In-place copy ``src`` into ``dst`` on the local shard.

    ``dst`` / ``src`` may each be a ``DTensor`` or a plain ``Tensor``. When a
    side is a ``DTensor`` its local shard (``to_local()``, which returns the
    underlying storage) is used, so writes propagate back to the ``DTensor``.
    Returns ``dst``.
    """
    local_src = src.to_local() if isinstance(src, DTensor) else src
    local_dst = dst.to_local() if isinstance(dst, DTensor) else dst
    local_dst.copy_(local_src)
    return dst
