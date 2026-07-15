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
"""experimental_attention_variant utils."""
from scipy.linalg import hadamard

from mindspore import ops
from mindspore.mint.distributed import get_world_size, all_reduce

from hyper_parallel.core.dtensor.dtensor import DTensor

from mindspore import nn, Tensor, mint
import mindspore.common.dtype as mstype


_INDEXER_LOSS_LOGGING_TRACKER: dict = {}


def get_indexer_loss_tracker() -> dict:
    """Return the module-level indexer loss tracker."""
    # pylint: disable=W0602
    global _INDEXER_LOSS_LOGGING_TRACKER
    return _INDEXER_LOSS_LOGGING_TRACKER


def save_to_indexer_losses_tracker(loss, layer_number, num_layers):
    """Record one layer's indexer loss into the module-level tracker."""
    if layer_number is None:
        return
    tracker = get_indexer_loss_tracker()
    if not tracker:
        tracker["values"] = mint.zeros(num_layers)
        tracker["is_indexer"] = [False] * num_layers
    if isinstance(loss, DTensor):
        loss = loss.to_local()
    if hasattr(loss, "detach"):
        loss = loss.detach()
    tracker["values"][layer_number - 1] += loss
    tracker["is_indexer"][layer_number - 1] = True


def clear_indexer_losses_tracker() -> None:
    """Clear the indexer losses."""
    tracker = get_indexer_loss_tracker()
    for value in tracker["values"]:
        value.zero_()


def track_indexer_metrics(group=None, group_size=None, pp_group=None,
                          pp_group_size=None, has_last=True):
    """Track sparse-attention indexer loss across data and pipeline axes.

    Every PP stage participates, including stages without an indexer.  Values
    and the number of contributing indexer layers are first summed over PP so
    the last stage can report the same layer average as a non-pipelined model.
    The resulting average is then reduced over the data/loss mesh.
    """
    tracker = get_indexer_loss_tracker()
    if "values" in tracker:
        indexer_loss_sum = tracker["values"].sum()
        num_indexer_layers = sum(tracker["is_indexer"])
    else:
        indexer_loss_sum = mint.zeros((), dtype=mstype.float32)
        num_indexer_layers = 0
    indexer_layer_count = Tensor(float(num_indexer_layers), dtype=mstype.float32)

    if pp_group is not None and (pp_group_size is None or pp_group_size > 1):
        all_reduce(indexer_loss_sum, op=ops.ReduceOp.SUM, group=pp_group)
        all_reduce(indexer_layer_count, op=ops.ReduceOp.SUM, group=pp_group)

    if "values" in tracker:
        clear_indexer_losses_tracker()
    pp_active = pp_group is not None and (pp_group_size is None or pp_group_size > 1)
    if (pp_active and not has_last) or float(indexer_layer_count.asnumpy()) == 0.0:
        return None

    avg_indexer_loss = indexer_loss_sum / indexer_layer_count
    if group is not None and (group_size is None or group_size > 1):
        all_reduce(avg_indexer_loss, op=ops.ReduceOp.SUM, group=group)
        if group_size is None:
            group_size = get_world_size(group)
        avg_indexer_loss /= group_size
    return avg_indexer_loss


class Hadamard(nn.Cell):
    """Hadamard Transform."""
    def __init__(self, head_dim):
        super().__init__()
        self.hadamard_mat = None
        self.linear = mint.nn.functional.linear
        self.head_dim = head_dim

    def construct(self, x):
        """do hadamard transform."""
        if self.hadamard_mat is None:
            self.hadamard_mat = Tensor(hadamard(self.head_dim), mstype.float32)
        x = self.linear(x, self.hadamard_mat.astype(x.dtype))
        x = x * self.head_dim ** -0.5
        return x
