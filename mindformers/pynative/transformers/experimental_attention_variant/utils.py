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


def track_indexer_metrics():
    """Track the sparse attention indexer metrics for logging."""
    tracker = get_indexer_loss_tracker()
    if "values" not in tracker:
        return None

    indexer_loss_values = tracker["values"]
    num_indexer_layers = sum(tracker["is_indexer"])
    if num_indexer_layers == 0:
        return None

    # Average across layers that actually own an indexer;
    # layers without one contribute zero in `tracker["values"]` so they must not be in the divisor.
    avg_indexer_loss = indexer_loss_values.sum() / num_indexer_layers

    if get_world_size() > 1:
        all_reduce(avg_indexer_loss, op=ops.ReduceOp.SUM)
        avg_indexer_loss /= get_world_size()
    clear_indexer_losses_tracker()

    return avg_indexer_loss


class Hadamard(nn.Cell):
    """Hadamard Transform."""
    def __init__(self, head_dim):
        super().__init__()
        self.hadamard_mat = None
        self.bmm = mint.matmul
        self.head_dim = head_dim

    def construct(self, x):
        """do hadamard transform."""
        if self.hadamard_mat is None:
            self.hadamard_mat = Tensor(hadamard(self.head_dim) * self.head_dim ** -0.5, mstype.float32)
        x = self.bmm(x, self.hadamard_mat.astype(x.dtype))
        return x
