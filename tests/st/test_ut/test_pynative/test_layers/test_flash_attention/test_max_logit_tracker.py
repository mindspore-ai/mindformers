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
"""Tests for the per-head max attention logit tracking that feeds Muon's qk_clip."""
import numpy as np
import pytest
from mindspore import Tensor, context, nn
from mindspore import dtype as mstype

from mindformers.pynative.layers.max_logit_tracker import MaxAttentionLogitTracker


class _TrackerCell(MaxAttentionLogitTracker, nn.Cell):
    """Minimal cell that only does the tracking, so the reduction can be checked directly."""

    def __init__(self, head_num, enabled=True):
        super().__init__()
        self.head_num = head_num
        self.init_max_logits_tracking(enabled)


def _stats_with_distinct_head_maxima(shape, head_dim):
    """Statistics whose maximum along every non-head axis is a different value per head.

    Head ``h`` gets the values ``h * 100 + position``, so the per-head maximum is
    unmistakably ``h * 100 + (elements_per_head - 1)`` and any reduction that mixes heads
    lands on a value belonging to some other head.
    """
    head_num = shape[head_dim]
    head_first_shape = (head_num,) + tuple(
        size for dim, size in enumerate(shape) if dim != head_dim
    )
    per_head = int(np.prod(head_first_shape[1:]))
    head_first = np.stack([
        (head * 100 + np.arange(per_head, dtype=np.float32)).reshape(head_first_shape[1:])
        for head in range(head_num)
    ])
    expected = np.array([head * 100 + per_head - 1 for head in range(head_num)], dtype=np.float32)
    return Tensor(np.moveaxis(head_first, 0, head_dim).copy(), mstype.float32), expected


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    "shape, head_dim",
    [
        ((6, 4, 8), 1),      # ordinary FlashAttention, TND: (T, N, G)
        ((2, 4, 6, 8), 1),   # ordinary FlashAttention, BNSD: (B, N, S, G)
        ((4, 6, 1), 0),      # DSA dense after from_cp_layout, TND: (N, T, 1)
        ((2, 4, 6, 1), 1),   # DSA dense after from_cp_layout, BSND: (B, N, S, 1)
        ((6, 5, 4), -1),     # DSA sparse, head last
    ],
)
def test_update_max_logits_keeps_each_head_separate(shape, head_dim):
    """Every head's maximum must be its own, for each layout a caller passes.

    This is the property the whole module exists for: reducing over the wrong axes raises
    nothing and simply maxes across heads, so only a per-head check can catch it.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    head_num = shape[head_dim]
    cell = _TrackerCell(head_num)
    stats, expected = _stats_with_distinct_head_maxima(shape, head_dim % len(shape))

    cell.update_max_logits(stats, head_dim)

    np.testing.assert_allclose(cell.max_logits_val.asnumpy(), expected, rtol=0, atol=0)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_update_max_logits_accumulates_a_running_max():
    """Successive steps keep the largest value seen, never the latest one."""
    context.set_context(mode=context.PYNATIVE_MODE)
    cell = _TrackerCell(2)

    cell.update_max_logits(Tensor(np.array([[[3.0], [1.0]]], np.float32), mstype.float32), 1)
    cell.update_max_logits(Tensor(np.array([[[2.0], [5.0]]], np.float32), mstype.float32), 1)

    np.testing.assert_allclose(cell.max_logits_val.asnumpy(), np.array([3.0, 5.0], np.float32))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_reset_parameter_clears_the_running_max():
    """``GPTModel.reset_max_attention_logit`` drives this once per step."""
    context.set_context(mode=context.PYNATIVE_MODE)
    cell = _TrackerCell(2)
    cell.update_max_logits(Tensor(np.array([[[3.0], [5.0]]], np.float32), mstype.float32), 1)

    cell.reset_parameter()

    np.testing.assert_allclose(cell.max_logits_val.asnumpy(), np.zeros(2, np.float32))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tracking_off_registers_no_buffer():
    """``GPTModel`` collects tracked layers by ``hasattr``, so an untracked cell must have none."""
    context.set_context(mode=context.PYNATIVE_MODE)
    cell = _TrackerCell(2, enabled=False)

    assert not hasattr(cell, "max_logits_val")
    cell.reset_parameter()  # must stay a no-op rather than create the buffer
    assert not hasattr(cell, "max_logits_val")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_buffer_follows_this_rank_s_local_head_count():
    """A buffer wider than the kernel's local head count is a silent qk_clip failure.

    ``MLASelfAttention._clip_projection`` derives ``shards = row_scales.shape[0] //
    local_weight_rows`` and indexes by the weight's split id. If the buffer spans every head
    while the kernel only ever fills the rows for this rank's partition, ``shards`` comes out
    greater than one and the ranks that own a later partition read rows nobody wrote --
    zeros, which clamp to a scale of 1, i.e. no clipping at all. Measured on 8 cards with
    ``tp=2`` and 128 heads: the buffer stayed 128 wide while the kernel reported 64 heads,
    and half of every rank's vector stayed zero. ``head_num`` is therefore read live (it is
    rewritten to the local count after ``__init__``) and the buffer resizes to whatever the
    kernel actually returns.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    cell = _TrackerCell(4)  # global head count, as a config would give it
    # The kernel reports two heads: this rank owns half of them.
    stats = Tensor(np.array([[[1.0], [2.0]]], np.float32), mstype.float32)

    cell.update_max_logits(stats, 1)

    np.testing.assert_allclose(cell.max_logits_val.asnumpy(), np.array([1.0, 2.0], np.float32))
    cell.head_num = 1  # Parallelization may rewrite the head count before delayed initialization.
    cell.reset_parameter()
    assert cell.max_logits_val.shape == (1,)
    np.testing.assert_array_equal(cell.max_logits_val.asnumpy(), np.zeros(1, np.float32))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('sparse,tnd,shape,axis', [
    (True, True, (1, 6, 4), 2), (True, False, (2, 3, 5, 4), 3),
    (False, True, (4, 6, 1), 0), (False, False, (2, 4, 6, 1), 1)])
def test_dsa_construct_tracks_kernel_statistics(sparse, tnd, shape, axis):
    """Exercise DSAttention.construct so a missing hook or a wrong head axis fails."""
    from types import SimpleNamespace
    from mindformers.parallel_core.transformer_config import MLATransformerConfig
    from mindformers.pynative.transformers.experimental_attention_variant.dsa import DSAttention, DSAttentionSubmodules

    stats, expected = _stats_with_distinct_head_maxima(shape, axis)

    class Kernel(nn.Cell):
        """Return distinguishable per-head statistics without invoking attention kernels."""
        def __init__(self):
            super().__init__()
            self.softmax_converter = SimpleNamespace(from_cp_layout=lambda a, b: (a, b))

        def construct(self, *args, **_kwargs):
            return args[0], stats, stats

    config = MLATransformerConfig(
        num_layers=1, hidden_size=16, num_attention_heads=4, seq_length=2048,
        experimental_attention_variant='dsa', dsa_indexer_use_sparse_loss=sparse,
        kv_lora_rank=512, qk_head_dim=128, qk_pos_emb_head_dim=64, v_head_dim=128,
        dsa_indexer_n_heads=8, dsa_indexer_head_dim=128, dsa_indexer_topk=2048,
        use_eod_attn_mask_compression=tnd, track_max_attention_logit=True)
    cell = DSAttention(config, DSAttentionSubmodules(), layer_number=0, is_index_leader=False)
    if sparse:
        cell.sparse_flash_attention = Kernel()
    else:
        cell.dense_flash_attention = Kernel()
    values = Tensor(np.zeros((6, 4, 576)), mstype.float32)
    cell(values, values, values, None)
    np.testing.assert_array_equal(cell.max_logits_val.asnumpy(), expected)
