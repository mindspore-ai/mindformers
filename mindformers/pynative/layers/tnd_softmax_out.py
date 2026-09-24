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
"""TND-ordered FlashAttention softmax statistics, via hyper-parallel's aclnn V4 kernel.

Under TND the stock kernel (aclnn v1) writes its softmax statistics in **NTD** order --
segmented per document, head-major inside a segment -- while the tensor it returns is
*declared* ``(T, N, G)``. Every consumer that indexes that tensor by its declared shape is
therefore wrong, silently: the element count matches, nothing raises, and only whatever
reads the statistics comes out wrong. aclnn V4 takes a ``softmaxOutLayout`` argument that
makes the statistics come out in the declared order instead, which removes the mismatch at
the source rather than asking each consumer to repack.

Two callers need this, so it lives here rather than in either of them: DSA's dense warm-up
(whose indexer loss consumes the statistics) and the ordinary FlashAttention layer (whose
per-head max logit feeds Muon's qk_clip).
"""
from mindformers.tools import logger

__all__ = [
    "tnd_softmax_out_available",
    "request_tnd_softmax_out",
    "local_v4_varlen_flash_attention",
]

# The two symbols the aclnn V4 flash-attention change added to hyper-parallel: the
# process-wide switch that makes its dispatched operator use V4, and the operator itself,
# which callers use directly where nothing dispatches. Imported at module load and not
# inside the call so that an older hyper-parallel fails while the config is being built
# rather than mid-forward.
try:
    from hyper_parallel.core.shard.ops.parallel_ms_flash_attention_score import set_tnd_softmax_out
except ImportError:
    set_tnd_softmax_out = None
try:
    from hyper_parallel.custom_ops.experimental import npu_flash_attention_varlen_v4
except ImportError:
    npu_flash_attention_varlen_v4 = None


# The hyper-parallel switch this drives is process-wide, while the decision to use it is per
# cell. Record what each cell asks for, and who asked, so a disagreement fails loudly here
# instead of silently feeding V4-ordered statistics to a consumer still expecting v1's order
# (element counts match either way, so nothing would raise).
_TND_SOFTMAX_OUT_REQUEST = {"value": None, "requester": ""}


def tnd_softmax_out_available() -> bool:
    """Whether this hyper-parallel build provides the aclnn V4 flash-attention change."""
    return set_tnd_softmax_out is not None and npu_flash_attention_varlen_v4 is not None


def request_tnd_softmax_out(enabled: bool, requester: str = "") -> None:
    """Set hyper-parallel's process-wide TND softmax switch, refusing conflicting requests."""
    previous = _TND_SOFTMAX_OUT_REQUEST["value"]
    if previous is not None and previous != enabled:
        raise ValueError(
            "TND-ordered softmax statistics must agree across every attention cell in a "
            f"process: {_TND_SOFTMAX_OUT_REQUEST['requester'] or 'an earlier cell'} already "
            f"configured it as {previous}, now {requester or 'another cell'} asks for "
            f"{enabled}. The switches that reach here are `dsa_dense_fa_tnd_softmax` (DSA's "
            "dense warm-up) and Muon's qk_clip on an ordinary TND FlashAttention layer."
        )
    # A hyper-parallel without the aclnn V4 flash-attention change exports neither symbol, and
    # this configuration then has no way to be honoured: refuse it by name here, while the
    # model is being built. Silently falling back to the stock kernel would be the very bug
    # this path was fixed for -- v1-ordered statistics read as if they were TND-ordered.
    if enabled and not tnd_softmax_out_available():
        missing = [
            name for name, symbol in (
                ("set_tnd_softmax_out", set_tnd_softmax_out),
                ("npu_flash_attention_varlen_v4", npu_flash_attention_varlen_v4),
            ) if symbol is None
        ]
        raise ValueError(
            f"TND-ordered softmax statistics ({requester or 'requested'}) are not supported "
            f"by this hyper-parallel build: it does not provide {', '.join(missing)} (the "
            "aclnn V4 flash-attention change). Upgrade hyper-parallel, or turn the requesting "
            "switch off."
        )
    if set_tnd_softmax_out is not None:
        set_tnd_softmax_out(enabled)
    _TND_SOFTMAX_OUT_REQUEST["value"] = enabled
    if enabled:
        _TND_SOFTMAX_OUT_REQUEST["requester"] = requester
        logger.info("TND-ordered softmax statistics enabled by %s.", requester or "an attention cell")


def local_v4_varlen_flash_attention(
        query, key, value, attention_mask, actual_seq_qlen, actual_seq_kvlen,
        softmax_scale, head_num, sparse_mode,
):
    """Run the aclnn V4 varlen kernel directly, on plain tensors.

    Callers reach this only where nothing dispatches (``cp == 1``, so the input is a plain
    tensor rather than a ``DTensor``): with a context-parallel boundary hyper-parallel swaps
    the same kernel in inside its own distributed operator, and going around that would skip
    the CP folding / load balancing / ``actual_seq_len`` adjustment living there.
    """
    softmax_max, softmax_sum, attention_output = npu_flash_attention_varlen_v4(
        query, key, value,
        atten_mask=attention_mask,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        scale_value=softmax_scale,
        head_num=int(head_num),
        sparse_mode=int(sparse_mode),
        pre_tokens=2147483647,
        next_tokens=0,
        inner_precise=0,
        tnd_softmax_out=True,
    )
    return attention_output, softmax_max, softmax_sum
