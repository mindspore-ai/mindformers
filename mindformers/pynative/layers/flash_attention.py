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
"""Flash Attention Layer"""
__all__ = ['FlashAttention']

import math
from typing import Optional, Union

import mindspore.common.dtype as mstype
import mindspore as ms
from hyper_parallel import SkipDTensorDispatch
from hyper_parallel import DTensor
from hyper_parallel.core.dtensor.layout import _infer_slice_area_by_rank
from mindspore import ops, mint, Parameter
from mindspore.common.tensor import Tensor
from mindspore.communication import get_rank
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from mindformers.parallel_core.transformer_config import TransformerConfig, MLATransformerConfig


def _local_head_slice(softmax_val, head_dim=1):
    """Return this rank's ``(begin, end)`` slice of the GLOBAL head dim, or ``None``.

    Under Ulysses (or hybrid) context parallelism the seq->head all-to-all leaves
    ``softmax_val`` as a DTensor sharded along the head dim across the cp ranks,
    so the local ``amax`` only covers this rank's heads while ``max_logits_val``
    holds every head. The slice bounds are derived from the DTensor layout itself
    (same machinery ``distribute_tensor(...).to_local()`` uses), so any head
    sharding scheme is handled without knowledge of the CP style internals.
    Returns ``None`` when ``softmax_val`` is not a DTensor (colossal CP / no CP),
    where the local head count already equals the parameter length.
    """
    if not isinstance(softmax_val, DTensor):
        return None
    layout = softmax_val.layout
    rank_list = tuple(int(r) for r in layout.rank_list)
    rank = int(get_rank())
    if rank not in rank_list:
        return None
    local_shape = tuple(int(d) for d in softmax_val.to_local().shape)
    full_shape = tuple(int(d) for d in layout.get_global_shape(local_shape))
    slice_area = _infer_slice_area_by_rank(
        tuple(int(d) for d in layout.mesh_shape),
        list(layout.tensor_map),
        rank_list.index(rank),
        full_shape,
    )
    begin, end = slice_area[head_dim]
    return int(begin), int(end)


class FlashAttention(Cell):
    """
    FlashAttention Layer.

    This class implements the FlashAttention mechanism for fast and memory-efficient attention computation.
    It supports multiple attention types, mask modes, and is optimized for parallel training including
    tensor and context parallelism.

    Reference:
        "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
        https://arxiv.org/abs/2205.14135

    Args:
        config (Union[TransformerConfig, MLATransformerConfig]): Configuration object containing model hyperparameters,
            including number of heads, and more.
        layer_number (int): The index of the current layer within the transformer stack.
        softmax_scale (float, optional): Scaling factor for the attention logits before softmax.
            If None, it defaults to 1 / sqrt(head_dim).

    Inputs:
        - **query** (Tensor): The query tensor with shape (B, S1, H1) or (B, N1, S1, D).
        - **key** (Tensor): The key tensor with shape (B, S2, H2) or (B, N2, S2, D).
        - **value** (Tensor): The value tensor with shape (B, S2, H2) or (B, N2, S2, D).
        - **attn_mask** (Tensor): Attention mask. A value of 0 keeps the element;
          a value of 1 masks it out. Shape can vary based on attention mode.
        - **alibi_mask** (Tensor, optional): Positional bias tensor for ALiBi attention.
          Used for large sequences and causal masks.
        - **prefix** (Tensor, optional): Prefix lengths for prefix attention mode.
          Not implemented yet.
        - **padding_mask** (None): Reserved for future use.
        - **actual_seq_qlen** (Tensor[int32], optional): Actual valid sequence lengths of the query.
        - **actual_seq_kvlen** (Tensor[int32], optional): Actual valid sequence lengths of the key/value.

    Outputs:
        - **attention_out** (Tensor): The attention output tensor with the same shape and type as `query`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self,
                 config: Union[TransformerConfig, MLATransformerConfig],
                 layer_number,
                 softmax_scale: Optional[float] = None,
                 ):
        super().__init__()

        # FA (Flash Attention) is an optimized version of DotProductAttention in Megatron v0.12.0,
        # with nearly identical computational precision.

        self.config = config
        self.layer_number = max(1, layer_number)

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        if config.multi_latent_attention:
            hidden_size_per_attention_head = config.qk_head_dim + config.qk_pos_emb_head_dim
        else:
            hidden_size_per_attention_head = projection_size // config.num_attention_heads

        # MindSpore FlashAttentionScore
        self.head_num = config.num_attention_heads
        self.input_layout = config.input_layout
        self.sparse_mode = config.sparse_mode
        self.pre_tokens = 2147483647
        self.next_tokens = 0
        self.scalar_value = 1. / math.sqrt(hidden_size_per_attention_head) if softmax_scale is None else softmax_scale
        self.inner_precise = 0

        self.flash_attention = FlashAttentionScore(
            head_num=self.head_num,
            scale_value=self.scalar_value,
            pre_tokens=self.pre_tokens,
            next_tokens=self.next_tokens,
            inner_precise=self.inner_precise,
            input_layout=self.input_layout,
            sparse_mode=self.sparse_mode,
        )

        # Note: only support config.apply_query_key_layer_scaling be set False
        # FusedScaleMaskSoftmax does not require implementation.

        self.use_alibi_mask = config.use_alibi_mask

        if self.use_alibi_mask:
            self.alibi_rescale_mul = mint.mul

        self.bnsd_transpose = mint.permute
        self.bsh_transpose = mint.permute
        self.merge_head_transpose = mint.permute
        self.reshape = mint.reshape
        self.fa_out_transpose = mint.permute
        self.cast = ops.cast

        self.track_max_attention_logit = getattr(config, "track_max_attention_logit", False)
        if self.track_max_attention_logit:
            self.max_logits_val = Parameter(
                mint.empty((self.head_num,), dtype=mstype.float32),
                requires_grad=False
            )
            self.amax = mint.amax
            self.maximum = mint.maximum

    def construct(self,
                  query: Tensor,
                  key: Tensor,
                  value: Tensor,
                  attention_mask: Tensor,
                  alibi_mask=None,
                  prefix=None,
                  padding_mask=None,
                  actual_seq_qlen=None,
                  actual_seq_kvlen=None):
        """Forward process of the AttentionMaskMF"""
        if attention_mask is not None:
            attention_mask = self.cast(attention_mask, ms.uint8)

        if self.input_layout == "TND":
            softmax_val, _, _, output = self.flash_attention(query=query,
                                                             key=key,
                                                             value=value,
                                                             real_shift=alibi_mask,
                                                             padding_mask=padding_mask,
                                                             attn_mask=attention_mask,
                                                             prefix=prefix,
                                                             actual_seq_qlen=actual_seq_qlen,
                                                             actual_seq_kvlen=actual_seq_kvlen)
            if self.track_max_attention_logit:
                self._update_max_logits(softmax_val, (0, 2), running=False)
            return output

        input_already_in_layout = bool(getattr(self, "_mf_runtime_input_already_in_fa_layout", False))
        if self.input_layout == "BSH" and input_already_in_layout:
            bsz, q_seq_len = query.shape[:2]
            kv_seq_len = key.shape[1]
        elif self.input_layout == "BNSD" and input_already_in_layout:
            bsz = query.shape[0]
            q_seq_len = query.shape[2]
            kv_seq_len = key.shape[2]
        else:
            q_seq_len, bsz = query.shape[:2]
            kv_seq_len = key.shape[0]
        if self.input_layout == "BNSD":
            if not input_already_in_layout:
                query = self.bnsd_transpose(query, (1, 2, 0, 3))
                key = self.bnsd_transpose(key, (1, 2, 0, 3))
                value = self.bnsd_transpose(value, (1, 2, 0, 3))
        elif self.input_layout == "BSH":
            if not input_already_in_layout:
                query = self.bsh_transpose(query, (1, 0, 2))
                key = self.bsh_transpose(key, (1, 0, 2))
                value = self.bsh_transpose(value, (1, 0, 2))
        else:
            query = self.reshape(query, (q_seq_len, bsz, -1))
            key = self.reshape(key, (kv_seq_len, bsz, -1))
            value = self.reshape(value, (kv_seq_len, bsz, -1))
        if hasattr(query, "contiguous"):
            query = query.contiguous()
        if hasattr(key, "contiguous"):
            key = key.contiguous()
        if hasattr(value, "contiguous"):
            value = value.contiguous()
        if self.use_alibi_mask:
            alibi_rescale_factor = Tensor([1.0 / self.scalar_value], dtype=mstype.float16)
            alibi_mask = self.alibi_rescale_mul(alibi_mask, self.cast(alibi_rescale_factor, alibi_mask.dtype))

        softmax_val, _, _, output = self.flash_attention(query=query,
                                                         key=key,
                                                         value=value,
                                                         real_shift=alibi_mask,
                                                         padding_mask=padding_mask,
                                                         attn_mask=attention_mask,
                                                         prefix=prefix,
                                                         actual_seq_qlen=actual_seq_qlen,
                                                         actual_seq_kvlen=actual_seq_kvlen)
        if self.track_max_attention_logit:
            self._update_max_logits(softmax_val, (0, 2, 3), running=True)

        if self.input_layout == "BNSD":
            if input_already_in_layout:
                pass
            else:
                output = self._merge_heads(output)
        elif self.input_layout == "BSH":
            if not input_already_in_layout:
                output = self.fa_out_transpose(output, (1, 0, 2))
        return output

    def _merge_heads(self, x):
        """
        Convert a 4D input tensor to a 3D output tensor.

        Inputs:
            x: input tensor

        Output:
            x_merge: the 3D output tensor
        """
        x = self.merge_head_transpose(x, (0, 2, 1, 3))  # dp,tp,cp,1 -> dp,cp,tp,1
        bs, seq_len, n_head, head_dim = x.shape
        new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = self.reshape(x, new_shape)
        x_merge = self.fa_out_transpose(x_merge, (1, 0, 2))
        return x_merge

    def _update_max_logits(self, softmax_val, reduce_dims, running):
        """Track per-head max attention logits for qk_clip; context-parallel aware.

        Colossal CP shards the SEQ dim: ``amax`` inside ``SkipDTensorDispatch``
        yields this rank's local chunk-max for EVERY head (same length as
        ``max_logits_val``) — the plain path below. Ulysses / hybrid CP instead
        shard the HEAD dim (seq->head all-to-all), so the local ``amax`` covers
        only this rank's heads; those are written into the matching slice of
        ``max_logits_val`` (slice bounds derived from ``softmax_val``'s DTensor
        layout). In both cases the dp x cp ``all_reduce(MAX)`` in
        ``synced_max_attention_logit_fires`` later merges the per-rank values
        (non-owned head entries stay at their reset value 0, and 0 never exceeds
        the clip threshold) into the full per-head vector, so the fire check and
        clip scales always see every head.
        """
        head_slice = _local_head_slice(softmax_val)
        # ``max_logits_val`` may be head-sharded over the tensor-parallel group
        # (parallelize.py:562) or replicated. Everything inside the
        # ``SkipDTensorDispatch`` block below operates on the LOCAL shard, but
        # ``.shape[0]`` is the GLOBAL head count when the parameter is sharded.
        # Comparing the local ``amax`` head count against the global length made
        # the TP-sharded case wrongly take the head-slice path and index the
        # local shard with GLOBAL coordinates -> ``max_logits_val[16:32]`` on a
        # length-16 local shard = empty ``[0]`` -> ``Maximum`` broadcast crash.
        # Compare against the LOCAL length (a sharded parameter exposes its
        # per-rank shard via ``local_shape``) so the TP-sharded case (local heads
        # == local amax) takes the safe local-to-local path; the replicated /
        # context-parallel head-shard case is unchanged (local length == global).
        mlv_local_shape = getattr(self.max_logits_val, "local_shape", None)
        n_param = int(mlv_local_shape[0]) if mlv_local_shape is not None \
            else int(self.max_logits_val.shape[0])
        with SkipDTensorDispatch():
            max_logits = self.amax(softmax_val, dim=reduce_dims, keepdim=False)
            n_local = int(max_logits.shape[0])
            if n_local == n_param:
                if running:
                    max_logits = self.maximum(self.max_logits_val, max_logits)
                self.max_logits_val.copy_(max_logits.detach())
                return
            if head_slice is None or head_slice[1] - head_slice[0] != n_local:
                raise ValueError(
                    f"max_logits tracking: local head count ({n_local}) does not match "
                    f"max_logits_val length ({n_param}), and softmax_val carries no "
                    f"matching head-shard layout (head_slice={head_slice})."
                )
            begin, end = head_slice
            if running:
                max_logits = self.maximum(self.max_logits_val[begin:end], max_logits)
            self.max_logits_val[begin:end] = max_logits.detach()

    def reset_parameter(self):
        """Reset FlashAttention parameters for delayed initialization."""
        if self.track_max_attention_logit and hasattr(self, 'max_logits_val'):
            self.max_logits_val.zero_()
