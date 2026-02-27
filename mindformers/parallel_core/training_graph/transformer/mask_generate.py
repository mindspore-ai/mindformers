# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Attention Mask Generate"""
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.ops import operations as P
from mindspore.ops import auto_generate as aclnn_ops
from mindspore.context import ParallelMode
import mindspore.common.dtype as mstype
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.device_matrix import layout
from mindformers.parallel_core.training_graph.communication import get_dp_cp_id


class CausalMaskGenerate(nn.Cell):
    """Get the upper triangular matrix from the input_ids.

    Args:
        seq_length (int): The length of the input sequence.
        compute_type (mstype): The compute type of the input tensor. Default: mstype.float16.
        is_dynamic (bool): Whether the input_ids is dynamic. Default: False.
        pad_token_id (int): The pad token id. Default: 0.
        use_flash_attention (bool): Whether to use the flash attention. Default: False.
        use_prompt_flash_attention (bool): Whether to use the prompt flash attention. Default: False.
        use_incre_flash_attention (bool): Whether to use the incremental flash attention. Default: False.
        use_attn_mask_compression (bool): Whether to use the attention mask compression. Default: False.
    """

    def __init__(self,
                 seq_length: int,
                 compute_type: mstype = mstype.float16,
                 is_dynamic: bool = False,
                 pad_token_id: int = 0,
                 use_flash_attention: bool = False,
                 use_prompt_flash_attention: bool = False,
                 use_incre_flash_attention: bool = False,
                 use_attn_mask_compression: bool = False,
                 config: TransformerConfig = None
                 ):
        super().__init__()
        self.dtype = compute_type
        self.is_dynamic = is_dynamic
        self.pad_token_id = pad_token_id
        self.use_flash_attention = use_flash_attention
        self.use_attn_mask_compression = use_attn_mask_compression
        self.seq_length = seq_length
        self.use_prompt_flash_attention = use_prompt_flash_attention
        self.use_incre_flash_attention = use_incre_flash_attention
        self.is_first_iteration = True
        self.multiply_data = Tensor([-10000.0], dtype=compute_type)
        self.one = Tensor([1.0], dtype=compute_type)
        if use_attn_mask_compression:
            if seq_length < 2048:
                raise ValueError("seq_length should be larger than 2048 when use mask_compression")
            self.lower_triangle_mask = ms.Tensor(np.triu(np.ones((2048, 2048), dtype=np.int8), k=1), dtype=ms.uint8)
        else:
            self.lower_triangle_mask = Tensor(np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8)),
                                              dtype=compute_type)
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.not_equal = P.NotEqual()
        self.expand_dim = P.ExpandDims()
        self.slice = P.StridedSlice()
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.expand_dim_post = P.ExpandDims()
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(config)
        else:
            self.shard(config)

    def construct(self, tokens=None, masks=None):
        """Forward process of the CausalMask

        Args:
            tokens (Tensor): The input tokens. Default: None.
            masks (Tensor): The input masks. Default: None.

        Returns:
            Tensor, the upper triangle attention mask carrying 0 and 1 values
        """
        if self.use_attn_mask_compression:
            attention_mask = self.lower_triangle_mask
            return attention_mask
        if tokens is not None:
            bs = self.shape(tokens)[0]
            seq_len = self.shape(tokens)[1]
            input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), self.dtype)
        else:
            bs = self.shape(masks)[0]
            seq_len = self.shape(masks)[1]
            input_mask = self.cast(masks, self.dtype)
        shape_right = (bs, 1, seq_len)

        # Mask the padded inputs
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = mask_right
        if not self.is_dynamic:
            lower_triangle = self.expand_dim(self.lower_triangle_mask, 0)
        else:
            lower_triangle_mask = self.slice(self.lower_triangle_mask, (0, 0), (seq_len, seq_len), (1, 1))
            lower_triangle = self.expand_dim(lower_triangle_mask, 0)

        # the returned shape is [bs, 1, seq_length, seq_length]
        attention_mask = self.mul(attention_mask, lower_triangle)
        attention_mask = self.sub(self.one, attention_mask)
        attention_mask = self.expand_dim_post(attention_mask, 1)
        if self.use_flash_attention or self.use_prompt_flash_attention:
            attention_mask = self.cast(attention_mask, mstype.uint8)
        return attention_mask

    def shard(self, config: TransformerConfig):
        """sharding operators
        """
        dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        self.not_equal.shard(((dp, 1), ()))
        self.expand_dim.shard(((1, 1),))
        self.mul.shard(((dp, 1, 1), (1, 1, 1)))
        self.sub.shard(((1,), (dp, 1, 1)))
        self.expand_dim_post.shard(((dp, 1, 1),))

    def sharding_propagation(self, config: TransformerConfig):
        pass


class CausalEODMaskGenerate(nn.Cell):
    """
    Get the upper triangular eod matrix from the actual sequence length. Only support self-attention.

    Args:
        config (TransformerConfig): The transformer configuration.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.seq_length = config.seq_length
        self.dtype = config.compute_dtype

        _, cp_id = get_dp_cp_id(config)
        self.cp = config.context_parallel_size
        self.cp_id = cp_id

        self.is_first_iteration = True
        self.one = Tensor([1.0], dtype=self.dtype)
        self.clamp= aclnn_ops.ClampScalar()
        self.slice = aclnn_ops.SliceExt()
        self.cast = aclnn_ops.Cast()
        self.reshape = aclnn_ops.Reshape()
        self.sub = aclnn_ops.SubExt()
        self.tril = aclnn_ops.TrilExt()
        self.roll = aclnn_ops.Roll(1, 1)
        self.arange = aclnn_ops.Arange()
        self.repeat = aclnn_ops.RepeatInterleaveTensor()
        self.tile = aclnn_ops.Tile()
        self.equal = aclnn_ops.Equal()
        self.generate_full_mask = P.Morph(self._generate_full_mask,
                                          self.eod_full_infer_shape,
                                          lambda *args: mstype.uint8
                                          ).add_prim_attr("self_define_shard", True)
        if _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
            self.shard()

    def eod_full_infer_shape(self, *args):
        """morph infer shape"""
        b, _ = args[0]
        return [b, 1, self.seq_length, self.seq_length]

    def _generate_full_mask(self, actual_seq_len):
        """generate full eod mask given actual_seq_len"""
        bsz = actual_seq_len.shape[0]
        offset_bsz = self.cast(self.reshape(self.arange(0, bsz, 1) * self.seq_length, (bsz, 1)), mstype.int32)
        new_actual_seq_len = self.clamp(actual_seq_len, 0, bsz * self.seq_length)
        new_actual_seq_len = new_actual_seq_len - offset_bsz
        prev_seq_len = self.roll(new_actual_seq_len)
        prev_seq_len[:, 0] = 0
        interleave_seq_len = self.sub(new_actual_seq_len, prev_seq_len)
        interleave_seq_len = self.reshape(interleave_seq_len, (-1,))
        mask_1d = self.repeat(self.arange(0, interleave_seq_len.shape[0], 1), interleave_seq_len)
        mask_1d = self.reshape(mask_1d, (bsz, -1))

        sliced_sq = self.seq_length // self.cp
        sliced_mask_1d = self.slice(mask_1d, 1, sliced_sq * self.cp_id, sliced_sq * (self.cp_id + 1), 1)
        eod_row = self.reshape(sliced_mask_1d, (bsz, -1, 1))
        eod_col = self.reshape(mask_1d, (bsz, 1, -1))
        eod_mat1 = self.tile(eod_row, (1, 1, self.seq_length))
        eod_mat2 = self.tile(eod_col, (1, sliced_sq, 1))
        eod_mat = self.equal(eod_mat1, eod_mat2)
        mask = self.cast(self.sub(self.one, self.tril(eod_mat, self.cp_id * sliced_sq)), mstype.uint8)
        mask = self.reshape(mask, (-1, 1, sliced_sq, self.seq_length))
        return mask

    def construct(self, actual_seq_len):
        """
        Forward process of the CausalEodMask

        Args:
            actual_seq_len (Tensor): Size of sequence corresponding to each sub-sequence, array with increasing values.

        Returns:
            Tensor, the eod attention mask carrying 0 and 1 values
        """
        return self.generate_full_mask(actual_seq_len)

    def shard(self):
        """sharding operators"""
        self.generate_full_mask.shard((layout("dp", "None"),), (layout("dp", "None", "cp", "None"),))
