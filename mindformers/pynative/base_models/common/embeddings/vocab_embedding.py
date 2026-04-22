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
""" For language model """
__all__ = [
    "VocabEmbedding",
]

from typing import Callable

import mindspore._checkparam as Validator
from mindspore import nn, mint
from mindspore.common import dtype
from mindspore.common.parameter import Parameter


class VocabEmbedding(nn.Cell):
    """Vocab Embedding.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.

    Args:
        num_embeddings (int): vocabulary size.
        embedding_dim (int): size of hidden state.
        init_method (Callable): The initialization method.
    """
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            init_method: Callable,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # use gather instead of embedding to avoid the error of hyper-parllel
        self.embedding = mint.gather
        self.tile = mint.tile
        self.reshape = mint.reshape

        self.init_method = init_method
        self.weight = Parameter(mint.empty([self.num_embeddings, self.embedding_dim]), name="weight")

    def reset_parameter(self):
        """Reset embedding weights for delayed initialization."""
        self.weight.normal_(mean=0.0, std=0.01)

    def construct(self, input_):
        """
        Forward of vocab embedding.

        input_: (B, S)
        weight: (V, H)
        output: (B, S, H)
        """
        Validator.check_type_name("input_ids", input_.dtype, [dtype.int32, dtype.int64], self.cls_name)

        _, seq_len = input_.shape

        input_ = self.reshape(input_, (-1, 1))
        input_ = self.tile(input_, (1, self.embedding_dim))
        masked_input = input_

        output = self.embedding(self.weight, 0, masked_input)
        output = self.reshape(output, (-1, seq_len, self.embedding_dim))

        return output
