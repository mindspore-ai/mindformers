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
"""BatchInvariantLinear units for tensor parallelism"""

__all__ = [
    "BatchInvariantColumnParallelLinear",
    "BatchInvariantQKVParallelLinear",
    "BatchInvariantRowParallelLinear",
    "BatchInvariantVocabParallelEmbedding",
    "BatchInvariantReplicatedLinear"
]

from typing import Callable, List, Optional

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore.common.initializer import initializer

import ms_custom_ops

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.parallel_state import ProcessGroup, default_pgs
from mindformers.parallel_core.inference.tensor_parallel.layers import (LinearMethodBase,
                                                                        ColumnParallelLinear,
                                                                        RowParallelLinear,
                                                                        MergedColumnParallelLinear,
                                                                        QKVParallelLinear,
                                                                        ReplicatedLinear,
                                                                        VocabParallelEmbedding)
from mindformers.parallel_core.inference.quantization.base_config import (QuantizeMethodBase,
                                                                          QuantizationConfig)
from mindformers.parallel_core.inference.weights_utils import set_weight_attrs
from mindformers.parallel_core.inference.utils import divide
from mindformers.tools.utils import is_pynative

class BatchInvariantLinearMethod(LinearMethodBase):
    """BatchInvariantLinear method without quantization."""

    def create_weights(self, layer: ms.nn.Cell, input_size_per_partition: int,
                       output_partition_sizes: List[int], params_dtype, **extra_weight_attrs):
        if extra_weight_attrs.get('transpose_b'):
            weight_shape = (int(sum(output_partition_sizes)), int(input_size_per_partition))
        else:
            weight_shape = (int(input_size_per_partition), int(sum(output_partition_sizes)))
        weight = Parameter(initializer("zeros", weight_shape, params_dtype), requires_grad=False)
        self.input_size_per_partition = int(input_size_per_partition)
        self.output_size_per_partition = int(sum(output_partition_sizes))
        if extra_weight_attrs.get('transpose_b'):
            set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        else:
            set_weight_attrs(weight, {"input_dim": 0, "output_dim": 1})
        layer.insert_param_to_cell("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        self.config = layer.config
        self.transpose_b = extra_weight_attrs.get('transpose_b')
        self.tarnspose = ops.Transpose()
        self.matmul = ms_custom_ops.matmul_batch_invariant
        self.cast = ops.Cast()

    def apply(self, layer: ms.nn.Cell, x: Tensor, weight: Tensor, bias: Parameter = None):
        origin_dtype = x.dtype
        output_shape = x.shape[:-1] + (self.output_size_per_partition,)

        x = mint.reshape(x, (-1, self.input_size_per_partition))
        x = self.cast(x, layer.compute_dtype)
        weight = self.cast(weight, layer.compute_dtype)
        if self.transpose_b:
            weight = self.tarnspose(weight, (1, 0))
        output_parallel = self.matmul(x, weight)

        if bias is not None and not layer.skip_bias_add:
            bias = self.cast(bias, layer.compute_dtype)
            output_parallel = mint.add(output_parallel, bias)

        output_parallel = mint.reshape(output_parallel, output_shape)
        output_parallel = self.cast(output_parallel, origin_dtype)
        return output_parallel


class BatchInvariantEmbeddingMethod(QuantizeMethodBase):
    """BatchInvariant Unquantized method for embeddings."""

    def create_weights(self, layer: nn.Cell, input_size_per_partition: int,
                       output_partition_sizes: List[int], params_dtype, **extra_weight_attrs):
        """Create weights for embedding layer."""
        weight = Parameter(
            mint.zeros((sum(output_partition_sizes), input_size_per_partition), dtype=params_dtype),
            requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.insert_param_to_cell("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

        self.input_size_per_partition = int(input_size_per_partition)
        self.output_size_per_partition = int(sum(output_partition_sizes))
        self.config = layer.config
        self.transpose_b = True
        self.transpose = ops.Transpose()
        self.matmul = ms_custom_ops.matmul_batch_invariant
        self.gather = ops.Gather()
        self.bias_add = ops.Add()

    def apply(self, layer: nn.Cell, x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
        origin_dtype = x.dtype
        output_shape = x.shape[:-1] + (self.output_size_per_partition,)

        x = mint.reshape(x, (-1, layer.input_size))
        x = self.cast(x, layer.compute_dtype)
        weight = self.cast(weight, layer.compute_dtype)
        if self.transpose_b:
            weight = self.transpose(weight, (1, 0))
        output_parallel = self.matmul(x, weight)

        if bias is not None and not layer.skip_bias_add:
            bias = self.cast(bias, layer.compute_dtype)
            output_parallel = mint.add(output_parallel, bias)

        output_parallel = mint.reshape(output_parallel, output_shape)
        output_parallel = self.cast(output_parallel, origin_dtype)
        return output_parallel

    def embedding(self, layer: nn.Cell, input_: Tensor) -> Tensor:
        return self.gather(layer.weight, input_, 0)


class BatchInvariantColumnParallelLinear(ColumnParallelLinear):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            *,
            config: TransformerConfig,
            init_method: Callable = "normal",
            bias: bool = True,
            gather_output: bool = False,
            stride: int = 1,
            keep_master_weight_for_test: bool = False,
            skip_bias_add: bool = False,
            skip_weight_param_allocation: bool = False,
            embedding_activation_buffer: Optional[List[Tensor]] = None,
            is_expert: bool = False,
            tp_comm_buffer_name: str = None,
            transpose_b: bool = True,
            compute_dtype: mstype = mstype.bfloat16,
            tp_group: ProcessGroup = default_pgs,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = ""
    ):
        super(ColumnParallelLinear, self).__init__(
            input_size,
            output_size,
            skip_bias_add,
            config.params_dtype,
            quant_config=quant_config,
            prefix=prefix
        )
        if stride > 1:
            raise NotImplementedError(f"For BatchInvariantColumnParallelLinear, `stride > 1` is not supported for now, "
                                      f"but got `stride={stride}`")
        if keep_master_weight_for_test:
            raise NotImplementedError(
                "For BatchInvariantColumnParallelLinear, `keep_master_weight_for_test` is not supported for now")
        if skip_bias_add:
            raise NotImplementedError("For BatchInvariantColumnParallelLinear, `skip_bias_add=True` is not supported for now")
        if embedding_activation_buffer is not None:
            raise NotImplementedError(
                "For BatchInvariantColumnParallelLinear, `embedding_activation_buffer` is not supported for now")
        if tp_comm_buffer_name is not None:
            raise NotImplementedError("For BatchInvariantColumnParallelLinear, `tp_comm_buffer_name` is not supported for now")

        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.init_method = init_method
        self.has_bias = bias
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.skip_weight_param_allocation = skip_weight_param_allocation
        self.transpose_b = transpose_b
        self.compute_dtype = compute_dtype
        self.params_dtype = config.params_dtype
        self.cast = P.Cast()
        self.matmul = P.MatMul(transpose_b=self.transpose_b)

        self.tp_group = tp_group
        self.tensor_parallel_group_size = self.tp_group.size
        self.output_size_per_partition = divide(output_size, self.tensor_parallel_group_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tensor_parallel_group_size)
                for output_size in self.output_sizes
            ]

        bias_shape = (sum(self.output_partition_sizes),)
        if self.has_bias:
            self.bias = Parameter(
                mint.empty(bias_shape, dtype=self.params_dtype),
                name="bias"
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.bias = None

        self.quant_method = BatchInvariantLinearMethod()

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
            transpose_b=self.transpose_b
        )


class BatchInvariantMergedColumnParallelLinear(MergedColumnParallelLinear):
    def __init__(self,
                 hidden_size: int,
                 ffn_hidden_size: int,
                 *,
                 config: TransformerConfig,
                 bias: bool = True,
                 gather_output: bool = False,
                 is_expert: bool = False,
                 transpose_b: bool = True,
                 compute_dtype: mstype = None,
                 tp_group: ProcessGroup = default_pgs,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""
    ):
        self.params_dtype = config.params_dtype

        # Divide the weight matrix along the last dimension.
        self.tp = tp_group
        output_size = (
            ffn_hidden_size
        )
        self.output_sizes = [
            ffn_hidden_size,
            ffn_hidden_size,
        ]

        BatchInvariantColumnParallelLinear.__init__(
            self,
            input_size=hidden_size,
            output_size=output_size,
            config=config,
            bias=bias,
            gather_output=gather_output,
            is_expert=is_expert,
            transpose_b=transpose_b,
            compute_dtype=compute_dtype,
            tp_group=tp_group,
            quant_config=quant_config,
            prefix=prefix
        )


class BatchInvariantQKVParallelLinear(QKVParallelLinear):
    def __init__(
            self,
            hidden_size: int,
            head_size: int,
            total_num_heads: int,
            total_num_kv_heads: int,
            *,
            config: TransformerConfig,
            bias: bool = True,
            gather_output: bool = False,
            transpose_b: bool = True,
            compute_dtype: mstype = None,
            tp_group: ProcessGroup = default_pgs,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = ""
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.params_dtype = config.params_dtype

        # Divide the weight matrix along the last dimension.
        self.tp = tp_group
        tp_size = self.tp.size
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        output_size = (
            (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        )
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]

        BatchInvariantColumnParallelLinear.__init__(
            self,
            input_size=hidden_size,
            output_size=output_size,
            config=config,
            bias=bias,
            gather_output=gather_output,
            transpose_b=transpose_b,
            compute_dtype=compute_dtype,
            tp_group=tp_group,
            quant_config=quant_config,
            prefix=prefix
        )


class BatchInvariantRowParallelLinear(RowParallelLinear):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            *,
            config: TransformerConfig,
            init_method: Callable = "normal",
            bias: bool = True,
            input_is_parallel: bool = True,
            skip_bias_add: bool = False,
            stride: int = 1,
            keep_master_weight_for_test: bool = False,
            delay_allreduce: bool = False,
            is_expert: bool = False,
            tp_comm_buffer_name: str = None,
            transpose_b: bool = True,
            compute_dtype: mstype = mstype.bfloat16,
            tp_group: ProcessGroup = default_pgs,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = ""
    ):
        super(RowParallelLinear, self).__init__(
            input_size,
            output_size,
            skip_bias_add,
            config.params_dtype,
            quant_config=quant_config,
            prefix=prefix
            )
        if stride > 1:
            raise NotImplementedError(f"For BatchInvariantRowParallelLinear, `stride > 1` is not supported for now, "
                                      f"but got `stride={stride}`")
        if skip_bias_add:
            raise NotImplementedError("For BatchInvariantRowParallelLinear, `skip_bias_add=True` is not supported for now")
        if keep_master_weight_for_test:
            raise NotImplementedError("For BatchInvariantRowParallelLinear, `keep_master_weight_for_test=True` "
                                      "is not supported for now.")
        if tp_comm_buffer_name:
            raise NotImplementedError("For BatchInvariantRowParallelLinear, `tp_comm_buffer_name` is not supported for now.")
        if delay_allreduce and bias:
            raise RuntimeError(
                "In BatchInvariantRowParallelLinear, `delay_allreduce` and `has_bias` cannot be enabled simultaneously, "
                "otherwise the accuracy will be incorrect."
            )

        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.input_is_parallel = input_is_parallel
        self.has_bias = bias
        self.skip_bias_add = skip_bias_add

        self.is_pynative = is_pynative()
        self.tp_group = tp_group
        self.tensor_parallel_group_size = self.tp_group.size
        self.input_size_per_partition = divide(input_size, self.tensor_parallel_group_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]
        self.compute_dtype = compute_dtype
        self.delay_allreduce = delay_allreduce
        self.transpose_b = transpose_b
        self.params_dtype = config.params_dtype
        self.cast = P.Cast()
        self.matmul = P.MatMul(transpose_b=self.transpose_b)

        bias_shape = (self.output_size,)
        if self.has_bias:
            self.bias = Parameter(
                mint.empty(bias_shape, dtype=self.params_dtype),
                name="bias"
                )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.bias = None

        self.quant_method = BatchInvariantLinearMethod()
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
            transpose_b=self.transpose_b
        )


class BatchInvariantReplicatedLinear(ReplicatedLinear):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            *,
            config: TransformerConfig,
            init_method: Callable = "normal",
            bias: bool = True,
            stride: int = 1,
            keep_master_weight_for_test: bool = False,
            skip_bias_add: bool = False,
            skip_weight_param_allocation: bool = False,
            embedding_activation_buffer: Optional[List[Tensor]] = None,
            is_expert: bool = False,
            tp_comm_buffer_name: str = None,
            transpose_b: bool = True,
            compute_dtype: mstype = None,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = ""
    ):
        super(ReplicatedLinear, self).__init__(
            input_size,
            output_size,
            skip_bias_add,
            config.params_dtype,
            quant_config=quant_config,
            prefix=prefix
            )
        if stride > 1:
            raise NotImplementedError(f"For BatchInvariantReplicatedLinear, `stride > 1` is not supported for now, "
                                      f"but got `stride={stride}`")
        if skip_bias_add:
            raise NotImplementedError("For BatchInvariantReplicatedLinear, `skip_bias_add=True` is not supported for now")
        if keep_master_weight_for_test:
            raise NotImplementedError(
                "For BatchInvariantReplicatedLinear, `keep_master_weight_for_test` is not supported for now")
        if embedding_activation_buffer is not None:
            raise NotImplementedError(
                "For BatchInvariantReplicatedLinear, `embedding_activation_buffer` is not supported for now")
        if tp_comm_buffer_name is not None:
            raise NotImplementedError("For BatchInvariantReplicatedLinear, `tp_comm_buffer_name` is not supported for now")

        self.input_size = input_size
        self.output_size = output_size
        self.output_size = [self.output_size]
        self.config = config
        self.init_method = init_method
        self.has_bias = bias
        self.skip_bias_add = skip_bias_add
        self.skip_weight_param_allocation = skip_weight_param_allocation
        self.transpose_b = transpose_b
        self.compute_dtype = compute_dtype
        self.params_dtype = config.params_dtype
        self.cast = P.Cast()
        self.matmul = P.MatMul(transpose_b=self.transpose_b)

        self.tensor_parallel_group_size = 1

        bias_shape = (self.output_size,)
        if self.has_bias:
            self.bias = Parameter(
                mint.empty(bias_shape, dtype=self.params_dtype),
                name="bias"
            )
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.bias = None

        self.quant_method = BatchInvariantLinearMethod()
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
            transpose_b=self.transpose_b
        )


class BatchInvariantVocabParallelEmbedding(VocabParallelEmbedding):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            *,
            init_method: Callable,
            config: TransformerConfig,
            reduce_scatter_embeddings: bool = False,
            tp_group: ProcessGroup = default_pgs,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = ""
    ):
        super(VocabParallelEmbedding, self).__init__()

        if reduce_scatter_embeddings:
            raise NotImplementedError("For BatchInvariantVocabParallelEmbedding, "
                                      "reduce_scatter_embeddings is not supported for now")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.quant_method = BatchInvariantEmbeddingMethod()
        self.tp_group = tp_group
        self.tensor_parallel_group_size = self.tp_group.size
        self.config = config
        rank_id = self.tp_group.rank

        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = self._vocab_range_from_global_vocab_size(
            self.num_embeddings, rank_id, self.tensor_parallel_group_size
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )
        self.max_index_per_partition = Tensor(self.num_embeddings_per_partition - 1, dtype=mstype.int32)
        self.expand_dims = ops.ExpandDims()
        self.gather = ops.Gather()
        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            params_dtype=config.params_dtype,
            weight_loader=self.weight_loader,
        )
