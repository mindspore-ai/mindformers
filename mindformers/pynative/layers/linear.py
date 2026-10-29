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
"""Linear units for tensor parallelism."""
__all__ = [
    "Linear",
]

from typing import Callable

from mindspore import nn, Tensor, mint, ops
from mindspore.common.parameter import Parameter

from hyper_parallel import DTensor

from mindformers.models.utils import convert_mstype
from mindformers.parallel_core.utils.init_method import init_method_zero


class Linear(nn.Cell):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Tensor parallelism runs on plain local tensors: the weight is stored as a
    sharded DTensor (TP shard + FSDP) and localised (``to_local``) here at
    compute time, so the matmul is a pure local op. The tensor-parallel
    collective is written by hand in forward hooks attached around this module
    (colwise: all-gather the sequence-sharded input; rowwise: reduce-scatter the
    output) -- never inside the module. A rowwise module sets ``skip_add_bias``
    so its Replicate bias is added by the post-hook *after* the reduce (adding it
    here would count it once per TP rank).

    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
        compute_dtype (str): The data type of the computation (e.g., 'bf16', 'float16').
        params_dtype (str): Optional storage dtype. Defaults to ``compute_dtype``.
        init_method (Callable): The initialization method. Default: None.
        bias (bool): Whether to include bias in the linear layer. Default: True.
        skip_weight_param_allocation (bool): Whether to skip weight parameter allocation. Default: False.
        bias_init (Callable): The initialization method for bias. Default: None.
        output_groups (int): Store equal output sections as an explicit leading
            axis, so column parallelism shards within every section. Default: 1.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 compute_dtype: str,
                 params_dtype: str = None,
                 init_method: Callable = None,
                 bias: bool = True,
                 skip_weight_param_allocation: bool = False,
                 bias_init: Callable = None,
                 output_groups: int = 1,
                 ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.init_method = init_method
        self.skip_weight_param_allocation = skip_weight_param_allocation
        self.has_bias = bias
        if output_groups < 1 or output_size % output_groups:
            raise ValueError("output_groups must be positive and divide output_size.")
        self.output_groups = output_groups
        self.compute_dtype = convert_mstype(compute_dtype)
        self.params_dtype = (
            convert_mstype(params_dtype) if params_dtype is not None else self.compute_dtype
        )

        # use_cpu_initialization configuration is not supported for now.
        if skip_weight_param_allocation:
            self.weight = None
        else:
            # Weight is stored as (output_size, input_size) and transposed at runtime
            weight_shape = (output_size, input_size)
            if output_groups > 1:
                weight_shape = (output_groups, output_size // output_groups, input_size)
            self.weight = Parameter(mint.empty(weight_shape, dtype=self.params_dtype), name='weight')

        if self.has_bias:
            if bias_init is None:
                bias_init = init_method_zero(self.params_dtype)
            self.bias_init = bias_init
            bias_shape = (output_size,) if output_groups == 1 else (output_groups, output_size // output_groups)
            self.bias = Parameter(mint.empty(bias_shape, dtype=self.params_dtype), name='bias')
        else:
            self.bias_init = None
            self.bias = None

        self.matmul = mint.matmul
        self.transpose = mint.transpose
        self.cast = ops.cast
        self.add = mint.add

    #: Rowwise sets this so its Replicate bias is added by the reduce post-hook,
    #: after the collective, instead of inside the (per-rank) matmul.
    skip_add_bias: bool = False

    def construct(self, input_: Tensor, weight: Tensor = None) -> Tensor:
        """Forward of Linear (local matmul on the local weight shard).

        Args:
            input_ (Tensor): The input tensor.
            weight (Tensor): The weight tensor. Default: None.

        Returns:
            output (Tensor): The output tensor.
        """
        if weight is None:
            if self.skip_weight_param_allocation:
                raise ValueError("For Linear, when `skip_weight_param_allocation` is enabled,"
                                 " `weight` is required, but got None")
            weight = self.weight

        ori_dtype = input_.dtype

        # Weight is a sharded DTensor (TP + FSDP); compute on its local shard.
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        if self.output_groups > 1:
            weight = weight.reshape(-1, weight.shape[-1])
        if weight.dtype != self.compute_dtype:
            weight = self.cast(weight, self.compute_dtype)
        if input_.dtype != self.compute_dtype:
            input_ = self.cast(input_, self.compute_dtype)

        # Transpose weight from (output_size, input_size) to (input_size, output_size)
        weight = self.transpose(weight, 1, 0)

        # Directly use 3D input: (batch, seq, input_size) @ (input_size, output_size) -> (batch, seq, output_size)
        output = self.matmul(input_, weight)

        if self.has_bias and not self.skip_add_bias:
            bias = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
            if self.output_groups > 1:
                bias = bias.reshape(-1)
            if bias.dtype != self.compute_dtype:
                bias = self.cast(bias, self.compute_dtype)
            output = self.add(output, bias)

        if output.dtype != ori_dtype:
            output = self.cast(output, ori_dtype)

        return output

    def reset_parameter(self):
        """Reset linear parameters for delayed initialization."""
        if not self.skip_weight_param_allocation and self.weight is not None:
            self.weight.normal_(mean=0.0, std=0.01)
        if self.has_bias and self.bias is not None and self.bias_init is not None:
            self.bias.zero_()
