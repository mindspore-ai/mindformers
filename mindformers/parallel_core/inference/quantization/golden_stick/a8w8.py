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
"""A8W8 quantization method."""

import mindspore
from mindspore import Tensor, Parameter, ops, nn
from mindspore.common.initializer import initializer
from mindspore.ops.auto_generate import QuantBatchMatmul, QuantV2
from mindformers.parallel_core.inference.weights_utils import set_weight_attrs
from mindformers.parallel_core.inference.quantization import QuantizationConfig
from mindformers.parallel_core.inference.tensor_parallel.layers import LinearMethodBase


class A8W8LinearMethod(LinearMethodBase):
    """Linear method with A8W8 quantization."""

    def __init__(self, quant_config: QuantizationConfig) -> None:
        self.quant_config = quant_config
        self.quant = QuantV2()
        self.bias_add = ops.Add()
        self.is_modelslim = self.quant_config.is_modelslim
        self.is_ms_custom_ops = False
        try:
            import ms_custom_ops
            self.is_ms_custom_ops = True
            self.ms_custom_ops = ms_custom_ops
        except ModuleNotFoundError:
            pass

    def create_weights(self,
                       layer: mindspore.nn.Cell,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int],
                       params_dtype,
                       **extra_weight_attrs):
        output_size_per_partition = sum(output_partition_sizes)
        self.output_size_per_partition = output_size_per_partition
        self.input_size_per_partition = input_size_per_partition
        self.params_dtype = params_dtype
        self.matmul = QuantBatchMatmul(transpose_x1=False,
                                       transpose_x2=True,
                                       dtype=self.params_dtype)

        weight_shape = (self.output_size_per_partition, self.input_size_per_partition)
        weight = Parameter(initializer('ones', weight_shape, mindspore.int8), requires_grad=False)
        deq_scale_shape = self.output_size_per_partition
        scale_dtype = mindspore.float32
        deq_scale = Parameter(
            initializer('ones', deq_scale_shape, scale_dtype), name="deq_scale", requires_grad=False)
        shape = (self.output_size_per_partition,)
        quant_bias = Parameter(initializer('zeros', shape, mindspore.int32), name="quant_bias", requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(deq_scale, {"output_dim": 0})
        set_weight_attrs(quant_bias, {"output_dim": 0})

        set_weight_attrs(weight, extra_weight_attrs)
        set_weight_attrs(deq_scale, extra_weight_attrs)
        set_weight_attrs(quant_bias, extra_weight_attrs)

        if not self.is_modelslim:
            input_scale_shape = (input_size_per_partition,)
            input_scale = Parameter(
                initializer('ones', input_scale_shape, self.params_dtype), name="input_scale", requires_grad=False)
            input_offset = Parameter(
                initializer('zeros', input_scale_shape, mindspore.int8), name="input_offset", requires_grad=False)
            set_weight_attrs(input_offset, {"input_dim": 0})
            set_weight_attrs(input_scale, {"input_dim": 0})
        else:
            input_scale_shape = (1,)
            input_scale = Parameter(
                initializer('ones', input_scale_shape, self.params_dtype), name="input_scale", requires_grad=False)
            input_offset = Parameter(
                initializer('zeros', input_scale_shape, self.params_dtype),
                name="input_offset", requires_grad=False)
            beta_shape = (input_size_per_partition,)
            beta = Parameter(initializer('zeros', beta_shape, self.params_dtype), name="beta", requires_grad=False)
            set_weight_attrs(beta, {"input_dim": 0})
            set_weight_attrs(beta, extra_weight_attrs)

        set_weight_attrs(input_scale, extra_weight_attrs)
        set_weight_attrs(input_offset, extra_weight_attrs)

        if layer is not None:
            layer.insert_param_to_cell("weight", weight)
            layer.insert_param_to_cell("deq_scale", deq_scale)
            layer.insert_param_to_cell("input_scale", input_scale)
            layer.insert_param_to_cell("input_offset", input_offset)
            layer.insert_param_to_cell("quant_bias", quant_bias)
            if self.is_modelslim:
                layer.insert_param_to_cell("beta", beta)
        return weight

    def process_weights_after_loading(self, layer: nn.Cell) -> None:
        """
        Process the weight after loading.
        This can be used for example, to transpose weights for computation.
        """
        if self.is_ms_custom_ops:
            layer.weight = self.ms_custom_ops.trans_data(layer.weight, transdata_type=1)
        if not self.is_modelslim:
            return
        input_scale = 1 / layer.input_scale.asnumpy()
        layer.input_scale = Parameter(
            Tensor(input_scale, dtype=mindspore.bfloat16), name=layer.input_scale.name, requires_grad=False)

    def apply(self,
              layer: mindspore.nn.Cell,
              x: mindspore.Tensor,
              weight: mindspore.Tensor,
              bias: mindspore.Parameter = None) -> Tensor:
        if weight is None:
            weight = layer.weight
        deq_scale = layer.deq_scale
        input_scale = layer.input_scale
        input_offset = layer.input_offset
        quant_bias = layer.quant_bias
        if self.is_modelslim:
            beta = layer.beta
            x = ops.add(x, beta)
        qx = self.quant(x, input_scale, input_offset, False, "ROUND", mindspore.dtype.int8)
        output_shape = qx.shape[:-1] + (self.output_size_per_partition,)
        qx = qx.reshape(-1, self.input_size_per_partition)
        if self.is_ms_custom_ops:
            out = self.ms_custom_ops.quant_batch_matmul(qx, weight, deq_scale, None, quant_bias, None,
                                                        transpose_x1=False, transpose_x2=True,
                                                        x2_format="FRACTAL_NZ", output_dtype=self.params_dtype)
        else:
            out = self.matmul(qx, weight, deq_scale, None, quant_bias, None)
        if bias is not None:
            out = self.bias_add(out, bias)
        out = out.reshape(output_shape)
        return out
