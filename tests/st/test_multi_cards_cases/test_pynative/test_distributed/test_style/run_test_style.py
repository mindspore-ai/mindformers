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
"""Two-card forward/backward validation for local tensor-parallel styles."""

import argparse
import numpy as np

import mindspore as ms
from mindspore import nn, Tensor, mint
from mindspore.communication import get_rank, init

from hyper_parallel import DTensor, init_device_mesh
from mindformers.pynative.distributed.style import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
)
from mindformers.pynative.layers.linear import Linear


class _TwoLinear(nn.Cell):
    """Column-parallel linear followed by row-parallel linear."""

    def __init__(self):
        super().__init__()
        self.linear_fc1 = Linear(4, 8, "float32", "float32", bias=False)
        self.linear_fc2 = Linear(8, 4, "float32", "float32", bias=False)

    def construct(self, value):
        return self.linear_fc2(self.linear_fc1(value))


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _set_weights(network, weight1, weight2):
    network.linear_fc1.weight.set_data(Tensor(weight1, ms.float32))
    network.linear_fc2.weight.set_data(Tensor(weight2, ms.float32))


def _forward_and_grads(network, value):
    params = network.trainable_params()

    def loss_fn(input_):
        return mint.sum(network(input_))

    grad_fn = ms.value_and_grad(loss_fn, grad_position=0, weights=params)
    loss, (input_grad, param_grads) = grad_fn(value)
    return network(value), loss, input_grad, param_grads


def run_colwise_rowwise(mesh):
    """Compare TP forward, input gradient, and both weight gradients to dense reference."""
    rank = get_rank()
    rng = np.random.default_rng(17)
    global_input = rng.normal(size=(4, 1, 4)).astype(np.float32)
    weight1 = rng.normal(size=(8, 4)).astype(np.float32)
    weight2 = rng.normal(size=(4, 8)).astype(np.float32)

    reference = _TwoLinear()
    _set_weights(reference, weight1, weight2)
    ref_output, _, ref_input_grad, ref_param_grads = _forward_and_grads(
        reference, Tensor(global_input)
    )

    parallel = _TwoLinear()
    _set_weights(parallel, weight1, weight2)
    ColwiseParallel(gather_input=True)._apply(parallel.linear_fc1, mesh)
    RowwiseParallel(reduce_mode="reduce_scatter")._apply(parallel.linear_fc2, mesh)
    local_input = Tensor(np.split(global_input, 2, axis=0)[rank])
    output, _, input_grad, param_grads = _forward_and_grads(parallel, local_input)

    np.testing.assert_allclose(
        output.asnumpy(), np.split(ref_output.asnumpy(), 2, axis=0)[rank], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        input_grad.asnumpy(), np.split(ref_input_grad.asnumpy(), 2, axis=0)[rank], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        _local(param_grads[0]).asnumpy(),
        np.split(ref_param_grads[0].asnumpy(), 2, axis=0)[rank],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        _local(param_grads[1]).asnumpy(),
        np.split(ref_param_grads[1].asnumpy(), 2, axis=1)[rank],
        rtol=1e-5,
        atol=1e-5,
    )
    print("ColwiseParallel+RowwiseParallel forward/backward PASSED")


def run_rowwise_all_reduce(mesh):
    """Validate the rowwise AllReduce output mode and its local input gradient."""
    rank = get_rank()
    rng = np.random.default_rng(23)
    global_input = rng.normal(size=(3, 1, 4)).astype(np.float32)
    weight = rng.normal(size=(5, 4)).astype(np.float32)

    reference = Linear(4, 5, "float32", "float32", bias=False)
    reference.weight.set_data(Tensor(weight))
    ref_output, _, ref_input_grad, _ = _forward_and_grads(reference, Tensor(global_input))

    parallel = Linear(4, 5, "float32", "float32", bias=False)
    parallel.weight.set_data(Tensor(weight))
    RowwiseParallel(reduce_mode="all_reduce")._apply(parallel, mesh)
    local_input = Tensor(np.split(global_input, 2, axis=-1)[rank])
    output, _, input_grad, _ = _forward_and_grads(parallel, local_input)

    np.testing.assert_allclose(output.asnumpy(), ref_output.asnumpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        input_grad.asnumpy(), np.split(ref_input_grad.asnumpy(), 2, axis=-1)[rank], rtol=1e-5, atol=1e-5
    )
    print("RowwiseParallel all_reduce forward/backward PASSED")


def run_colwise_gather_output(mesh):
    """Validate gather-output slice backward and sequence-input reduce-scatter backward."""
    rank = get_rank()
    rng = np.random.default_rng(29)
    global_input = rng.normal(size=(4, 1, 4)).astype(np.float32)
    weight = rng.normal(size=(8, 4)).astype(np.float32)

    reference = Linear(4, 8, "float32", "float32", bias=False)
    reference.weight.set_data(Tensor(weight))
    ref_output, _, ref_input_grad, ref_param_grads = _forward_and_grads(
        reference, Tensor(global_input)
    )

    parallel = Linear(4, 8, "float32", "float32", bias=False)
    parallel.weight.set_data(Tensor(weight))
    ColwiseParallel(gather_input=True, gather_output=True)._apply(parallel, mesh)
    local_input = Tensor(np.split(global_input, 2, axis=0)[rank])
    output, _, input_grad, param_grads = _forward_and_grads(parallel, local_input)

    np.testing.assert_allclose(output.asnumpy(), ref_output.asnumpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        input_grad.asnumpy(), np.split(ref_input_grad.asnumpy(), 2, axis=0)[rank], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        _local(param_grads[0]).asnumpy(),
        np.split(ref_param_grads[0].asnumpy(), 2, axis=0)[rank],
        rtol=1e-5,
        atol=1e-5,
    )
    print("ColwiseParallel gather_output forward/backward PASSED")


def run_automatic_input_sharding(mesh):
    """Validate that automatic TP/SP slicing gathers local gradients in backward."""
    rng = np.random.default_rng(31)
    global_input = rng.normal(size=(4, 1, 4)).astype(np.float32)
    weight = rng.normal(size=(5, 4)).astype(np.float32)

    reference = Linear(4, 5, "float32", "float32", bias=False)
    reference.weight.set_data(Tensor(weight))
    ref_output, _, ref_input_grad, _ = _forward_and_grads(reference, Tensor(global_input))

    parallel = Linear(4, 5, "float32", "float32", bias=False)
    parallel.weight.set_data(Tensor(weight))
    RowwiseParallel(reduce_mode="all_reduce", input_is_parallel=False)._apply(parallel, mesh)
    output, _, input_grad, _ = _forward_and_grads(parallel, Tensor(global_input))
    np.testing.assert_allclose(output.asnumpy(), ref_output.asnumpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(input_grad.asnumpy(), ref_input_grad.asnumpy(), rtol=1e-5, atol=1e-5)

    identity = nn.Identity()
    SequenceParallel(sequence_dim=0, input_is_parallel=False)._apply(identity, mesh)
    grad_fn = ms.grad(lambda value: mint.sum(identity(value)))
    sequence_grad = grad_fn(Tensor(global_input))
    np.testing.assert_array_equal(sequence_grad.asnumpy(), np.ones_like(global_input))
    print("RowwiseParallel/SequenceParallel input sharding backward PASSED")


def main():
    parser = argparse.ArgumentParser(description="Run local style distributed gradient tests")
    parser.add_argument("--tp", type=int, default=2)
    args = parser.parse_args()

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    init()
    mesh = init_device_mesh("npu", (args.tp,), mesh_dim_names=("tp",))
    run_colwise_rowwise(mesh)
    run_rowwise_all_reduce(mesh)
    run_colwise_gather_output(mesh)
    run_automatic_input_sharding(mesh)


if __name__ == "__main__":
    main()
