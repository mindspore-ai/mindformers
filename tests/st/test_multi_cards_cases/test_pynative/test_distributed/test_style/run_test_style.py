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
"""Run script for testing PrepareModuleInput and PrepareModuleOutput."""
import argparse
import numpy as np

from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.core.placement_types import Shard, Replicate

import mindspore as ms
from mindspore import nn, Tensor
from mindspore.communication import init
from mindformers.pynative.distributed.style import (
    PrepareModuleInput,
    PrepareModuleOutput,
    PrepareModuleInputOutput,
)
from mindformers.pynative.layers.linear import Linear
from mindformers.parallel_core.utils.init_method import init_method_normal

TEST_RANDOM_SEED = 42

def create_test_tensors():
    """Create test tensors with fixed random seed."""
    np.random.seed(TEST_RANDOM_SEED)
    return {
        "tensor_2x16": Tensor(
            np.random.randn(2, 16).astype(np.float32), dtype=ms.float32
        ),
        "tensor_2x8": Tensor(
            np.random.randn(2, 8).astype(np.float32), dtype=ms.float32
        ),
        "tensor_4x4": Tensor(
            np.random.randn(4, 4).astype(np.float32), dtype=ms.float32
        ),
    }


class TestModule(nn.Cell):
    """Test module for PrepareModuleInput and PrepareModuleOutput."""

    def __init__(self):
        super().__init__()
        self.linear = Linear(
            input_size=8,
            output_size=8,
            params_dtype="float32",
            compute_dtype="float32",
            init_method=init_method_normal(),
            bias=True
        )

    def construct(self, x, y=None, mask=None):
        if mask is not None:
            return mask
        if y is not None:
            return x, y
        return self.linear(x)


def _verify_dtensor_result(res, expected_res, err_msg, use_local_output=False):
    """Verify DTensor result matches expected DTensor."""
    if use_local_output:
        assert isinstance(res, Tensor)
        assert not isinstance(res, DTensor)
        np.testing.assert_array_equal(
            res.asnumpy(),
            expected_res.asnumpy(),
            err_msg=err_msg,
        )
    else:
        assert isinstance(res, DTensor)
        assert res.placements == expected_res.placements
        np.testing.assert_array_equal(
            res.to_local().asnumpy(),
            expected_res.to_local().asnumpy(),
            err_msg=err_msg,
        )


def _create_reference_module(source_module):
    """Create a reference module with same weights as source."""
    ref_module = TestModule()
    source_weight = source_module.linear.weight.value()
    ref_module.linear.weight.set_data(source_weight)
    return ref_module


def _compute_expected_output(
    module, input_tensor, input_layout, desired_layout, device_mesh
):
    """Compute expected output after input conversion."""
    dtensor = DTensor.from_local(input_tensor, device_mesh, (input_layout,))
    converted_dtensor = dtensor.redistribute(device_mesh, (desired_layout,))
    local_tensor = converted_dtensor.to_local()
    ref_module = _create_reference_module(module)
    return ref_module(local_tensor)


def run_prepare_module_input(device_mesh, test_tensors):
    """Test PrepareModuleInput covering multiple scenarios."""

    # Test: Multiple inputs
    module = TestModule()
    prepare_inps = PrepareModuleInput(
        input_layouts=(Shard(0), None),
        desired_input_layouts=(Replicate(), None),
    )
    # pylint: disable=protected-access
    prepare_inps._apply(module, device_mesh)

    tensor1 = test_tensors["tensor_2x8"]
    tensor2 = test_tensors["tensor_4x4"]
    output_x, output_y = module(tensor1, tensor2)

    sharded_dtensor = DTensor.from_local(tensor1, device_mesh, (Shard(0),))
    expected_dtensor = sharded_dtensor.redistribute(
        device_mesh, (Replicate(),)
    )
    _verify_dtensor_result(
        output_x,
        expected_dtensor,
        "Multiple inputs test: Output_x mismatch",
    )
    assert output_y is tensor2
    print("PrepareModuleInput::multiple_inputs PASSED")

    # Test: DTensor input with use_local_output
    module_local = TestModule()
    prepare_local = PrepareModuleInput(
        input_layouts=(Shard(0),),
        desired_input_layouts=(Shard(1),),
        use_local_output=True,
    )
    # pylint: disable=protected-access
    prepare_local._apply(module_local, device_mesh)

    test_tensor = test_tensors["tensor_2x16"]
    test_dtensor = DTensor.from_local(test_tensor, device_mesh, (Shard(0),))
    result = module_local(test_dtensor)
    expected_result = _compute_expected_output(
        module_local, test_tensor, Shard(0), Shard(1), device_mesh
    )
    _verify_dtensor_result(
        result,
        expected_result,
        "use_local_output test: Result mismatch",
        use_local_output=True,
    )
    print("PrepareModuleInput::dtensor_input_use_local_output PASSED")

    # Test: Keyword arguments
    module_kwargs = TestModule()
    prepare_kwargs = PrepareModuleInput(
        input_kwarg_layouts={"mask": Replicate()},
        desired_input_kwarg_layouts={"mask": Shard(0)},
    )
    # pylint: disable=protected-access
    prepare_kwargs._apply(module_kwargs, device_mesh)

    x_tensor = test_tensors["tensor_2x8"]
    mask_tensor = test_tensors["tensor_2x8"]
    output = module_kwargs(x_tensor, mask=mask_tensor)

    replicated_dtensor = DTensor.from_local(
        mask_tensor, device_mesh, (Replicate(),)
    )
    expected_dtensor = replicated_dtensor.redistribute(
        device_mesh, (Shard(0),)
    )
    _verify_dtensor_result(
        output, expected_dtensor, "Kwargs test: Output mismatch"
    )
    print("PrepareModuleInput::kwargs_input PASSED")


def run_prepare_module_output(device_mesh, test_tensors):
    """Test PrepareModuleOutput covering multiple scenarios."""

    # Test: Multiple outputs
    module_multi = TestModule()
    prepare_output_multi = PrepareModuleOutput(
        output_layouts=(Shard(0), None),
        desired_output_layouts=(Replicate(), None),
        use_local_output=False,
    )
    # pylint: disable=protected-access
    prepare_output_multi._apply(module_multi, device_mesh)

    test_tensor = test_tensors["tensor_4x4"]
    output_x, output_y = module_multi(test_tensor, y=test_tensor)

    sharded_dtensor = DTensor.from_local(test_tensor, device_mesh, (Shard(0),))
    expected_dtensor = sharded_dtensor.redistribute(
        device_mesh, (Replicate(),)
    )
    _verify_dtensor_result(
        output_x, expected_dtensor, "Multiple outputs test: Output_x mismatch"
    )
    assert output_y is test_tensor
    print("PrepareModuleOutput::multiple_outputs PASSED")

    # Test: Shard dimension change with use_local_output
    module_shard = TestModule()
    prepare_shard = PrepareModuleOutput(
        output_layouts=(Shard(0),),
        desired_output_layouts=(Shard(1),),
        use_local_output=True,
    )
    # pylint: disable=protected-access
    prepare_shard._apply(module_shard, device_mesh)
    test_tensor = test_tensors["tensor_2x8"]
    result = module_shard(test_tensor)

    ref_module = _create_reference_module(module_shard)
    ref_output = ref_module(test_tensor)
    sharded_dtensor = DTensor.from_local(
        ref_output, device_mesh, (Shard(0),)
    )
    expected_dtensor = sharded_dtensor.redistribute(
        device_mesh, (Shard(1),)
    )
    expected_tensor = expected_dtensor.to_local()
    _verify_dtensor_result(
        result,
        expected_tensor,
        "Shard change test: Result mismatch",
        use_local_output=True,
    )
    print("PrepareModuleOutput::shard_dim_change_use_local_output PASSED")


def run_prepare_module_input_output(device_mesh, test_tensors):
    """Test PrepareModuleInputOutput covering multiple scenarios."""

    module_kwargs = TestModule()
    prepare_kwargs = PrepareModuleInputOutput(
        input_kwarg_layouts={"mask": Shard(0)},
        desired_input_kwarg_layouts={"mask": Replicate()},
        output_layouts=(Replicate(),),
        desired_output_layouts=(Shard(1),),
        use_local_output=False,
    )
    # pylint: disable=protected-access
    prepare_kwargs._apply(module_kwargs, device_mesh)

    x_tensor = test_tensors["tensor_2x8"]
    mask_tensor = test_tensors["tensor_2x8"]
    result = module_kwargs(x_tensor, mask=mask_tensor)

    sharded_mask_dtensor = DTensor.from_local(
        mask_tensor, device_mesh, (Shard(0),)
    )
    expected_mask_dtensor = sharded_mask_dtensor.redistribute(
        device_mesh, (Replicate(),)
    )
    expected_output_dtensor = expected_mask_dtensor.redistribute(
        device_mesh, (Shard(1),)
    )
    _verify_dtensor_result(
        result,
        expected_output_dtensor,
        "Kwargs I/O test: Result mismatch",
    )
    print("PrepareModuleInputOutput::kwargs_io PASSED")


def main():
    """Main function to run tests with configurable parameters.""" 

    parser = argparse.ArgumentParser(
        description="Run PrepareModuleInput/Output/InputOutput tests"
    )
    parser.add_argument(
        "--tp", type=int, default=2, help="Tensor parallel size"
    )

    args = parser.parse_args()

    init()
    device_mesh = init_device_mesh(mesh_shape=(args.tp,), alias_name=("tp",))

    test_tensors = create_test_tensors()

    run_prepare_module_input(device_mesh, test_tensors)
    run_prepare_module_output(device_mesh, test_tensors)
    run_prepare_module_input_output(device_mesh, test_tensors)


if __name__ == "__main__":
    main()
