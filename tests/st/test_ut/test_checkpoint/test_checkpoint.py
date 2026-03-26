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
"""Test checkpoint module."""
# pylint: disable=W0621
import os
import json
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import mindspore as ms

from mindspore import Tensor, Parameter, nn
from mindspore.common import dtype as mstype

from mindformers.checkpoint.checkpoint import (
    AsyncSaveManager,
    save_checkpoint,
    load_checkpoint,
    check_the_param_for_load_ckpt,
    load_parameters,
    get_checkpoint_path,
    CommonInfo,
    load_hf_checkpoint,
)
from mindformers.checkpoint.converter.template import WeightTemplate
from mindformers.checkpoint.sharded_tensor import build_sharded_tensor
from mindformers.models.qwen3.configuration_qwen3 import Qwen3Config
from mindformers.models.qwen3.utils import Qwen3PreTrainedModel
from mindformers.parallel_core.transformer_config_utils import convert_to_transformer_config
from mindformers.checkpoint.utils import (
    get_common_filename,
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    get_checkpoint_iter_dir,
    FileType
)


class SimpleNet(nn.Cell):
    """Simple network for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(10, 20)
        self.fc2 = nn.Dense(20, 1)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@pytest.fixture
def simple_network():
    """Create a simple network for testing."""
    return SimpleNet()


@pytest.fixture
def optimizer(simple_network):
    """Create an optimizer for testing."""
    return nn.Adam(simple_network.trainable_params(), learning_rate=0.001)


def _qwen3_hf_mock_weight_shapes():
    """HF tensor name -> shape for Qwen3 dense mock (single layer 0)."""
    h, mi, v = 32, 64, 128
    hs2 = h // 2
    return {
        "model.embed_tokens.weight": (v, h),
        "model.norm.weight": (h,),
        "lm_head.weight": (v, h),
        "model.layers.0.self_attn.q_proj.weight": (h, h),
        "model.layers.0.self_attn.k_proj.weight": (hs2, h),
        "model.layers.0.self_attn.v_proj.weight": (hs2, h),
        "model.layers.0.self_attn.o_proj.weight": (h, h),
        "model.layers.0.input_layernorm.weight": (h,),
        "model.layers.0.self_attn.q_norm.weight": (8,),
        "model.layers.0.self_attn.k_norm.weight": (8,),
        "model.layers.0.mlp.gate_proj.weight": (mi, h),
        "model.layers.0.mlp.up_proj.weight": (mi, h),
        "model.layers.0.mlp.down_proj.weight": (h, mi),
        "model.layers.0.post_attention_layernorm.weight": (h,),
    }


def _write_qwen3_hf_mock_checkpoint(checkpoint_dir):
    """
    Create a HuggingFace-style directory layout with Qwen3 dense tensor names
    (index.json + single safetensors shard). Shapes are minimal and self-consistent.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    st_name = "model-00001-of-00001.safetensors"
    st_path = os.path.join(checkpoint_dir, st_name)

    shape_map = _qwen3_hf_mock_weight_shapes()
    weights = {k: np.random.randn(*s).astype(np.float32) for k, s in shape_map.items()}

    to_save = [{"name": name, "data": Tensor(arr, dtype=ms.float32)} for name, arr in weights.items()]
    ms.save_checkpoint(to_save, st_path, format="safetensors")

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    weight_map = {k: st_name for k in weights}
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)


def _qwen3_reshard_numpy_dict():
    """Simulated ReshardLoader output: HF tensor names for Qwen3 dense."""
    shape_map = _qwen3_hf_mock_weight_shapes()
    return {k: np.random.randn(*s).astype(np.float32) for k, s in shape_map.items()}


def _qwen3_minimal_dst_sharded_metas_layer0():
    """MF-side ShardedTensor metas (layout=None; tests only need keys + shapes for wiring)."""
    h = 32
    mf_shapes = (
        ("embedding.word_embeddings.weight", (128, h)),
        ("decoder.final_layernorm.weight", (h,)),
        ("output_layer.weight", (128, h)),
        ("decoder.layers.0.self_attention.linear_qkv.weight", (h * 2, h)),
        ("decoder.layers.0.self_attention.linear_proj.weight", (h, h)),
        ("decoder.layers.0.input_layernorm.weight", (h,)),
        ("decoder.layers.0.self_attention.q_layernorm.weight", (8,)),
        ("decoder.layers.0.self_attention.k_layernorm.weight", (8,)),
        ("decoder.layers.0.mlp.linear_fc1.weight", (128, h)),
        ("decoder.layers.0.mlp.linear_fc2.weight", (h, 64)),
        ("decoder.layers.0.pre_mlp_layernorm.weight", (h,)),
    )
    return {
        name: build_sharded_tensor(
            param_name=name,
            param_dtype=ms.float32,
            local_shape=shape,
            global_shape=shape,
            global_offset=(0,) * len(shape),
            axis_fragmentations=(1,) * len(shape),
            layout=None,
        )
        for name, shape in mf_shapes
    }


@pytest.fixture
def qwen3_hf_mock_checkpoint_dir(tmp_path):
    """Directory containing Qwen3 dense HF mock checkpoint (index + safetensors)."""
    _write_qwen3_hf_mock_checkpoint(str(tmp_path))
    return tmp_path


class TestSaveCheckpoint:
    """Test save checkpoint scenarios - Basic and advanced tests."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_checkpoint_basic(self, tmp_path, simple_network, optimizer):
        """
        Feature: Test basic save_checkpoint functionality.
        Description: Test saving checkpoint with network and optimizer.
        Expectation: Checkpoint directory and files should be created.
        """
        iteration = 100
        common_info = CommonInfo(
            epoch_num=1,
            step_num=iteration,
            global_step=iteration,
            loss_scale=1.0,
            global_batch_size=128
        )

        save_checkpoint(
            iteration=iteration,
            network=simple_network,
            optimizer=optimizer,
            common_info=common_info,
            save_checkpoint_path=tmp_path
        )
        checkpoint_dir = get_checkpoint_iter_dir(tmp_path, iteration)
        assert os.path.exists(checkpoint_dir)

        # Check common.json exists
        common_file = get_common_filename(tmp_path, iteration)
        assert os.path.exists(common_file)

        # Check model file exists
        model_file = get_checkpoint_name(checkpoint_dir, None, 0, 1, FileType.MODEL) + '.safetensors'
        assert os.path.exists(model_file)

        # Check optimizer file exists
        optimizer_file = get_checkpoint_name(checkpoint_dir, None, 0, 1, FileType.OPTIMIZER) + '.safetensors'
        assert os.path.exists(optimizer_file)

        # Check latest_iteration.txt exists
        tracker_file = get_checkpoint_tracker_filename(tmp_path)
        assert os.path.exists(tracker_file)
        with open(tracker_file, 'r', encoding='utf-8') as f:
            assert f.read().strip() == str(iteration)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_checkpoint_without_optimizer(self, tmp_path, simple_network):
        """
        Feature: Test save_checkpoint without optimizer.
        Description: Test saving checkpoint with only network.
        Expectation: Checkpoint should be saved without optimizer files.
        """
        iteration = 200
        common_info = CommonInfo(epoch_num=1, global_step=iteration)

        save_checkpoint(
            iteration=iteration,
            network=simple_network,
            optimizer=None,
            common_info=common_info,
            save_checkpoint_path=tmp_path
        )
        checkpoint_dir = get_checkpoint_iter_dir(tmp_path, iteration)
        assert os.path.exists(checkpoint_dir)

        # Check model file exists
        model_file = get_checkpoint_name(checkpoint_dir, None, 0, 1, FileType.MODEL) + '.safetensors'
        assert os.path.exists(model_file)

        # Check latest_iteration.txt exists
        tracker_file = get_checkpoint_tracker_filename(tmp_path)
        assert os.path.exists(tracker_file)
        with open(tracker_file, 'r', encoding='utf-8') as f:
            assert f.read().strip() == str(iteration)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_checkpoint_with_async_manager(self, tmp_path, simple_network, optimizer):
        """
        Feature: Test save_checkpoint with async save manager.
        Description: Test async save functionality.
        Expectation: Async save manager should work correctly.
        """
        iteration = 300
        common_info = CommonInfo(epoch_num=1, global_step=iteration)
        async_manager = AsyncSaveManager(async_save=False)

        save_checkpoint(
            iteration=iteration,
            network=simple_network,
            optimizer=optimizer,
            common_info=common_info,
            async_save_manager=async_manager,
            save_checkpoint_path=tmp_path
        )
        # When async_save=False, need to manually call maybe_finalize to execute finalize_fns
        async_manager.maybe_finalize()

        checkpoint_dir = get_checkpoint_iter_dir(tmp_path, iteration)
        assert os.path.exists(checkpoint_dir)

        # Check model file exists
        model_file = get_checkpoint_name(checkpoint_dir, None, 0, 1, FileType.MODEL) + '.safetensors'
        assert os.path.exists(model_file)

        # Check latest_iteration.txt exists
        tracker_file = get_checkpoint_tracker_filename(tmp_path)
        assert os.path.exists(tracker_file)
        with open(tracker_file, 'r', encoding='utf-8') as f:
            assert f.read().strip() == str(iteration)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_checkpoint_with_custom_path(self, tmp_path, simple_network):
        """
        Feature: Test save_checkpoint with custom save_checkpoint_path.
        Description: Test saving checkpoint to specified directory path.
        Expectation: Checkpoint should be saved to the specified path.
        """
        iteration = 400
        custom_path = os.path.join(tmp_path, "custom_checkpoint_dir")
        common_info = CommonInfo(epoch_num=1, global_step=iteration)

        save_checkpoint(
            iteration=iteration,
            network=simple_network,
            optimizer=None,
            common_info=common_info,
            save_checkpoint_path=custom_path
        )
        checkpoint_dir = get_checkpoint_iter_dir(custom_path, iteration)
        assert os.path.exists(checkpoint_dir)

        # Check model file exists
        model_file = get_checkpoint_name(checkpoint_dir, None, 0, 1, FileType.MODEL) + '.safetensors'
        assert os.path.exists(model_file)

        # Check latest_iteration.txt exists
        tracker_file = get_checkpoint_tracker_filename(custom_path)
        assert os.path.exists(tracker_file)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_checkpoint_with_keep_max_num(self, tmp_path, simple_network):
        """
        Feature: Test save_checkpoint with keep_max_num.
        Description: Test checkpoint cleanup with keep_max_num limit.
        Expectation: Old checkpoints should be cleaned up.
        """
        common_info = CommonInfo(epoch_num=1, global_step=600)
        current_ckpt_step_list = []
        # Save multiple checkpoints
        for i in range(3):

            save_checkpoint(
                iteration=600 + i,
                network=simple_network,
                optimizer=None,
                common_info=common_info,
                keep_max_num=2,
                save_checkpoint_path=tmp_path,
                current_ckpt_step_list=current_ckpt_step_list
            )
        # Check that only keep_max_num checkpoints exist
        checkpoint_dirs = [d for d in os.listdir(tmp_path) if d.startswith("iteration_")]
        assert len(checkpoint_dirs) <= 2

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_checkpoint_with_remove_redundancy(self, tmp_path, simple_network):
        """
        Feature: Test save_checkpoint with remove_redundancy.
        Description: Test saving checkpoint with redundancy removal enabled.
        Expectation: Checkpoint should be saved with redundancy removal.
        """
        iteration = 700
        common_info = CommonInfo(epoch_num=1, global_step=iteration)

        save_checkpoint(
            iteration=iteration,
            network=simple_network,
            optimizer=None,
            common_info=common_info,
            remove_redundancy=True,
            save_checkpoint_path=tmp_path
        )
        checkpoint_dir = get_checkpoint_iter_dir(tmp_path, iteration)
        assert os.path.exists(checkpoint_dir)

        # Check model file exists
        model_file = get_checkpoint_name(checkpoint_dir, None, 0, 1, FileType.MODEL) + '.safetensors'
        assert os.path.exists(model_file)


class TestSaveCommonInfo:
    """Test save CommonInfo scenarios."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_and_load_common_info(self, tmp_path):
        """
        Feature: Test save and load CommonInfo.
        Description: Test complete save/load cycle of CommonInfo.
        Expectation: CommonInfo should be saved and loaded correctly.
        """
        common_info = CommonInfo(
            epoch_num=1,
            step_num=100,
            global_step=100,
            loss_scale=2.5,
            global_batch_size=128
        )

        common_path = os.path.join(tmp_path, "common.json")
        common_info.save_common(common_path)
        assert os.path.exists(common_path)

        loaded_info = CommonInfo.load_common(common_path)
        assert loaded_info.epoch_num == 1
        assert loaded_info.step_num == 100
        assert loaded_info.global_step == 100
        assert loaded_info.loss_scale == 2.5
        assert loaded_info.global_batch_size == 128

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_common_info_partial_fields_and_overwrite(self, tmp_path):
        """
        Feature: Test save CommonInfo with partial fields and overwrite.
        Description: Test saving CommonInfo with only some fields set, then overwrite with different values.
        Expectation: Unset fields should be None after loading, and file should be overwritten with new values.
        """
        common_path = os.path.join(tmp_path, "common.json")

        # Test partial fields
        common_info1 = CommonInfo(epoch_num=5, global_step=1000)
        common_info1.save_common(common_path)

        loaded_info = CommonInfo.load_common(common_path)
        assert loaded_info.epoch_num == 5
        assert loaded_info.global_step == 1000
        assert loaded_info.step_num is None
        assert loaded_info.loss_scale is None

        # Test overwrite
        common_info2 = CommonInfo(epoch_num=2, global_step=200)
        common_info2.save_common(common_path)

        loaded_info = CommonInfo.load_common(common_path)
        assert loaded_info.epoch_num == 2
        assert loaded_info.global_step == 200

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_common_info_nonexistent_file(self, tmp_path):
        """
        Feature: Test loading nonexistent CommonInfo file.
        Description: Attempt to load from non-existent file.
        Expectation: FileNotFoundError should be raised.
        """
        nonexistent_path = os.path.join(tmp_path, "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            CommonInfo.load_common(nonexistent_path)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_common_info_invalid_json(self, tmp_path):
        """
        Feature: Test loading invalid JSON CommonInfo file.
        Description: Attempt to load from file with invalid JSON.
        Expectation: ValueError should be raised.
        """
        invalid_path = os.path.join(tmp_path, "invalid.json")
        with open(invalid_path, "w", encoding='utf-8') as f:
            f.write("invalid json content")

        with pytest.raises(ValueError):
            CommonInfo.load_common(invalid_path)


class TestLoadCheckpoint:
    """Test load checkpoint scenarios."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @pytest.mark.parametrize("create_empty_dir,expected_exception", [
        (False, ValueError),
        (True, Exception),
    ])
    def test_load_checkpoint_invalid_scenarios(self, tmp_path, simple_network, create_empty_dir, expected_exception):
        """
        Feature: Test load_checkpoint with invalid scenarios.
        Description: Test loading from non-existent checkpoint or empty directory.
        Expectation: Appropriate exception should be raised.
        """
        if create_empty_dir:
            invalid_ckpt_path = os.path.join(tmp_path, "empty_ckpt")
            os.makedirs(invalid_ckpt_path, exist_ok=True)
        else:
            invalid_ckpt_path = os.path.join(tmp_path, "invalid_ckpt")
        with pytest.raises(expected_exception):
            load_checkpoint(invalid_ckpt_path, simple_network)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_checkpoint_with_network_and_optimizer(self, tmp_path, simple_network, optimizer):
        """
        Feature: Test load_checkpoint with network and optimizer.
        Description: Test loading checkpoint into both network and optimizer.
        Expectation: Parameters should be loaded successfully.
        """
        iteration = 100
        common_info = CommonInfo(epoch_num=1, global_step=iteration)
        save_checkpoint(
            iteration=iteration,
            network=simple_network,
            optimizer=optimizer,
            common_info=common_info,
            save_checkpoint_path=tmp_path
        )
        # Save original parameter values
        original_network_params = {name: param.data.asnumpy().copy()
                                   for name, param in simple_network.parameters_dict().items()}
        original_optimizer_params = {name: param.data.asnumpy().copy()
                                     for name, param in optimizer.parameters_dict().items()}

        # Create a new network and optimizer to load into
        new_network = SimpleNet()
        new_optimizer = nn.Adam(new_network.trainable_params(), learning_rate=0.001)
        load_checkpoint(tmp_path, new_network, optimizer=new_optimizer)

        # Verify network parameters are loaded correctly
        for name, original_value in original_network_params.items():
            loaded_value = new_network.parameters_dict()[name].data.asnumpy()
            np.testing.assert_array_equal(loaded_value, original_value,
                                          err_msg=f"Network parameter {name} was not loaded correctly")
        # Verify optimizer parameters are loaded correctly
        for name, original_value in original_optimizer_params.items():
            loaded_value = new_optimizer.parameters_dict()[name].data.asnumpy()
            np.testing.assert_array_equal(loaded_value, original_value,
                                          err_msg=f"Optimizer parameter {name} was not loaded correctly")

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_checkpoint_network_only(self, tmp_path, simple_network):
        """
        Feature: Test load_checkpoint with network only.
        Description: Test loading checkpoint into network without optimizer.
        Expectation: Network parameters should be loaded successfully.
        """
        iteration = 200
        common_info = CommonInfo(epoch_num=1, global_step=iteration)

        save_checkpoint(
            iteration=iteration,
            network=simple_network,
            optimizer=None,
            common_info=common_info,
            save_checkpoint_path=tmp_path
        )

        # Save original parameter values
        original_params = {name: param.data.asnumpy().copy()
                           for name, param in simple_network.parameters_dict().items()}
        new_network = SimpleNet()
        load_checkpoint(tmp_path, new_network)

        for name, original_value in original_params.items():  # Verify parameters are loaded correctly
            loaded_value = new_network.parameters_dict()[name].data.asnumpy()
            np.testing.assert_array_equal(loaded_value, original_value, err_msg=f"Parameter {name} was not loaded.")


class TestLoadParameters:
    """Test load_parameters scenarios."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_parameters_network_only(self, simple_network):
        """
        Feature: Test load_parameters with network only.
        Description: Test loading parameters into network without optimizer.
        Expectation: Parameters should be loaded successfully.
        """
        state_dict = {
            "fc1.weight": Parameter(Tensor(np.ones((20, 10)), dtype=mstype.float32), name="fc1.weight"),
            "fc1.bias": Parameter(Tensor(np.zeros(20), dtype=mstype.float32), name="fc1.bias"),
            "fc2.weight": Parameter(Tensor(np.ones((1, 20)), dtype=mstype.float32), name="fc2.weight"),
            "fc2.bias": Parameter(Tensor(np.zeros(1), dtype=mstype.float32), name="fc2.bias")
        }
        load_parameters(simple_network, state_dict)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_parameters_with_optimizer(self, simple_network, optimizer):
        """
        Feature: Test load_parameters with network and optimizer.
        Description: Test loading parameters into both network and optimizer.
        Expectation: Parameters should be loaded successfully.
        """
        state_dict = {
            "fc1.weight": Parameter(Tensor(np.ones((20, 10)), dtype=mstype.float32), name="fc1.weight"),
            "fc1.bias": Parameter(Tensor(np.zeros(20), dtype=mstype.float32), name="fc1.bias"),
            "fc2.weight": Parameter(Tensor(np.ones((1, 20)), dtype=mstype.float32), name="fc2.weight"),
            "fc2.bias": Parameter(Tensor(np.zeros(1), dtype=mstype.float32), name="fc2.bias")
        }
        load_parameters(simple_network, state_dict, optimizer)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_parameters_with_state_dict_opt(self, simple_network, optimizer):
        """
        Feature: Test load_parameters with state_dict_opt.
        Description: Test loading with separate optimizer state dict.
        Expectation: Parameters should be loaded correctly.
        """
        state_dict = {"fc1.weight": Parameter(Tensor(np.ones((20, 10)), dtype=mstype.float32), name="fc1.weight"), }
        state_dict_opt = {}
        load_parameters(simple_network, state_dict, optimizer, state_dict_opt)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_parameters_invalid_inputs(self):
        """
        Feature: Test load_parameters with invalid inputs.
        Description: Test error handling for invalid inputs.
        Expectation: Appropriate exceptions should be raised.
        """
        net = SimpleNet()
        # Test with None network
        with pytest.raises(Exception):
            load_parameters(None, {})
        # Test with invalid state_dict
        with pytest.raises(Exception):
            load_parameters(net, "invalid_state_dict")
        # Test with invalid optimizer
        with pytest.raises(Exception):
            load_parameters(net, {}, optimizer="invalid_optimizer")


class TestGetCheckpointPath:
    """Test get_checkpoint_path scenarios."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_checkpoint_path_empty_string(self):
        """
        Feature: Test get_checkpoint_path with empty string.
        Description: Test with empty checkpoint path.
        Expectation: Should return empty string.
        """
        assert get_checkpoint_path("") == ""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @pytest.mark.parametrize("is_file", [False, True])
    def test_get_checkpoint_path_invalid_scenarios(self, tmp_path, is_file):
        """
        Feature: Test get_checkpoint_path with invalid scenarios.
        Description: Test with non-existent path or file instead of directory.
        Expectation: ValueError should be raised.
        """
        if is_file:
            invalid_path = os.path.join(tmp_path, "test.txt")
            with open(invalid_path, "w", encoding='utf-8') as f:
                f.write("test")
        else:
            invalid_path = os.path.join(tmp_path, "non_existent")
        with pytest.raises(ValueError):
            get_checkpoint_path(invalid_path)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_checkpoint_path_with_hf_index(self, tmp_path):
        """
        Feature: Test get_checkpoint_path with HuggingFace index file.
        Description: Test with model.safetensors.index.json.
        Expectation: Should validate HF checkpoint files.
        """
        # Create HF index file
        hf_index_json = os.path.join(tmp_path, "model.safetensors.index.json")
        index_data = {
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.weight": "model-00002-of-00002.safetensors"
            }
        }
        with open(hf_index_json, "w", encoding='utf-8') as f:
            json.dump(index_data, f)

        # Create referenced safetensors files
        os.makedirs(tmp_path, exist_ok=True)
        with open(os.path.join(tmp_path, "model-00001-of-00002.safetensors"), "w", encoding='utf-8') as f:
            f.write("mock")
        with open(os.path.join(tmp_path, "model-00002-of-00002.safetensors"), "w", encoding='utf-8') as f:
            f.write("mock")

        result = get_checkpoint_path(tmp_path)
        assert result == tmp_path

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_checkpoint_path_with_missing_hf_files(self, tmp_path):
        """
        Feature: Test get_checkpoint_path with missing HF files.
        Description: Test when index references non-existent files.
        Expectation: ValueError should be raised.
        """
        hf_index_json = os.path.join(tmp_path, "model.safetensors.index.json")
        index_data = {
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.weight": "nonexistent.safetensors"
            }
        }
        with open(hf_index_json, "w", encoding='utf-8') as f:
            json.dump(index_data, f)

        with pytest.raises(ValueError):
            get_checkpoint_path(tmp_path)


class TestCheckParamForLoad:
    """Test check_the_param_for_load_ckpt scenarios."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_param_valid(self, tmp_path, simple_network):
        """
        Feature: Test check_the_param_for_load_ckpt with valid parameters.
        Description: Test with valid checkpoint path and network.
        Expectation: Should pass without error.
        """
        os.makedirs(tmp_path, exist_ok=True)
        check_the_param_for_load_ckpt(tmp_path, simple_network)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @pytest.mark.parametrize("use_none_network", [True, False])
    def test_check_param_invalid_scenarios(self, tmp_path, simple_network, use_none_network):
        """
        Feature: Test check_the_param_for_load_ckpt with invalid scenarios.
        Description: Test with None network or non-existent checkpoint path.
        Expectation: ValueError should be raised.
        """
        if use_none_network:
            os.makedirs(tmp_path, exist_ok=True)
            ckpt_path = tmp_path
            network = None
        else:
            ckpt_path = os.path.join(tmp_path, "non_existent")
            network = simple_network
        with pytest.raises(ValueError):
            check_the_param_for_load_ckpt(ckpt_path, network)


class TestLoadHfCheckpoint:
    """Test load_hf_checkpoint scenarios."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_hf_checkpoint_invalid_template(self, qwen3_hf_mock_checkpoint_dir):
        """
        Feature: Test load_hf_checkpoint when core network has no WeightTemplate.
        Description: Mock core_network.template=None (e.g. model without register_hf_weight_template).
        Expectation: ValueError is raised and message mentions template registration.
        """
        net = MagicMock()
        net.template = None
        ckpt_dir = str(qwen3_hf_mock_checkpoint_dir)
        with patch("mindformers.checkpoint.checkpoint.get_core_network", return_value=net):
            with pytest.raises(ValueError, match="template"):
                load_hf_checkpoint(ckpt_dir, net)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_hf_checkpoint_with_qwen3_mock_weights(self, qwen3_hf_mock_checkpoint_dir):
        """
        Feature: Test load_hf_checkpoint with Qwen3 dense HF mock weights.
        Description: Disk layout uses Qwen3 dense tensor names; Reshard/metadata/load_parameters are patched
            to drive the Reshard -> WeightTemplate.get_mf_state_dict -> load_parameters path.
        Expectation: load_parameters is called once; pure-rename embedding (HF->MF name + same shape);
            fused QKV linear_qkv shape matches QKVConvertOp (GQA: ng * (dim*nh/ng + 2*dim) rows x hidden).
        """
        ckpt_dir = str(qwen3_hf_mock_checkpoint_dir)

        qwen_cfg = Qwen3Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
        )
        template = WeightTemplate(weight_converters=Qwen3PreTrainedModel.weight_converters)
        template.set_model_config(convert_to_transformer_config(qwen_cfg))

        core = MagicMock()
        core.template = template

        mock_net = MagicMock()
        mock_net.parameters_dict = MagicMock(return_value={
            "embedding.word_embeddings.weight": MagicMock(),
            "decoder.layers.0.self_attention.linear_qkv.weight": MagicMock(),
        })

        reshard_out = _qwen3_reshard_numpy_dict()

        with patch("mindformers.checkpoint.checkpoint.get_core_network", return_value=core):
            with patch(
                "mindformers.checkpoint.checkpoint.get_sharded_tensor_from_cell",
                return_value=_qwen3_minimal_dst_sharded_metas_layer0(),
            ):
                with patch(
                    "mindformers.checkpoint.checkpoint.get_metadata_of_checkpoint",
                    return_value=({}, {}),
                ):
                    with patch("mindformers.checkpoint.checkpoint.ReshardLoader") as mock_rl:
                        mock_rl.return_value.load.return_value = reshard_out
                        with patch("mindformers.checkpoint.checkpoint.load_parameters") as mock_lp:
                            load_hf_checkpoint(ckpt_dir, mock_net, reshard_worker_num=1)
                            mock_lp.assert_called_once()
                            args, _ = mock_lp.call_args
                            state_dict = args[1]
                            assert isinstance(state_dict, dict)
                            assert len(state_dict) > 0

                            qkv_mf = "decoder.layers.0.self_attention.linear_qkv.weight"
                            assert qkv_mf in state_dict
                            assert state_dict[qkv_mf].name == qkv_mf
                            nh = qwen_cfg.num_attention_heads
                            ng = qwen_cfg.num_key_value_heads
                            dim = qwen_cfg.head_dim
                            hidden = qwen_cfg.hidden_size
                            # Same layout as QKVConvertOp._hf_to_mf: concat q/k/v then [*, hidden_size]
                            expected_qkv_rows = ng * (dim * (nh // ng) + dim + dim)
                            assert tuple(state_dict[qkv_mf].shape) == (expected_qkv_rows, hidden)

                            hf_shapes = _qwen3_hf_mock_weight_shapes()
                            # Pure RenameConvertOp sample: HF model.embed_tokens -> MF embedding.word_embeddings, shape unchanged
                            emb_mf = "embedding.word_embeddings.weight"
                            assert emb_mf in state_dict
                            assert state_dict[emb_mf].name == emb_mf
                            assert tuple(state_dict[emb_mf].shape) == hf_shapes["model.embed_tokens.weight"]
