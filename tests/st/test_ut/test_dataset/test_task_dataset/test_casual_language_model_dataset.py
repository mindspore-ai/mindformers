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
"""test casual_language_model_dataset"""
import os
import copy
import tempfile
from contextlib import ExitStack
import unittest
from unittest.mock import patch
import numpy as np
import pytest
from mindformers import MindFormerConfig, CausalLanguageModelDataset
from mindformers.dataset.causal_language_model_dataset import dyn_batch_wrapper, get_input_data_batch_slice_map
from tests.st.test_ut.test_dataset.get_test_data import get_mindrecord_data


class TestCausalLanguageModelDataset(unittest.TestCase):
    """A test class for testing CausalLanguageModelDataset"""

    @classmethod
    def setUpClass(cls):
        cls.resources = ExitStack()
        cls.addClassCleanup(cls.resources.close)
        cls.path = cls.resources.enter_context(tempfile.TemporaryDirectory())
        get_mindrecord_data(cls.path)
        cls.default_config = MindFormerConfig(
            **{
                "data_loader": {
                    "type": "MindDataset",
                    "dataset_dir": cls.path,
                    "shuffle": False
                },
                "batch_size": 1,
                "input_columns": ["input_ids"],
                "seed": 0,
                'prefetch_size': 1,
                'numa_enable': False,
                'drop_remainder': True
            }
        )

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        dataset = CausalLanguageModelDataset(self.default_config)
        for item in dataset.create_tuple_iterator():
            assert item[0].shape == (1, 16)
            assert list(item[0][0].asnumpy()) == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24]
            assert len(item) == 1
            break
        config = copy.deepcopy(self.default_config)
        config.data_loader.dataset_dir = os.path.join(self.path, "test.mindrecord")
        dataset = CausalLanguageModelDataset(config)
        for item in dataset.create_tuple_iterator():
            assert item[0].shape == (1, 16)
            assert list(item[0][0].asnumpy()) == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24]
            assert len(item) == 1
            break

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_dataset_dir(self):
        """test dataset dir logic"""
        config = copy.deepcopy(self.default_config)
        del config.data_loader.dataset_dir
        with pytest.raises(Exception):
            assert CausalLanguageModelDataset(config)
        config.data_loader.dataset_files = [os.path.join(self.path, "test.mindrecord")]
        dataset = CausalLanguageModelDataset(config)
        for item in dataset.create_tuple_iterator():
            assert item[0].shape == (1, 16)
            assert list(item[0][0].asnumpy()) == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24]
            assert len(item) == 1
            break

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_dynamic_batch(self):
        """test dynamic batch logic"""
        config = copy.deepcopy(self.default_config)
        config.dynamic_batch = True
        dataset = CausalLanguageModelDataset(config)
        for item in dataset.create_tuple_iterator():
            assert item[0].shape == (1, 16)
            assert list(item[0][0].asnumpy()) == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24]
            assert len(item) == 1
            break
        res = dyn_batch_wrapper([[-100], [2, 3, 4]], [], divisor=1, remainder=1)
        assert all(res[0][0] == [-100, 0, 0])
        assert all(res[0][1] == [2, 3, 4])
        res = dyn_batch_wrapper([[2, 3, 4], [-100]], [], divisor=1, remainder=1)
        assert all(res[0][0] == [2, 3, 4])
        assert all(res[0][1] == [-100, 0, 0])
        res = dyn_batch_wrapper([[-100], [2, 3, 4]], [], divisor=None, remainder=None)
        assert all(res[0][0] == [-100, 0, 0])
        assert all(res[0][1] == [2, 3, 4])

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_eod_reset(self):
        """test eod_reset logic"""
        config = copy.deepcopy(self.default_config)
        config.eod_reset = True
        config.eod_token_id = 0
        config.output_columns = ["input_ids", "position_id", "attention_mask"]
        dataset = CausalLanguageModelDataset(config)
        for item in dataset.create_tuple_iterator():
            assert item[0].shape == (1, 16)
            assert list(item[0][0].asnumpy()) == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24]
            assert list(item[1][0].asnumpy()) == [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            assert len(item) == 3
            assert item[2].shape == (1, 15, 15)
            break

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_input_data_batch_slice_map(self):
        """test get_input_data_batch_slice_map logic"""
        input_ids = np.asarray([[44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24]], dtype=np.int32)
        res = get_input_data_batch_slice_map(input_ids, eod_token_id=0, dis=1)
        assert len(res) == 3
        assert res[0].shape == (1, 16)
        assert res[1].shape == (1, 15)
        assert res[2].shape == (1, 15, 15)
        assert list(res[0][0]) == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24]
        assert list(res[1][0]) == [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert list(res[2][-1][-1]) == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_other(self):
        """test other logic"""
        from mindformers.dataset.dataloader import build_dataset_loader
        config = copy.deepcopy(self.default_config)
        data_path = config.data_loader.pop("dataset_dir")
        dataloader = build_dataset_loader(
            config.data_loader, default_args={
                "dataset_files": [os.path.join(data_path, "test.mindrecord")],
                "num_shards": 1,
                "shard_id": 0,
                "columns_list": config.input_columns
            }
        )
        config.data_loader = dataloader
        dataset = CausalLanguageModelDataset(config)
        for item in dataset.create_tuple_iterator():
            assert list(item[0][0].asnumpy()) == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24]
            break

    @patch.object(CausalLanguageModelDataset, "_is_data_parallel", return_value=True)
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_is_data_parallel(self, mock_is_data_parallel):
        """test is data_parallel logic"""
        config = copy.deepcopy(self.default_config)
        config.eod_reset = True
        config.eod_token_id = 0
        config.output_columns = ["input_ids", "position_id", "attention_mask"]
        dataset = CausalLanguageModelDataset(config)
        mock_is_data_parallel.assert_called_once_with()
        for item in dataset.create_tuple_iterator():
            assert list(item[0][0].asnumpy()) == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24]
            break

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_token_monitor(self):
        """test token_monitor does not change data"""
        config = copy.deepcopy(self.default_config)
        config.token_monitor = True
        dataset = CausalLanguageModelDataset(config)
        for item in dataset.create_tuple_iterator():
            assert item[0].shape == (1, 16)
            assert list(item[0][0].asnumpy()) == [44, 47, 53, 0, 3, 59, 3, 39, 9, 19, 21, 50, 36, 23, 6, 24]
            assert len(item) == 1
            break
