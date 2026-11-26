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
"""test training dataloader"""
from contextlib import ExitStack
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from mindformers import PreTrainedTokenizerFast
from mindformers.dataset.dataloader.training_dataloader import TrainingDataset, run_cmd


class TestTrainingDataSet(unittest.TestCase):
    """ A test class for testing Training dataset"""

    @classmethod
    def setUpClass(cls):
        resources = ExitStack()
        cls.addClassCleanup(resources.close)
        cls.path = resources.enter_context(tempfile.TemporaryDirectory())
        text = "one two three four five six seven eight nine"
        tokens = ["[UNK]", "[PAD]", *text.split()]
        backend = Tokenizer(WordLevel({token: index for index, token in enumerate(tokens)}, unk_token="[UNK]"))
        backend.pre_tokenizer = WhitespaceSplit()
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
        with open(os.path.join(cls.path, "train.json"), "w", encoding="utf-8") as sample:
            json.dump({"input": [text]}, sample)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_default(self):
        """test default logic"""
        dataset = TrainingDataset(
            self.path,
            column_names=["input_ids", "attention_mask"],
            file_format="json",
            tokenizer=self.tokenizer,
            max_length=8,
            text_col="input",
            shuffle=False,
            process_num=1,
        )
        assert hasattr(dataset, "current_samples")
        assert dataset.current_samples[0].tolist() == [2, 3, 4, 5, 6, 7, 8, 9]
        assert dataset.column_names == ['input_ids', 'attention_mask']
        assert dataset.text_col == "input"

    # pylint: disable=W0212
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_other(self):
        """test other logic"""
        dataset = TrainingDataset(
            os.path.join(self.path, "train.json"),
            column_names=["input_ids", "attention_mask"],
            file_format="json",
            tokenizer=self.tokenizer,
            max_length=8,
            text_col="input",
            shuffle=False,
            process_num=1,
        )
        res = next(dataset)
        assert res == ([2, 3, 4, 5, 6, 7, 8, 9], [1] * 8)
        res = dataset._check_format(dataset_dir=dataset.dataset_dir, file_format=dataset.format)
        assert res == "json"
        res = dataset._check_format(dataset_dir=dataset.dataset_dir, file_format="")
        assert res == "json"
        with pytest.raises(ValueError):
            assert dataset._check_format(dataset_dir=self.path, file_format="txt")
        res = dataset._tokenizer_func("one two three")
        assert res == [2, 3, 4]
        res = dataset._get_all_samples_number()
        assert res == 1
        dataset._reset_iter_index()
        assert dataset.iter_index == 0
        assert dataset.current_samples[0].tolist() == [2, 3, 4, 5, 6, 7, 8, 9]


class MockRun:
    """mock run"""
    def __init__(self):
        self.returncode = 0
        self.stderr = "stderr"
        self.stdout = "stdout"


@patch("subprocess.run")
@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_run_cmd(mock_get):
    """
    Feature: training_dataloader.run_cmd
    Description: test run_cmd function
    Expectation: success
    """
    mockrun = MockRun()
    mock_get.return_value = mockrun
    assert run_cmd("mock input", pipeline=True) == (True, "stdout")
    mockrun.returncode = 1
    mock_get.return_value = mockrun
    assert run_cmd("mock input") == (False, "stderr")
    mockrun.stderr = "No such file"
    assert run_cmd("mock input") == (False, "No such file")
    mockrun.stderr = "Files exists"
    assert run_cmd("mock input") == (False, "Files exists")
    mockrun.stderr = "permission denied"
    assert run_cmd("mock input") == (False, "permission denied")
