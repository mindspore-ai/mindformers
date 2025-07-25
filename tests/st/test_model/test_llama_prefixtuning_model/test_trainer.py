# Copyright 2024 Huawei Technologies Co., Ltd
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
"""
Test module for testing the llama_ interface used for mindformers.
How to run this:
pytest /tests/st/test_model/test_llama_prefixtuning_model/test_trainer.py
"""
import os
import numpy as np
import pytest

import mindspore as ms
from mindspore.dataset import GeneratorDataset
from mindformers.models.llama.llama import LlamaForCausalLM
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.models.llama.llama_tokenizer import LlamaTokenizer
from mindformers.pet.pet_config import PrefixTuningConfig
from mindformers.pet import get_pet_model
from mindformers import Trainer, TrainingArguments

ms.set_context(mode=0)

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


def generator_train():
    """train dataset generator"""
    seq_len = 513
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    labels = np.ones_like(input_ids)
    train_data = (input_ids, labels)
    for _ in range(16):
        yield train_data


def generator_eval():
    """eval dataset generator"""
    seq_len = 512
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    train_data = (input_ids)
    for _ in range(16):
        yield train_data


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestLlama2prefixtunigTrainerMethod:
    """A test class for testing pipeline."""

    def setup_method(self):
        """init task trainer."""
        args = TrainingArguments(batch_size=2)
        train_dataset = GeneratorDataset(generator_train, column_names=["input_ids", "labels"])
        eval_dataset = GeneratorDataset(generator_eval, column_names=["input_ids", "labels"])
        train_dataset = train_dataset.batch(batch_size=2)
        eval_dataset = eval_dataset.batch(batch_size=2)

        # padded_vocab_size=64793 to avoid tokenizer index violation
        model_config = LlamaConfig(num_layers=2, hidden_size=32, num_heads=2, seq_length=512)
        model_config.pet_config = PrefixTuningConfig(prefix_token_num=2, mid_dim=512, dropout_rate=0.01)
        model = LlamaForCausalLM(model_config)
        model = get_pet_model(model, model_config.pet_config)
        tokenizer = LlamaTokenizer(vocab_file=f"{root_path}/utils/llama2_tokenizer/tokenizer.model")

        self.task_trainer = Trainer(task='text_generation',
                                    model=model,
                                    model_name='llama2_7b',
                                    args=args,
                                    train_dataset=train_dataset,
                                    eval_dataset=eval_dataset,
                                    tokenizer=tokenizer)

    @pytest.mark.run(order=1)
    def test_finetune(self):
        """
        Feature: Trainer.finetune()
        Description: Test trainer for finetune.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.config.runner_config.epochs = 1
        self.task_trainer.finetune()

    @pytest.mark.run(order=2)
    def test_predict(self):
        """
        Feature: Trainer.predict()
        Description: Test trainer for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        self.task_trainer.predict(input_data="hello world!", max_length=20, repetition_penalty=1,
                                  top_k=3, top_p=1)
