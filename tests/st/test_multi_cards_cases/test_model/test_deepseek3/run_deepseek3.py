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
"""
Test module for testing the paralleled mcore deepseek3 interface used for mindformers.
"""
import os
import argparse

import mindspore as ms

from mindformers import build_context, MindFormerConfig
from mindformers.trainer import Trainer

from tests.st.training_checker import TrainingChecker

from data_gen_utils import get_tnd_dataset, get_dataset

CUR_DIR = os.path.dirname(__file__)

ms.set_context(mode=ms.GRAPH_MODE)


def ds3_train(config, dataset, construct_args_key, checker_config):
    """set model train."""
    callback = TrainingChecker(**checker_config)
    task_trainer = Trainer(task="text_generation",
                           args=config,
                           train_dataset=dataset,
                           callbacks=callback)

    task_trainer.config.train_dataset.construct_args_key = construct_args_key

    task_trainer.train()
    if checker_config.get('experiment_mode'):
        callback.get_experiment_results()


def parallel_train_dp2_mp2_ep2():
    """test mcore deepseekv3 train in dp=mp=ep=2."""
    ms.set_seed(0)
    config = MindFormerConfig(f'{CUR_DIR}/deepseekv3_train.yaml')
    config.print_separate_loss = False
    config.train_precision_sync = True
    config.pretrained_model_dir = CUR_DIR
    config.runner_config.sink_mode = False
    config.parallel.full_batch = True
    config.parallel.dataset_strategy = 'full_batch'
    config.parallel_config.pipeline_stage = 1
    build_context(config)

    construct_args_key = ['input_ids', 'labels']
    model_config = config.model.model_config
    dataset = get_dataset(model_config.seq_length, model_config.vocab_size, 4, 20)

    loss_std = [12.940836, 12.92868, 12.9460125, 12.938551, 12.937703, 12.942499, 12.939261, 12.945714, 12.913517,
                12.936413, 12.884894, 12.926766, 12.9116, 12.922249, 12.893213, 12.89359, 12.877649, 12.854286,
                12.8743305, 12.83889]
    checker_config = {
        'loss_list_std': loss_std,
        'experiment_mode': False,
        'micro_batch_num': 1,
        'micro_batch_interleave_num': 1
    }
    ds3_train(config, dataset, construct_args_key, checker_config)


def parallel_train_dp2_pp2_ep2_tnd():
    """test mcore deepseekv3 train in dp=pp=ep=2 with TND layout."""
    ms.set_seed(0)
    config = MindFormerConfig(f'{CUR_DIR}/deepseekv3_train.yaml')
    config.print_separate_loss = False
    config.train_precision_sync = True
    config.pretrained_model_dir = CUR_DIR
    config.runner_config.sink_mode = True
    config.parallel.full_batch = False
    dp = config.parallel_config.data_parallel
    config.parallel_config.model_parallel = 1
    config.parallel_config.use_seq_parallel = False
    config.parallel.dataset_strategy = [[dp, 1], [dp, 1], [dp, 1]]
    config.model.model_config.offset = [1, 0]
    build_context(config)

    construct_args_key = ['input_ids', 'labels', 'actual_seq_len']
    model_config = config.model.model_config
    dataset = get_tnd_dataset(
        model_config.seq_length,
        model_config.vocab_size,
        config.parallel_config.micro_batch_num,
        batch_size=4, step_num=20
    )

    loss_std = [12.349523544311523, 12.358807563781738, 12.297042846679688, 12.314142227172852, 12.29040241241455,
                12.324695587158203, 12.3829984664917, 12.268146514892578, 12.32144832611084, 12.293285369873047,
                12.287753105163574, 12.266066551208496, 12.297396659851074, 12.248311996459961, 12.235709190368652,
                12.227277755737305, 12.152572631835938, 12.174047470092773, 12.204164505004883, 12.108986854553223]
    checker_config = {
        'loss_list_std': loss_std,
        'experiment_mode': False,
        'micro_batch_num': 2,
        'micro_batch_interleave_num': 1
    }
    ds3_train(config, dataset, construct_args_key, checker_config)


def parallel_train_dp2_mp2_ep2_calculate_per_token_loss_and_print_seperate_loss():
    """test mcore deepseekv3 train in dp=mp=ep=2."""
    ms.set_seed(0)
    config = MindFormerConfig(f'{CUR_DIR}/deepseekv3_train.yaml')
    config.print_separate_loss = True
    config.calculate_per_token_loss = True

    config.train_precision_sync = True
    config.pretrained_model_dir = CUR_DIR
    config.runner_config.sink_mode = False
    config.parallel.full_batch = True
    config.parallel.dataset_strategy = 'full_batch'
    config.parallel_config.pipeline_stage = 1
    build_context(config)

    construct_args_key = ['input_ids', 'labels']
    model_config = config.model.model_config
    dataset = get_dataset(model_config.seq_length, model_config.vocab_size, 4, 20)

    loss_std = [12.940073, 12.927917, 12.945321, 12.937913, 12.936846, 12.941753, 12.938346, 12.945015, 12.91268,
                12.935577, 12.884249, 12.925993, 12.910951, 12.921507, 12.8924885, 12.892721, 12.876872, 12.8534565,
                12.873688, 12.838091]

    checker_config = {
        'loss_list_std': loss_std,
        'experiment_mode': False,
        'micro_batch_num': 1,
        'micro_batch_interleave_num': 1
    }
    ds3_train(config, dataset, construct_args_key, checker_config)


TEST_MAP = {
    'parallel_train_dp2_mp2_ep2': parallel_train_dp2_mp2_ep2,
    'parallel_train_dp2_pp2_ep2_tnd': parallel_train_dp2_pp2_ep2_tnd,
    "parallel_train_dp2_mp2_ep2_calculate_per_token_loss_and_print_seperate_loss":
        parallel_train_dp2_mp2_ep2_calculate_per_token_loss_and_print_seperate_loss,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of deepseek model.')

    args = parser.parse_args()
    TEST_MAP[args.mode]()
