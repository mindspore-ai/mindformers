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
"""
Run mcore deepseekv3 Graph-mode training with balanced checkpoint saving, either from
scratch or resumed from a saved checkpoint.
"""
import os
import argparse
from types import MethodType

import mindspore as ms
from data_gen_utils import get_dataset, generate_weight

from mindformers import build_context, MindFormerConfig
from mindformers.trainer import Trainer

CUR_DIR = os.path.dirname(__file__)
TRAIN_STEPS = 20
SAVE_INTERVAL = 10

ms.set_context(mode=ms.GRAPH_MODE)


def build_config(save_path, load_path):
    """
    Build a dp=mp=pp=ep=2 config with balanced saving of the model and optimizer.

    Dropout and the router expert bias are switched off: the RNG state and `expert_bias` /
    `expert_load` are not part of the checkpoint, so with them on a resumed run would not
    reproduce an uninterrupted one bit for bit.
    """
    config = MindFormerConfig(f'{CUR_DIR}/deepseekv3_train.yaml')
    config.print_separate_loss = False
    config.train_precision_sync = True
    config.pretrained_model_dir = CUR_DIR
    config.output_dir = os.path.join(save_path, 'output')
    config.context.deterministic = 'ON'
    config.parallel.full_batch = False
    config.parallel.dataset_strategy = 'full_batch'
    config.parallel.strategy_ckpt_save_file = os.path.join(save_path, 'ckpt_strategy.ckpt')
    config.model.model_config.hidden_dropout = 0.0
    config.model.model_config.attention_dropout = 0.0
    config.model.model_config.moe_router_enable_expert_bias = False
    config.checkpoint = MindFormerConfig(
        save_path=os.path.join(save_path, 'checkpoint'),
        save_max=2,
        save_interleaved_steps=SAVE_INTERVAL,
        save_remove_redundancy=True,
        no_save_optim=False,
        async_save=False,
        prefix='ds3',
        load_path=load_path,
        no_load_optim=not load_path,
    )
    config.callbacks = [
        MindFormerConfig(type='MFLossMonitor', per_print_times=1),
        MindFormerConfig(type='CheckpointMonitor', checkpoint_format='safetensors'),
    ]
    return config


def ds3_train(save_path, load_path):
    """Train TRAIN_STEPS steps, saving every SAVE_INTERVAL steps; resume from `load_path` if given."""
    ms.set_seed(0)
    config = build_config(save_path, load_path)
    build_context(config)

    construct_args_key = ['input_ids', 'labels']
    model_config = config.model.model_config
    # batch 4 = micro_batch_num (2) x data_parallel (2)
    dataset = get_dataset(model_config.seq_length, model_config.vocab_size, 4, TRAIN_STEPS)

    task_trainer = Trainer(task='text_generation', args=config, train_dataset=dataset)
    task_trainer.config.train_dataset.input_columns = construct_args_key
    task_trainer.config.train_dataset.construct_args_key = construct_args_key

    def create_network(self, default_args):
        network = type(self).create_network(self, default_args)
        ms.load_param_into_net(network, generate_weight(network))
        return network

    task_trainer.trainer.create_network = MethodType(create_network, task_trainer.trainer)
    task_trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, required=True, help='directory of this run.')
    parser.add_argument('--load_path', type=str, default='', help='checkpoint to resume from.')
    args = parser.parse_args()
    ds3_train(args.save_path, args.load_path)
