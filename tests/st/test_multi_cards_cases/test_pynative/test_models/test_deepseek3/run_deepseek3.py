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
"""Run DeepSeek3 training"""

import argparse
import random

import numpy as np
import mindspore as ms

from mindformers.pynative.trainer import Trainer as PynativeTrainer

SEED = 42


def main():
    """Parse config path from command line arguments and launch DeepSeek3 training."""
    parser = argparse.ArgumentParser(description="Run DeepSeek3 training")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml file")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    ms.set_seed(SEED)

    trainer = PynativeTrainer(config=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
