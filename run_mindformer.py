# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Run MindFormer."""
import argparse
import os
import sys

from mindformers.pynative.trainer import Trainer
from mindformers.tools.logger import logger

STATIC_MODE_ERROR = (
    "GRAPH_MODE (static graph mode) has been deprecated and is no longer supported. "
    "Use a PyNative YAML config and launch with "
    "`python run_mindformer.py --config <pynative_config.yaml>` without --mode. "
    "Remove static mode settings (mode: 0 or context.mode: 0) from the YAML."
)


def main(args: argparse.Namespace):
    """Validate CLI arguments and launch the PyNative Trainer from a YAML path."""
    if args.mode == 0:
        raise ValueError(STATIC_MODE_ERROR)
    if args.mode not in (None, 1):
        raise ValueError("Only PyNative mode is supported. Omit --mode to launch.")

    config = Trainer._init_config(args.config)
    for mode in (getattr(config, "mode", None), config.context.mode):
        if mode == 0 or str(mode).upper() in ("0", "GRAPH_MODE"):
            raise ValueError(STATIC_MODE_ERROR)

    if args.mode == 1:
        logger.warning("--mode 1 is deprecated and has no effect. PyNative mode is now the default; "
                       "omit --mode when launching run_mindformer.py.")
    logger.info("Running MindFormers in PYNATIVE_MODE.")
    trainer = Trainer(config.to_dict())
    if trainer.is_inference:
        trainer.inference()
    else:
        trainer.train()


if __name__ == "__main__":
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Launch PyNative training or inference. Configure runtime options in the YAML file.",
        allow_abbrev=False)
    parser.add_argument(
        '--config',
        default=None,
        required=True,
        help='YAML config files')
    parser.add_argument(
        '--mode', default=None, type=int,
        help='Deprecated compatibility option. PyNative is the default. '
             '--mode 1 has no effect and emits a deprecation warning; --mode 0 is rejected.')
    parser.add_argument(
        '--register_path', default=None, type=str,
        help='the register path of outer API.')
    args_ = parser.parse_args()

    if args_.config is not None and not os.path.isabs(args_.config):
        args_.config = os.path.join(work_path, args_.config)

    if args_.register_path is not None:
        if not os.path.isabs(args_.register_path):
            args_.register_path = os.path.join(work_path, args_.register_path)
        # Setting Environment Variables: REGISTER_PATH For Auto Register to Outer API
        os.environ["REGISTER_PATH"] = args_.register_path
        if args_.register_path not in sys.path:
            sys.path.append(args_.register_path)

    main(args_)
