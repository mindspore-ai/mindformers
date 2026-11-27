# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Base Trainer."""
import os
import re
import socket
from typing import Optional, Union, List

import numpy as np
from PIL.Image import Image
import mindspore as ms
from mindspore import Tensor
from mindspore import get_ckpt_path_with_strategy
from mindspore.communication import get_rank
from mindspore.communication.management import get_group_size
from mindspore.train import Callback
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.engine.datasets import Dataset
from mindspore.nn import Optimizer, Cell
from mindspore.nn.utils import no_init_parameters

from mindformers.mindformer_book import MindFormerBook
from mindformers.core.callback import build_callback
from mindformers.core.context.build_context import (
    get_context as mf_get_context,
    set_context as mf_set_context,
)
from mindformers.dataset import build_dataset, check_dataset_config, check_dataset_iterable, BaseDataset
from mindformers.models import build_network, build_processor, PreTrainedModel, PreTrainedTokenizerBase, BaseImageProcessor
from mindformers.tools.register import MindFormerConfig
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params, get_real_group_size
from mindformers.core.callback.callback import (
    MFLossMonitor,
    TrainingStateMonitor,
    CheckpointMonitor,
    ExpertMigrateCallback,
    ColdHotExpertMonitor,
    TopkBiasBalanceCallback
)
from .training_args import TrainingArguments
from .utils import get_last_checkpoint, set_seed

# pylint: disable=import-outside-toplevel
SUPPORT_TASKS = MindFormerBook().get_trainer_support_task_list()
SUPPORT_MODEL_NAMES = MindFormerBook().get_model_name_support_list()
CURRENT_PROJECT_PATH = MindFormerBook().get_project_path()
DEFAULT_CONFIG_DIR = 'configs'
NEED_MERGES_FILE_TOKENIZERS = ["Qwen2Tokenizer"]
CALLBACK_HAS_SORT = [
    MFLossMonitor, TrainingStateMonitor, ExpertMigrateCallback, ColdHotExpertMonitor,
    TopkBiasBalanceCallback, CheckpointMonitor
]


class BaseTrainer:
    r"""Base Task Trainer.
    Args:
        task (str): The task name supported.
        model_name (str): The model name of Task-Trainer. Default: None
    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.
    """

    def __init__(self, task: str = None, model_name: str = None):

        try:
            host_name = socket.gethostname()
            host_ip = socket.gethostbyname(host_name)
            logger.info(f"host_name: {host_name}, host_ip: {host_ip}")
        except socket.gaierror:
            logger.warning("Cannot get host name and host ip.")

        if model_name is None:
            model_name = "model name unspecified."
        if task is None:
            task = "task name unspecified."
        logger.info("Now Running Task is: %s, Model is: %s", task, model_name)

        self.model_name = model_name
        self.task = task
        self.config = None
        self.default_task_config = None

        self.train_dataset = None
        self.eval_dataset = None
        self.network = None
        self.optimizer = None
        self.image_processor = None
        self.audio_processor = None
        self.tokenizer = None
        self.callbacks = None
        self.eval_callbacks = None
        self.compute_metrics = None
        self.kwargs = None
        self.pipeline_task = None
        self.global_batch_size = None

        self.network_delay_inited = False
        self.optimizer_delay_inited = False

        self.lr_scheduler = None
        self.grouped_lr_scheduler = None

        if task not in SUPPORT_TASKS.keys():
            logger.warning("Input task name is not in the supported list or unspecified.")

        if task in SUPPORT_TASKS.keys() and model_name not in SUPPORT_TASKS.get(task).keys():
            model_name_support_list = list(MindFormerBook().get_model_name_support_list_for_task(task))
            model_name_support_list.sort()
            logger.warning("Input model name is not in the supported list or unspecified.")
            logger.warning("See the list of supported task and model name: %s", model_name_support_list)
            logger.warning("The default model config: %s will now be used for the %s task ",
                           SUPPORT_TASKS.get(self.task).get("common"), task)
            self.model_name = "common"

    def set_config(self,
                   config: Optional[Union[dict, str, TrainingArguments]] = None,
                   is_full_config: bool = False):
        """Set the task config for task trainer."""
        if config is not None:
            if is_full_config:
                self.config = config
            else:
                self.setup_task_config()
                self.config = self._merge_config(config)
        else:
            self.setup_task_config()
            self.config = self.default_task_config
        if os.environ.get("RUN_MODE") != "predict":
            self._check_grad_accumulation_steps()
            self._check_global_batch_size_for_auto_parallel()

        return self.config

    def setup_task_config(self):
        """Set up the default task config."""
        task_config = None
        if self.task in SUPPORT_TASKS.keys() and self.model_name in SUPPORT_TASKS.get(self.task).keys():
            task_config = MindFormerConfig(SUPPORT_TASKS.get(self.task).get(self.model_name))

        if isinstance(task_config, MindFormerConfig):
            self.default_task_config = task_config
        else:
            logger.warning("If the default config arguments is not specified,"
                           "you must define the required model, optimizer, and so on"
                           "in the train or evaluate or predict attribute function.")

    def _check_global_batch_size_for_auto_parallel(self):
        """Check global batch size in auto parallel mode."""
        batch_size = self.config.runner_config.batch_size
        self.config.runner_config.mini_batch_size = batch_size
        gradient_accumulation_steps = self.config.runner_config.gradient_accumulation_steps
        dp = self.config.parallel_config.data_parallel
        micro_batch_num = self.config.parallel_config.micro_batch_num
        micro_batch_interleave_num = self.config.micro_batch_interleave_num
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        full_batch = ms.get_auto_parallel_context("full_batch")
        ds_stra = ms.get_auto_parallel_context("dataset_strategy")
        pp = self.get_pipeline_stages()

        if parallel_mode in ["semi_auto_parallel", "auto_parallel"]:
            if pp == 1 and micro_batch_num > 1:
                logger.warning("When pipeline parallel is not enabled, "
                               "config.parallel_config.micro_batch_num does not take effect. Reset it to 1.")
                micro_batch_num = self.config.parallel_config.micro_batch_num = 1
            if full_batch:
                if ds_stra != 'full_batch':
                    logger.warning(f"full_batch=True only supports dataset_strategy='full_batch', "
                                   f"reset dataset_strategy {ds_stra} to 'full_batch'.")
                    ms.set_auto_parallel_context(dataset_strategy='full_batch')

                if pp > 1:
                    self.global_batch_size = batch_size * dp * micro_batch_num * micro_batch_interleave_num
                    logger.info("Pipeline parallel was opened: pipeline_stages = %s, full batch is True, "
                                "gradient_accumulation_steps will not take effect in pipeline parallel, "
                                "global batch size will be changed: "
                                "global_batch_size = "
                                "batch_size * data_parallel * micro_batch_num * micro_batch_interleave_num "
                                "= %s = %s * %s * %s * %s).",
                                pp, self.global_batch_size, batch_size, dp, micro_batch_num,
                                micro_batch_interleave_num)
                    self.config.runner_config.batch_size = self.global_batch_size
                else:
                    self.global_batch_size = batch_size * dp * micro_batch_interleave_num * gradient_accumulation_steps
                    logger.info("The current parallel mode is %s, full batch is True,"
                                "so global batch size will be changed: "
                                "global_batch_size = batch_size * data_parallel * micro_batch_interleave_num "
                                "* gradient_accumulation_steps = %s = %s * %s * %s * %s",
                                parallel_mode, self.global_batch_size, batch_size, dp, micro_batch_interleave_num,
                                gradient_accumulation_steps)
                    self.config.runner_config.batch_size = self.global_batch_size
            else:  # full_batch = False
                if not isinstance(ds_stra, (tuple, list)):
                    raise ValueError("If set full_batch=False, dataset_strategy must be set as 'tuple', "
                                     "such as ((dp, 1), ).")
                ds_stra_dp = ds_stra[0][0]
                if dp != ds_stra_dp:
                    raise ValueError(f"data_parallel {dp} should be equal to dataset_strategy[0][0] {ds_stra_dp}.")

                if pp > 1:
                    per_batch_size = batch_size * micro_batch_num * micro_batch_interleave_num
                    self.global_batch_size = per_batch_size * dp
                    logger.info("Pipeline parallel was opened: pipeline_stages = %s, full batch is False, "
                                "gradient_accumulation_steps will not take effect in pipeline parallel, "
                                "batch size per card will be changed: "
                                "per_batch_size = batch_size * micro_batch_num * micro_batch_interleave_num "
                                "= %s = %s * %s * %s).",
                                pp, per_batch_size, batch_size, micro_batch_num,
                                micro_batch_interleave_num)
                    logger.info("global_batch_size = per_batch_size * data_parallel = %s * %s = %s",
                                per_batch_size, dp, self.global_batch_size)
                    self.config.runner_config.batch_size = per_batch_size
                else:
                    per_batch_size = batch_size * micro_batch_interleave_num * gradient_accumulation_steps
                    self.global_batch_size = per_batch_size * dp
                    logger.info("The current parallel mode is %s, full batch is False, "
                                "batch size per card will be changed: "
                                "per_batch_size = batch_size * micro_batch_interleave_num * "
                                "gradient_accumulation_steps = %s = %s * %s * %s).",
                                parallel_mode, per_batch_size, batch_size, micro_batch_interleave_num,
                                gradient_accumulation_steps)
                    logger.info("global_batch_size = per_batch_size * data_parallel = %s * %s = %s",
                                per_batch_size, dp, self.global_batch_size)
                    self.config.runner_config.batch_size = per_batch_size
        else:
            logger.info("The current parallel mode is %s, batch size per card will not be changed: "
                        "batch_size_per_card = %s",
                        parallel_mode, batch_size)
            self.global_batch_size = batch_size * get_real_group_size() * gradient_accumulation_steps
            logger.info(
                "global_batch_size = batch_size_per_card * device_num * gradient_accumulation_steps "
                "= %s = %s * %s * %s",
                self.global_batch_size, batch_size, get_real_group_size(), gradient_accumulation_steps)
            self.config.runner_config.batch_size = batch_size * gradient_accumulation_steps
            self.config.parallel_config.data_parallel = 1
            self.config.parallel_config.model_parallel = 1
            self.config.parallel_config.context_parallel = 1
            self.config.parallel_config.expert_parallel = 1
            self.config.parallel_config.pipeline_stage = 1
            self.config.parallel_config.micro_batch_num = 1
            self.config.parallel_config.use_seq_parallel = False
            logger.info("parallel_config will be change to default config: %s.",
                        self.config.parallel_config)
        self.config.runner_config.global_batch_size = self.global_batch_size

    def _check_grad_accumulation_steps(self):
        """check the gradient accumulation steps."""
        if self.config.runner_config.gradient_accumulation_steps is None:
            self.config.runner_config.gradient_accumulation_steps = 1
        if not isinstance(self.config.runner_config.gradient_accumulation_steps, int) or \
            isinstance(self.config.runner_config.gradient_accumulation_steps, bool):
            raise ValueError("gradient_accumulation should be integer but got "
                             f"{type(self.config.runner_config.gradient_accumulation_steps)}")
        if not self.config.runner_config.gradient_accumulation_steps >= 1:
            raise ValueError("gradient_accumulation should be greater or equal than 1, "
                             f"but got {self.config.runner_config.gradient_accumulation_steps}")
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        pp = self.get_pipeline_stages()
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"] and pp > 1 \
            and self.config.runner_config.gradient_accumulation_steps > 1:
            logger.warning("gradient_accumulation_steps will not take effect when using pipeline parallel, "
                           "reset the gradient_accumulation_steps from %s to 1.",
                           self.config.runner_config.gradient_accumulation_steps)
            self.config.runner_config.gradient_accumulation_steps = 1
        # grad accumulation not supported in data parallel/standalone mode for now
        if self.config.runner_config.gradient_accumulation_steps > 1 and \
            parallel_mode not in ["semi_auto_parallel", "auto_parallel"]:
            logger.warning("gradient_accumulation_steps currently need to be used in semi_auto_parallel mode, "
                           "but got %s mode, please check your runner config and parallel config. "
                           "Reset the gradient_accumulation_steps from %s to 1. ",
                           parallel_mode, self.config.runner_config.gradient_accumulation_steps)
            self.config.runner_config.gradient_accumulation_steps = 1
        if self.config.runner_config.gradient_accumulation_steps == 1 and pp == 1:
            logger.warning("The Lazy Inline compilation acceleration feature "
                           "only works with using gradient_accumulation_steps > 1 "
                           "when not in pipeline parallel mode (pipeline_stage = 1). "
                           "Current pipeline stage=1 but gradient_accumulation_steps=1, "
                           "the feature is disabled by default.")
            self.config.model.model_config.disable_lazy_inline = True

    @staticmethod
    def _check_training_network_no_use_past(network):
        if network is not None and hasattr(network.config, "use_past") and network.config.use_past:
            raise ValueError("In training process, network should be configured to use_past=False, "
                             f"but got use_past={network.config.use_past}")

    def _reset_dataset_batch_size(self):
        """Reset dataset batch size according to the global batch size of runner config."""
        check_dataset_config(self.config)

    def _merge_config(self, args):
        """Merge config from default task config."""
        if self.default_task_config is None:
            logger.warning("default task config is None, you will not be able to merge config parameters.")
            return args

        if isinstance(args, dict):
            self.default_task_config.merge_from_dict(args)
        elif isinstance(args, str):
            if not (os.path.realpath(args) and os.path.exists(args)):
                raise FileNotFoundError(f"config path must be exist, but get {args}.")
            if not args.endswith(('.yaml', '.yml')):
                raise ValueError(f"config file must be end with .yaml or .yml, but get {args}")
            self.default_task_config = MindFormerConfig(args)
        elif isinstance(args, TrainingArguments):
            args.convert_args_to_mindformers_config(self.default_task_config)
        else:
            logger.warning(
                "The type of config parameter to merge is not supported, "
                "currently supported types are [dict, config_path(str), TrainingArguments], "
                "but get %s", type(args))
        return self.default_task_config

    def create_dataset(self, is_train: bool = True, default_args: dict = None):
        """Create the dataset for training or evaluate."""
        dataset = self.train_dataset if is_train else self.eval_dataset
        dataset_task = self.config.train_dataset_task if is_train else self.config.eval_dataset_task

        if isinstance(dataset, (BaseDataset, Dataset)):
            return dataset

        if dataset is None:
            dataset = build_dataset(dataset_task, default_args=default_args)
        elif check_dataset_iterable(dataset):
            dataset_task.type = 'GeneralDataset'
            default_args = {} if default_args is None else default_args
            default_args["dataset"] = dataset
            dataset = build_dataset(dataset_task, default_args=default_args)
        else:
            raise ValueError("Dataset should be Dataset, iterator, iterable class which has `__iter__`, "
                             f"iterable class which has  `__get_item__` and `__len__`, but get {dataset}.")
        return dataset

    def create_train_dataset(self, default_args: dict = None):
        """Create the train dataset for training."""
        logger.info(".........Build Dataset From Config..........")
        self._reset_dataset_batch_size()
        train_dataset = self.create_dataset(is_train=True, default_args=default_args)
        return train_dataset

    def create_eval_dataset(self, default_args: dict = None):
        """Create the eval dataset for evaluate."""
        logger.info(".........Build Dataset From Config..........")
        self._reset_dataset_batch_size()
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        pipeline_stages = ms.get_auto_parallel_context("pipeline_stages")
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"] and pipeline_stages > 1:
            self.config.eval_dataset.batch_size = \
                self.config.eval_dataset.batch_size // self.config.parallel_config.micro_batch_num
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"]:
            self.config.eval_dataset.batch_size = \
                self.config.eval_dataset.batch_size // self.config.micro_batch_interleave_num
        # reduce batch size for that gradient_accumulation_steps will not take effect on eval
        if self.config.runner_config.gradient_accumulation_steps > 1:
            self.config.eval_dataset.batch_size = \
                self.config.eval_dataset.batch_size // self.config.runner_config.gradient_accumulation_steps
        if self.config.eval_dataset.batch_size < 1:
            logger.warning("eval_dataset batch_size is less than 1 after bs calculate, reset batch_size to 1, "
                           "please check your configs about batch_size, micro_batch_num micro_batch_interleave_num "
                           "and gradient_accumulation_steps.")
            self.config.eval_dataset.batch_size = 1
        logger.info("For evaluate phase, batch size for eval dataset is %s, different from training, "
                    "not multiplied by micro_batch_num, micro_batch_interleave_num and gradient_accumulation_steps",
                    self.config.eval_dataset.batch_size)
        eval_dataset = self.create_dataset(is_train=False, default_args=default_args)
        return eval_dataset

    def create_network(self, default_args: dict = None):
        """Create the network for task trainer."""
        logger.info(".........Build Network From Config..........")
        if self.config.get("pretrained_model_dir", None):
            self.config.model.pretrained_model_dir = self.config.pretrained_model_dir
        if self.config.get("generation_config", None):
            self.config.model.generation_config = self.config.generation_config
        network = build_network(self.config.model, default_args=default_args)
        self.real_model = network
        if hasattr(network, "check_pipeline_stage") and callable(network.check_pipeline_stage):
            network.check_pipeline_stage()
        return network

    def create_network_without_param_init(self, default_args: dict = None):
        """Create the network for task trainer without initialize parameters."""
        with no_init_parameters():
            network = self.create_network(default_args=default_args)
        logger.info("Parameters are not initialized during model initialization.")
        self.network_delay_inited = True
        return network

    def create_image_processor(self, default_args: dict = None):
        """Create the image processor for predict."""
        logger.info(".........Build Image Processor From Config..........")
        self.image_processor = build_processor(
            self.config.processor.image_processor, default_args=default_args)
        return self.image_processor

    def create_callbacks(self, default_args: dict = None):
        """Create the callback list for training."""
        logger.info(".........Build Callbacks for Train From Config..........")
        self.callbacks = []
        if self.config.profile:
            self.callbacks.append(self.config.profile_cb)
        self.callbacks.extend(build_callback(self.config.callbacks, default_args=default_args))
        return self.callbacks

    def create_eval_callbacks(self, default_args: dict = None):
        """Create the eval callback list for training."""
        logger.info(".........Build Callbacks for Evaluate From Config..........")
        self.eval_callbacks = []
        extend_eval_callbacks = build_callback(self.config.eval_callbacks, default_args=default_args)
        if extend_eval_callbacks:
            self.eval_callbacks.extend(extend_eval_callbacks)
        return self.eval_callbacks

    def count_parameters(self):
        """Count network parameters number."""
        if self.network is not None:
            logger.info("Network Parameters: %s M.", str(count_params(self.network)))
        else:
            logger.warning("Network is None, parameters incalculable.")

    def set_seed(self, seed: int = None):
        """Set seed for training."""
        if seed is None:
            if self.config.seed is None:
                raise ValueError("seed is not set in config, it is None.")
            set_seed(self.config.seed)
        else:
            set_seed(seed)

    def set_train_dataset(self, dataset):
        """Set the attribute of train dataset."""
        if dataset is None:
            raise ValueError("Train dataset is None")
        self.train_dataset = dataset

    def set_eval_dataset(self, dataset):
        """Set the attribute of eval dataset ."""
        if dataset is None:
            raise ValueError("Eval dataset is None")
        self.eval_dataset = dataset

    def set_network(self, network, is_train: bool = True):
        """Set the attribute of network."""
        if network is None:
            raise ValueError("network is None")
        if isinstance(network, (Cell, PreTrainedModel)):
            network.set_train(is_train)
        self.network = network

    def get_train_data_size(self):
        """Get train dataset size."""
        if self.train_dataset is None:
            raise NotImplementedError("train dataset is None")
        return self.train_dataset.get_dataset_size()

    def get_eval_data_size(self):
        """Get eval dataset size."""
        if self.eval_dataset is None:
            raise NotImplementedError("eval dataset is None")
        return self.eval_dataset.get_dataset_size()

    @staticmethod
    def get_pipeline_stages():
        """Get pipeline stages for task trainer."""
        pipeline_stages = ms.get_auto_parallel_context("pipeline_stages")
        return pipeline_stages

    def learning_rate_scale(self, base_learning_rate: float = 0., scale_factor: Optional[Union[float, int]] = 256.):
        """Scale learning rate for training."""
        if not isinstance(base_learning_rate, float):
            raise ValueError(f"learning rate must be float type, but get {type(base_learning_rate)}")
        if not isinstance(scale_factor, (float, int)):
            raise ValueError(f"scale_factor must be float or int type, but get {type(scale_factor)}")

        device_num = get_real_group_size()
        per_device_batch_size = self.config.train_dataset.batch_size
        learning_rate = (base_learning_rate * device_num * per_device_batch_size) / scale_factor
        return learning_rate

    def _process_megatron_dataset(self, dataset, config):
        """Dataset processing for Megatron Dataset."""
        if mf_get_context("dataset_broadcast_opt_level") < 3:
            raise ValueError("If using `BlendedMegatronDatasetDataLoader`, please set "
                             "`dataset_broadcast_opt_level: 3` in the `parallel_speed_up.json` file.")

        dataset_info = config.train_dataset.data_loader
        # reset dataset size to remove redundant data
        dataset = dataset.take(int(dataset_info.sizes[0]) // self.global_batch_size)
        logger.info(f"Use BlendedMegatronDatasetDataLoader, reset dataset size to {dataset.get_dataset_size()}.")

        # Sync assign eod compression arguments
        if self.config.train_dataset.data_loader.config.create_compressed_eod_mask:
            self.config.model.model_config.use_eod_attn_mask_compression = True
        return dataset, config

    @staticmethod
    def _check_sink_mode_with_ds_broadcast(config):
        """Check sink_mode with dataset_broadcast_opt_level."""
        ds_broadcast_level = mf_get_context("dataset_broadcast_opt_level")
        if ds_broadcast_level > 0 and not config.runner_config.sink_mode:
            raise ValueError(
                f"Current `dataset_broadcast_opt_level` is {ds_broadcast_level}, "
                f"if set larger than 0, then sink_mode should be set to be 'True'")

    @staticmethod
    def _check_input_sliced_sig(config, usage_info='special'):
        """Check input_sliced_sig in model config."""
        input_sliced_sig = config.model.model_config.get("input_sliced_sig")
        if not input_sliced_sig:
            raise ValueError(
                f"In this {usage_info} configuration, input_sliced_sig in model_config should be set 'True'")

    @staticmethod
    def _get_columns_with_strategy(config, dataset_config, dataloader_type):
        """
        Determine dataset sharding strategy and column names based on configuration.
        """
        dp = config.parallel_config.data_parallel

        # Default sharding strategy and column names
        dataset_strategy = 'full_batch'
        column_names = None

        create_compressed_eod_mask = dataset_config.get('create_compressed_eod_mask', False)
        create_attention_mask = dataset_config.get('create_attention_mask', False)

        if create_compressed_eod_mask:
            dataset_strategy = [[dp, 1], [dp, 1], [dp, 1], [dp, 1], [dp, 1]]
            column_names = ['input_ids', 'labels', 'loss_mask', 'position_ids', 'actual_seq_len']
            config.model.model_config.use_eod_attn_mask_compression = True

        elif create_attention_mask:
            dataset_strategy = [[dp, 1], [dp, 1], [dp, 1], [dp, 1], [dp, 1, 1, 1]]
            column_names = ['input_ids', 'labels', 'loss_mask', 'position_ids', 'attention_mask']

        elif dataloader_type == 'BlendedMegatronDatasetDataLoader':
            dataset_strategy = [[dp, 1], [dp, 1], [dp, 1], [dp, 1]]
            column_names = ['input_ids', 'labels', 'loss_mask', 'position_ids']

        # Allow overriding defaults from config
        dataset_strategy = config.parallel.get('dataset_strategy', dataset_strategy)
        column_names = config.train_dataset.get('input_columns', column_names)
        construct_args_key = config.train_dataset.get('construct_args_key', column_names)

        return config, dataset_strategy, column_names, construct_args_key

    @staticmethod
    def _set_full_batch_with_strategy(config, dataset_strategy):
        """
        Convert dataset strategy format and set auto parallel context.
        """
        # If full_batch is explicitly set in parallel config, log a warning
        full_batch = config.parallel.get('full_batch')
        if full_batch is not None:
            logger.warning("`full_batch` will be deprecated in future interfaces, use `dataset_strategy` instead.")

        # Convert strategy to correct format and set full_batch flag
        if dataset_strategy != 'full_batch':
            full_batch = False
            if isinstance(dataset_strategy, list):
                dataset_strategy = tuple(tuple(ds_item) for ds_item in dataset_strategy)
            elif not isinstance(dataset_strategy, tuple):
                raise ValueError('`dataset_strategy` should be list or tuple.')

        # Apply parallel context settings
        ms.set_auto_parallel_context(**{
            'full_batch': full_batch,
            'dataset_strategy': dataset_strategy
        })

        return full_batch, dataset_strategy

    @staticmethod
    def _set_dataset_broadcast_opt_level(config):
        """Set the dataset broadcast optimization level."""
        ds_broadcast_level = mf_get_context("dataset_broadcast_opt_level")
        if ds_broadcast_level == 0:
            logger.info("Set ds_broadcast_level=3 by default while enabling broadcast dataset.")
            ds_broadcast_level = 3

        # Override with the value from config if provided, otherwise keep the existing one
        ds_broadcast_level = config.context.get('dataset_broadcast_opt_level', ds_broadcast_level)
        mf_set_context(dataset_broadcast_opt_level=ds_broadcast_level)

    def _train_dataset_postprocess(self, dataset, config):
        """
        Postprocess the training dataset after construction.
        Mainly used to adjust dataset size for special dataloaders.
        """
        dataloader_config = config.train_dataset.get('data_loader', {})
        dataloader_type = dataloader_config.get('type')

        # Special handling for BlendedMegatronDatasetDataLoader
        if dataloader_type == "BlendedMegatronDatasetDataLoader":
            dataset_info = config.train_dataset.data_loader
            # Resize dataset to ensure it's divisible by global batch size (remove redundant data)
            dataset = dataset.take(int(dataset_info.sizes[0]) // self.global_batch_size)
            logger.info(
                f"Use BlendedMegatronDatasetDataLoader, reset dataset size to {dataset.get_dataset_size()}."
            )

        return dataset

    def _train_dataset_preprocess(self, config):
        """
        Preprocess training dataset configuration before building the dataloader.
        This includes validating broadcast options, setting dataset strategies,
        and assigning dataset column names depending on the dataloader type.
        """
        # Check dataset sink mode with dataset broadcast optimization level
        self._check_sink_mode_with_ds_broadcast(config)

        dataloader_config = config.train_dataset.get('data_loader', {})
        dataloader_type = dataloader_config.get('type')

        # Case 1: BlendedMegatronDatasetDataLoader
        if dataloader_type == 'BlendedMegatronDatasetDataLoader':
            # Validate that input slicing is enabled
            self._check_input_sliced_sig(config, dataloader_type)
            self._set_dataset_broadcast_opt_level(config)
            dataloader_config = dataloader_config.get('config')

        # Case 2: HFDataLoader
        if dataloader_type == 'HFDataLoader':
            # Collect handler types if present
            handler = []
            if dataloader_config.handler is not None:
                handler = [sub_handler.get('type') for sub_handler in dataloader_config.handler]

            # Packing or attention mask requires extended dataset strategy
            if 'PackingHandler' in handler:
                self._check_input_sliced_sig(config, f"{dataloader_type} with packing")
                config.train_dataset.data_loader.create_attention_mask = True

            if dataloader_config.streaming and isinstance(config.callbacks, list):
                for callback in config.callbacks:
                    if callback.get('type') == 'CheckpointMonitor':
                        dataloader_config.save_step = int(config.checkpoint.save_interleaved_steps)

        config, dataset_strategy, column_names, construct_args_key = self._get_columns_with_strategy(
            config, dataloader_config, dataloader_type)
        logger.info(f"Got dataset config: "
                    f"dataset_strategy: {dataset_strategy}, "
                    f"column_names: {column_names}, "
                    f"construct_args_key: {construct_args_key}")

        full_batch, dataset_strategy = self._set_full_batch_with_strategy(config, dataset_strategy)

        # Update config with resolved column names and construct keys
        config.train_dataset.input_columns = column_names
        config.train_dataset.construct_args_key = construct_args_key
        config.train_dataset_task.dataset_config = config.train_dataset

        config.parallel.full_batch = full_batch
        config.parallel.dataset_strategy = dataset_strategy

        return config

    @staticmethod
    def resume_ckpt_path_with_strategy(config):
        """Get resume checkpoint path with strategy.

        This method finds the appropriate checkpoint path for the current rank when resuming training
        with a distributed strategy.

        Args:
            config (MindFormerConfig): The configuration containing checkpoint and strategy settings.

        Returns:
            str: Path to the checkpoint file for the current rank after applying strategy.
                 Returns None if no valid checkpoint is found.

        Raises:
            ValueError: If strategy file for current rank does not exist.
        """

        cur_rank = get_rank()
        src_strategy_files = sorted(list(os.listdir(config.src_strategy_path_or_dir)))
        if len(src_strategy_files) - 1 < cur_rank:
            raise ValueError(f" rank {cur_rank} src_strategy is not exist")
        src_strategy_file = os.path.join(config.src_strategy_path_or_dir, src_strategy_files[cur_rank])
        if os.path.isfile(config.checkpoint.load_path):
            return get_ckpt_path_with_strategy(config.checkpoint.load_path, src_strategy_file)

        if os.path.isdir(config.checkpoint.load_path):
            device_nums = get_group_size()
            max_ckpt_path = ""
            max_time = 0
            for i in range(device_nums):
                cur_rank_ckpt_dir = os.path.join(config.checkpoint.load_path, f"rank_{i}")
                last_ckpt = get_last_checkpoint(cur_rank_ckpt_dir, config.load_ckpt_format)
                if last_ckpt is None:
                    continue
                cur_time = os.path.getmtime(last_ckpt)
                if cur_time > max_time:
                    max_time = cur_time
                    max_ckpt_path = last_ckpt
            pattern = fr'rank_\d+(?:_(\d+))?-(\d+)_(\d+)\.{config.load_ckpt_format}$'
            match = re.search(pattern, max_ckpt_path)
            if match:
                repeat_num = int(match.group(1)) if match.group(1) else 0
                if repeat_num == 0:
                    return get_ckpt_path_with_strategy(max_ckpt_path, src_strategy_file)
                for i in range(repeat_num, 0, -1):
                    # Replace the current repeat number in max_ckpt_path with i
                    cur_ckpt_path = re.sub(r'rank_\d+_\d+', f'rank_{cur_rank}_{i}', max_ckpt_path)
                    load_ckpt_new = get_ckpt_path_with_strategy(cur_ckpt_path, src_strategy_file)
                    if load_ckpt_new:
                        return load_ckpt_new
                # Try without repeat number (i=0 case)
                cur_ckpt_path = re.sub(r'rank_\d+_\d+', f'rank_{cur_rank}', max_ckpt_path)
                return get_ckpt_path_with_strategy(cur_ckpt_path, src_strategy_file)

        return None

    # pylint: disable=C0330
    def training_process(
            self,
            config: Optional[Union[dict, MindFormerConfig, TrainingArguments]] = None,
            network: Optional[Union[Cell, PreTrainedModel]] = None,
            dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
            optimizer: Optional[Optimizer] = None,
            callbacks: Optional[Union[Callback, List[Callback]]] = None,
            compute_metrics: Optional[Union[dict, set]] = None,
            **kwargs):
        """Report that legacy training has been retired."""
        raise RuntimeError("Use mindformers.pynative.trainer.Trainer for training.")

    def evaluate_process(
            self,
            config: Optional[Union[dict, MindFormerConfig, TrainingArguments]] = None,
            network: Optional[Union[Cell, PreTrainedModel]] = None,
            dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
            callbacks: Optional[Union[Callback, List[Callback]]] = None,
            compute_metrics: Optional[Union[dict, set]] = None,
            **kwargs):
        """Report that legacy evaluation has been retired."""
        raise RuntimeError("Legacy Trainer evaluation has been retired.")

    def predict_process(self,
                        config: Optional[Union[dict, MindFormerConfig, TrainingArguments]] = None,
                        input_data: Optional[Union[GeneratorDataset, Tensor, np.ndarray, Image, str, list]] = None,
                        task: str = None,
                        network: Optional[Union[Cell, PreTrainedModel]] = None,
                        tokenizer: Optional[PreTrainedTokenizerBase] = None,
                        image_processor: Optional[BaseImageProcessor] = None,
                        audio_processor: Optional[BaseImageProcessor] = None, **kwargs):
        """Report that legacy prediction has been retired."""
        raise RuntimeError("Use mindformers.pynative.trainer.Trainer for inference.")

    @staticmethod
    def sort_callbacks(default_callbacks):
        """Sort by execution order."""
        callbacks = [None] * len(CALLBACK_HAS_SORT)

        for callback in default_callbacks:
            flag = False
            for index, callback_ in enumerate(CALLBACK_HAS_SORT):
                if isinstance(callback, callback_):
                    callbacks[index] = callback
                    flag = True
            if not flag:
                callbacks.append(callback)

        return list(filter(lambda x: x is not None, callbacks))
