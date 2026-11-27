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
"""Causal Image Modeling Trainer."""
import os
from typing import Optional, List, Union
from pprint import pprint

from mindspore import Model
from mindspore.train import Callback
from mindspore.nn import TrainOneStepCell, Optimizer, Cell
from mindspore.dataset import GeneratorDataset

from mindformers.dataset import BaseDataset
from mindformers.models import PreTrainedModel, PreTrainedTokenizerBase
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType, MindFormerConfig
from mindformers.tools.check_rules import check_rules
from mindformers.models.auto import AutoModel
from mindformers.mindformer_book import MindFormerBook
from mindformers.utils.load_checkpoint_utils import get_load_path_after_hf_convert

from ..training_args import TrainingArguments
from ..base_trainer import BaseTrainer
from ..utils import transform_and_load_checkpoint, get_real_rank, mask_config_tokens

SUPPORT_MODEL_NAMES = MindFormerBook().get_model_name_support_list()


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class CausalLanguageModelingTrainer(BaseTrainer):
    """
    CausalLanguageModelingTrainer Task For Trainer.

    Args:
        model_name (str): The model name of Task-Trainer. Default: None
    """

    def __init__(self, model_name: str = None):
        super(CausalLanguageModelingTrainer, self).__init__("text_generation", model_name)

    def train(self,
              config: Optional[Union[dict, MindFormerConfig, TrainingArguments]] = None,
              network: Optional[Union[Cell, PreTrainedModel]] = None,
              dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        """
        Train task for CausalLanguageModeling Trainer.
        This function is used to train or fine-tune the network.
        The trainer interface is used to quickly start training for general task.
        It also allows users to customize the network, optimizer, dataset, wrapper, callback.

        Args:
            config (Optional[Union[dict, MindFormerConfig, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments class.
                Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]):
                The network for trainer.It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            dataset (Optional[Union[BaseDataset, GeneratorDataset]]):
                The training dataset.It support real dataset path or BaseDateset class or MindSpore
                Dataset class. Default: None.
            wrapper (Optional[TrainOneStepCell]):
                Wraps the `network` with the `optimizer`.It support TrainOneStepCell class of MindSpore.
                Default: None.
            optimizer (Optional[Optimizer]):
                he training network's optimizer. It support Optimizer class of MindSpore. Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]):
                The training callback function. It support CallBack or CallBack List of MindSpore. Default: None.
        """
        self.training_process(
            config=config,
            network=network,
            callbacks=callbacks,
            dataset=dataset,
            wrapper=wrapper,
            optimizer=optimizer,
            **kwargs)

    def evaluate(self,
                 config: Optional[Union[dict, MindFormerConfig, TrainingArguments]] = None,
                 network: Optional[Union[Cell, PreTrainedModel]] = None,
                 dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 compute_metrics: Optional[Union[dict, set]] = None,
                 **kwargs):
        """Run the legacy evaluation entry point."""
        return self.evaluate_process(config=config, network=network, dataset=dataset,
                                     callbacks=callbacks, compute_metrics=compute_metrics, **kwargs)

    def predict(self,
                config: Optional[Union[dict, MindFormerConfig, TrainingArguments]] = None,
                input_data: Optional[Union[str, list, GeneratorDataset]] = None,
                network: Optional[Union[Cell, PreTrainedModel]] = None,
                tokenizer: Optional[PreTrainedTokenizerBase] = None,
                **kwargs):
        """
        Executes the predict of the trainer.

        Args:
            config (Optional[Union[dict, MindFormerConfig, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments class.
                Default: None.
            input_data (Optional[Union[Tensor, str, list]]):
                The predict data. It supports 1) a text string to be translated, 1) a file name where each
                line is a text to be translated  and 3) a generator dataset. Default: None.
            network (Optional[Union[Cell, PreTrainedModel]]):
                The network for trainer. It supports model name or PreTrainedModel or MindSpore Cell class.
                Default: None.
            tokenizer (Optional[PreTrainedTokenizerBase]):
                The tokenizer for tokenizing the input text. Default: None.

        Returns:
            List, a list of prediction.
        """
        if input_data is None:
            input_data = config.input_data

        if not isinstance(input_data, (str, list, GeneratorDataset)):
            raise ValueError("Input data's type must be one of "
                             f"[str, list, GeneratorDataset], but got type {type(input_data)}")

        if isinstance(input_data, str) and os.path.isfile(input_data):
            with open(input_data, 'r') as fp:
                input_data_list = []
                for line in fp:
                    line = line.strip('\n')
                    line = line.replace(r"\n", "\n")
                    input_data_list.append(line)
            input_data = input_data_list

        return self.predict_process(config=config,
                                    input_data=input_data,
                                    task='text_generation',
                                    network=network,
                                    tokenizer=tokenizer,
                                    **kwargs)
