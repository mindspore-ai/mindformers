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

"""mindformers init"""

__version__ = "1.9.0"

from mindformers import (
    core,
    dataset,
    models,
    modules,
    pet,
    tools,
    trainer
)
from mindformers.trainer import (
    BaseTrainer,
    CausalLanguageModelingTrainer,
    GeneralTaskTrainer,
    MultiModalToTextGenerationTrainer,
    Trainer,
    TrainingArguments,
)
from mindformers.core import (
    CheckpointMonitor,
    EvalCallBack,
    MFLossMonitor,
    ProfileMonitor,
    SummaryMonitor,
    TrainingStateMonitor,
    build_context,
    get_context,
    init_context,
    set_context
)
from mindformers.dataset import (
    AlpacaInstructDataHandler,
    BCHW2BHWC,
    BaseDataset,
    BatchCenterCrop,
    BatchNormalize,
    BatchPILize,
    BatchResize,
    BatchToTensor,
    CausalLanguageModelDataset,
    CodeAlpacaInstructDataHandler,
    GeneralDataset,
    RandomCropDecodeResize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    build_data_handler,
    check_dataset_config,
    check_dataset_iterable
)
from mindformers.models import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskGeneration,
    AutoModelForPreTraining,
    AutoModelWithLMHead,
    AutoProcessor,
    AutoTokenizer,
    BaseAudioProcessor,
    BaseConfig,
    BaseImageProcessor,
    BaseModel,
    BaseProcessor,
    CONFIG_MAPPING,
    CONFIG_MAPPING_NAMES,
    IMAGE_PROCESSOR_MAPPING,
    ImageProcessingMixin,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    PretrainedConfig,
    TOKENIZER_MAPPING,
)
from mindformers.modules import (
    EmbeddingOpParallelConfig,
    OpParallelConfig,
    TransformerOpParallelConfig,
    TransformerRecomputeConfig
)
from mindformers.tools import (
    ActionDict,
    DictConfig,
    MindFormerConfig,
    MindFormerModuleType,
    MindFormerRegister,
    logger,
)
from mindformers import generation
from mindformers.generation import (
    BaseStreamer,
    FrequencyPenaltyLogitsProcessor,
    GenerationConfig,
    GenerationMixin,
    GreedySearchLogitsProcessor,
    LogitNormalization,
    LogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    PresencePenaltyLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SamplingLogitsProcessor,
    TemperatureLogitsWarper,
    TextIteratorStreamer,
    TextStreamer,
    TopKLogitsWarper,
    TopPLogitsWarper
)
from mindformers.pet import (
    AdaAdapter,
    AdaLoraAdapter,
    LoraAdapter,
    LoraConfig,
    LoraModel,
    PetAdapter,
    PetConfig,
    PrefixTuningAdapter,
    PrefixTuningConfig,
    Ptuning2Adapter,
    Ptuning2Config
)
from mindformers.run_check import run_check
from mindformers.mindformer_book import MindFormerBook

__all__ = ['run_check']
__all__.extend(name for name in core.__all__ if name in globals())
__all__.extend(dataset.__all__)
__all__.extend(generation.__all__)
__all__.extend(models.__all__)
__all__.extend(pet.__all__)
__all__.extend(tools.__all__)
__all__.extend(trainer.__all__)
