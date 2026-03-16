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
"""Config module for Pynative Trainer."""

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    get_origin,
    get_args,
    Type,
    ClassVar,
)
from pathlib import Path

import yaml
from mindformers.tools.check_rules import check_yaml_depth_before_loading
from .utils import check_type


@dataclass
class BaseConfig:
    """Base configuration class with load, merge, and validation capabilities."""

    allow_extra: ClassVar[bool] = False

    @classmethod
    def load_from_yaml(cls, path: str) -> "BaseConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            An instance of the configuration class loaded from the YAML file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the configuration data is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # Load YAML content and parse into dictionary
        with open(path, "r", encoding="utf-8") as f:
            check_yaml_depth_before_loading(f)
            f.seek(0)
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """
        Create a configuration instance from a dictionary with validation and merging.

        This method processes the input dictionary, converts values to appropriate types,
        handles nested configurations, and validates the final configuration.

        Args:
            data: Dictionary containing configuration key-value pairs.

        Returns:
            An instance of the configuration class initialized from the dictionary.

        Raises:
            TypeError: If the input data is not a dictionary.
            ValueError: If there are unknown configuration keys (when allow_extra=False)
                       or if the configuration values are invalid.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for {cls.__name__}, got {type(data)}")

        # Map class field names to their types for validation
        class_fields = {f.name: f.type for f in fields(cls)}
        init_args = {}
        extra_args = {}

        # Separate known fields from extra fields
        for k, v in data.items():
            if k in class_fields:
                # Convert value to expected type (handles nested configs, lists, etc.)
                init_args[k] = cls._convert_value(v, class_fields[k])
            else:
                extra_args[k] = v

        # Create instance with known fields
        try:
            instance = cls(**init_args)
        except TypeError as e:
            raise ValueError(f"Configuration error in {cls.__name__}: {e}") from e

        # Handle extra fields if allowed
        if extra_args:
            if not (cls.allow_extra or cls is BaseConfig):
                raise ValueError(
                    f"Unknown configuration keys for {cls.__name__}: {list(extra_args.keys())}"
                )

            # Recursively convert nested dictionaries to BaseConfig instances
            for k, v in extra_args.items():
                if isinstance(v, dict):
                    v = BaseConfig.from_dict(v)
                setattr(instance, k, v)

        # Validate all configuration values
        instance.validate()
        return instance

    @staticmethod
    def _convert_value(value: Any, field_type: Type) -> Any:
        """
        Convert a value to the expected type recursively.

        This method handles type conversion for:
        - Optional/Union types (extracts the non-None type)
        - Nested dataclass instances (recursively converts dictionaries)
        - Lists of dataclass instances (converts each item)

        Args:
            value: The value to convert.
            field_type: The expected type annotation for the field.

        Returns:
            The converted value matching the expected type.
        """
        origin = get_origin(field_type)

        # Handle Optional/Union types (e.g., Optional[str] or Union[int, str])
        # Extract the actual type by removing None from the union
        if origin is Union:
            args = get_args(field_type)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                field_type = non_none_args[0]
                origin = get_origin(field_type)

        # Recursive instantiation for nested dataclasses
        # If the field type is a dataclass and value is a dict, convert it
        if is_dataclass(field_type) and isinstance(value, dict):
            return field_type.from_dict(value)

        # Handle List[Dataclass] - convert each dict item in the list to dataclass instance
        if origin is list and isinstance(value, list):
            args = get_args(field_type)
            if args:
                arg_type = args[0]
                if is_dataclass(arg_type):
                    return [
                        arg_type.from_dict(item) if isinstance(item, dict) else item
                        for item in value
                    ]

        return value

    def _iter_params(self):
        """
        Iterate over all configuration fields (including extra ones).
        """
        results = {}
        class_fields = (f.name for f in fields(self))

        # Record defined dataclass fields
        for f in class_fields:
            results[f] = getattr(self, f)

        # Record extra fields
        for k, v in self.__dict__.items():
            # Ignore private fields (fields with '_' prefix) and class fields
            if k not in class_fields and not k.startswith("_"):
                results[k] = v
        return results

    def __repr__(self):
        """
        Convert the configuration object into a string representation.

        Returns:
            A string representation of the configuration object showing all fields.
        """
        fields_str = [f"{k}={repr(v)}" for k, v in self._iter_params().items()]
        return f"{self.__class__.__name__}({', '.join(fields_str)})"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration object to a dictionary representation.
        """
        return {k: self._to_dict_value(v) for k, v in self._iter_params().items()}

    @staticmethod
    def _to_dict_value(value: Any) -> Any:
        """
        Recursively convert a value to a dictionary-compatible representation.
        """
        if isinstance(value, BaseConfig):
            return value.to_dict()
        if is_dataclass(value) and not isinstance(value, type):
            return dataclasses.asdict(value)
        if isinstance(value, list):
            return [BaseConfig._to_dict_value(v) for v in value]
        if isinstance(value, dict):
            return {k: BaseConfig._to_dict_value(v) for k, v in value.items()}
        return value

    def __getattr__(self, name):
        """
        Support dictionary-like 'get' method for accessing configuration values.
        This allows using config.get('key', default) syntax similar to dictionaries.
        """
        if name == "get":
            # Return a get function that mimics dict.get() behavior
            def get_fn(key: str, default: Any = None) -> Any:
                return getattr(self, key, default)

            return get_fn
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __getitem__(self, key: str) -> Any:
        """
        Support dictionary-like access using square brackets.
        """
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(f"'{type(self).__name__}' object has no key '{key}'") from e

    def validate(self):
        """
        Validate all configuration field values against their type annotations.

        This method checks that each field value matches its declared type.
        Raises an exception if any validation fails.

        Raises:
            TypeError: If any field value does not match its declared type.
        """
        cls_name = type(self).__name__
        for f in fields(self):
            value = getattr(self, f.name)
            check_type(value, f.type, f"{cls_name}.{f.name}")


@dataclass
class CheckpointConfig(BaseConfig):
    """
    Configuration for model checkpoint saving and loading.
    """

    enable_save: bool = True
    """Whether to enable checkpoint saving"""

    save_path: str = ""
    """Directory to save checkpoints"""

    save_max: int = 5
    """Maximum number of checkpoints to retain"""

    save_interleaved_steps: int = 1000
    """Number of training steps between checkpoint saves"""

    no_save_optim: bool = False
    """Whether to skip saving optimizer states"""

    async_save: bool = False
    """Enable asynchronous checkpoint saving"""

    prefix: Optional[str] = "checkpoint"
    """Filename prefix for saved checkpoints"""

    remove_redundancy: bool = False
    """Remove redundant data when saving checkpoints"""

    load_path: str = ""
    """Directory to load checkpoints from"""

    load_balanced: bool = False
    """Enable balanced loading across ranks/devices"""

    no_load_optim: bool = False
    """Whether to skip loading optimizer states"""

    load_worker_number: int = 1
    """Number of worker threads used for checkpoint loading"""


@dataclass
class TrainingConfig(BaseConfig):
    """
    Training execution configuration.
    """

    steps: int = 1000
    """Total number of training steps"""

    local_batch_size: int = 1
    """Per-rank batch size"""

    global_batch_size: int = 1
    """Total number of samples processed per step"""

    max_norm: float = 1.0
    """Enable gradient clipping if value > 0"""

    seed: int = 42
    """Random seed for training"""

    deterministic: bool = False
    """Enable deterministic training behavior"""

    def post_init(self):
        """Post-initialization validation."""
        if self.global_batch_size <= 0:
            raise ValueError("training.global_batch_size in config must be positive")
        if self.local_batch_size <= 0:
            raise ValueError("training.local_batch_size in config must be positive")


@dataclass
class ParallelismConfig(BaseConfig):
    """
    Parallelism configuration for distributed training.
    """

    tensor_parallel: int = 1
    """Tensor parallelism degree"""

    context_parallel: int = 1
    """Context parallelism degree"""

    context_parallel_method: str = "colossal"
    """Implementation method for context parallelism"""

    pipeline_parallel: int = 1
    """Pipeline parallelism degree"""

    pipeline_parallel_layers_per_stage: Optional[List[List[int]]] = None
    """Layers assigned to each pipeline stage"""

    pipeline_parallel_schedule: str = "1f1b"
    """Pipeline execution schedule"""

    pipeline_parallel_microbatch_size: int = 1
    """Number of micro-batches per pipeline step"""

    pipeline_parallel_interleave_num: int = 1
    """Number of interleaved model chunks"""

    hsdp_shard_size: int = 1
    """HSDP sharding group size"""

    hsdp_optimizer_level: str = "Level1"
    """Optimizer state sharding level for HSDP"""

    hsdp_threshold: int = 64
    """Parameter size threshold for enabling HSDP sharding"""

    sequence_parallel: bool = False
    """Enable sequence parallelism"""


@dataclass
class OptimizerConfig(BaseConfig):
    """
    Optimizer configuration.
    """

    type: str = "AdamW"
    """Optimizer type"""

    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    """Betas for optimizer"""

    eps: float = 1.0e-8
    """Epsilon for optimizer"""

    weight_decay: float = 0.01
    """Weight decay"""


@dataclass
class LrSchedulerConfig(BaseConfig):
    """
    Learning rate scheduler configuration.
    """

    allow_extra = True

    type: str = None
    """Scheduler type"""

    learning_rate: float = 1e-5
    """Learning rate"""


@dataclass
class DataloaderConfig(BaseConfig):
    """
    Dataloader configuration (Extensible).
    """

    allow_extra = True

    type: str = None
    """Dataloader type"""

    shuffle: bool = False
    """Whether to shuffle the dataset"""

    column_names: List[str] = field(default_factory=lambda: ["input_ids", "labels"])
    """Input feature column names"""

    python_multiprocessing: bool = False
    """Whether to use python multiprocessing"""


@dataclass
class TrainDatasetConfig(BaseConfig):
    """
    Train dataset configuration.
    """

    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    """Dataloader configuration"""

    drop_remainder: bool = True
    """Whether to drop the remainder of the dataset"""

    num_parallel_workers: int = 8
    """Number of parallel data loading workers"""

    prefetch_size: int = 1
    """Number of prefetched batches"""

    numa_enable: bool = False
    """Enable NUMA-aware data loading"""


@dataclass
class ModelConfig(BaseConfig):
    """
    Model configuration.
    """

    allow_extra = True

    model_type: str = None
    """Model type"""

    architectures: str = None
    """Model architecture class name"""


@dataclass
class HealthCheckpointConfig(BaseConfig):
    """
    Health checkpoint configuration for monitoring training stability.
    """

    embedding_local_norm_threshold: float = 100.0
    """Threshold for embedding parameter local norm"""

    global_norm_skip_threshold: float = 100.0
    """Global norm threshold for skipping updates"""

    global_norm_skip_time: int = 10
    """Number of consecutive skips allowed"""


@dataclass
class TrainStateConfig(BaseConfig):
    """
    Training state monitoring configuration.
    """

    local_norm: bool = False
    """Monitor local gradient/parameter norm"""

    local_loss: bool = False
    """Monitor local training loss"""

    device_norm: bool = False
    """Monitor device-level norm statistics"""

    device_loss: bool = False
    """Monitor device-level loss statistics"""


@dataclass
class TensorboardConfig(BaseConfig):
    """
    TensorBoard logging configuration.
    """

    tensorboard_dir: str = ""
    """Directory to store TensorBoard logs"""

    tensorboard_log_interval: int = 1
    """Logging interval"""

    tensorboard_queue_size: int = 1000
    """Maximum size of the TensorBoard event queue"""


@dataclass
class MonitorConfig(BaseConfig):
    """
    Monitor configuration.
    """

    health_checkpoint: HealthCheckpointConfig = field(
        default_factory=HealthCheckpointConfig
    )
    train_state: TrainStateConfig = field(default_factory=TrainStateConfig)
    tensorboard: TensorboardConfig = field(default_factory=TensorboardConfig)


@dataclass
class CallbackConfig(BaseConfig):
    """
    Callback configuration (Extensible).
    """

    allow_extra = True
    type: str = None
    """Callback type"""


@dataclass
class ContextConfig(BaseConfig):
    """
    MindSpore context configuration.
    """

    mode: int = 1
    """Pynative Mode"""

    max_device_memory: str = "59GB"
    """Maximum device memory usage"""

    device_target: str = "Ascend"
    """Target device type"""


@dataclass
class ProfilerConfig(BaseConfig):
    """
    Profiler configuration.
    """

    enable_profiler: bool = False
    """Enable profiler"""

    start_step: int = 10
    """Profiling start step"""

    stop_step: int = 20
    """Profiling stop step"""

    output_path: str = ""
    """Profiler output directory"""

    profiler_level: str = "Level0"
    """Profiling detail level"""

    profile_memory: bool = False
    """Enable memory profiling"""


@dataclass
class RecomputeConfig(BaseConfig):
    """
    Recompute configuration for memory optimization.
    """

    full_recompute: bool = False
    """Enable fully recomputation"""

    select_recompute: bool = False
    """Enable selective recomputation"""


@dataclass
class TrainConfig(BaseConfig):
    """
    Top-level Pynative training configuration.

    This is the main configuration class that aggregates all sub-configurations
    for a complete training setup. It includes configurations for:
    - Checkpoint management
    - Training parameters
    - Parallelism strategies
    - Optimizer and learning rate scheduler
    - Dataset and dataloader
    - Model architecture
    - Monitoring and logging
    - MindSpore context
    - Profiling
    - Gradient recomputation
    - Training callbacks

    This configuration allows extra fields to support extensibility.
    """

    allow_extra = True
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LrSchedulerConfig = field(default_factory=LrSchedulerConfig)
    train_dataset: TrainDatasetConfig = field(default_factory=TrainDatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    recompute: RecomputeConfig = field(default_factory=RecomputeConfig)
    callbacks: List[CallbackConfig] = field(default_factory=list)
