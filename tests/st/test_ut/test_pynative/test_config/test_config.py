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
"""Test Config module."""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union, Any, Dict
import pytest

from mindformers.pynative.config.config import (
    BaseConfig,
    TrainConfig,
    CheckpointConfig,
    TrainingConfig,
    ParallelismConfig,
    OptimizerConfig,
    CallbackConfig,
)
from mindformers.pynative.config.utils import check_type


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(CURRENT_DIR, "train_config.yaml")


@dataclass
class SimpleConfig(BaseConfig):
    """Simple config for testing."""
    a: int = 1
    b: str = "test"


@dataclass
class NestedConfig(BaseConfig):
    """Nested config for testing."""
    simple: SimpleConfig = field(default_factory=SimpleConfig)
    c: List[int] = field(default_factory=list)


@dataclass
class ExtraConfig(BaseConfig):
    """Config allowing extra fields."""
    allow_extra = True
    a: int = 1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestConfig:
    """Test cases for Config module."""

    def test_base_config_initialization(self):
        """Test BaseConfig initialization and attribute access."""
        config = SimpleConfig(a=2, b="changed")
        assert config.a == 2
        assert config.b == "changed"

        # Test __getitem__
        assert config['a'] == 2
        with pytest.raises(KeyError):
            _ = config['non_existent']

        # Test get method via __getattr__
        assert config.get('b') == "changed"
        assert config.get('z', 'default') == 'default'

        # Test invalid attribute access
        with pytest.raises(AttributeError):
            config.non_existent_method()

    def test_base_config_repr(self):
        """Test BaseConfig string representation."""
        # Test with BaseConfig directly to use its __repr__
        config = BaseConfig()
        repr_str = repr(config)
        assert "BaseConfig" in repr_str

        # Test with subclass that disables generated __repr__
        @dataclass(repr=False)
        class NoReprConfig(BaseConfig):
            a: int = 1

        config2 = NoReprConfig()
        repr_str2 = repr(config2)
        assert "NoReprConfig" in repr_str2
        assert "a=1" in repr_str2

        # SimpleConfig (dataclass) uses generated __repr__, so it doesn't cover BaseConfig.__repr__
        config3 = SimpleConfig(a=2)
        repr_str3 = repr(config3)
        assert "SimpleConfig" in repr_str3

    def test_base_config_to_dict(self):
        """Test converting config to dictionary."""
        config = NestedConfig(simple=SimpleConfig(a=5), c=[1, 2])
        d = config.to_dict()
        expected = {
            'simple': {'a': 5, 'b': 'test'},
            'c': [1, 2]
        }
        assert d == expected

    def test_base_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {'a': 10, 'b': 'val'}
        config = SimpleConfig.from_dict(data)
        assert config.a == 10
        assert config.b == 'val'

        # Test invalid input type
        with pytest.raises(TypeError):
            SimpleConfig.from_dict("not a dict")

    def test_base_config_validation(self):
        """Test type validation during creation."""
        # Test invalid type
        data_invalid = {'a': 'not_int', 'b': 'val'}
        # validate() raises TypeError directly
        with pytest.raises(TypeError) as cm:
            SimpleConfig.from_dict(data_invalid)
        assert "Expected int for 'SimpleConfig.a'" in str(cm.value)

    def test_from_dict_missing_required(self):
        """Test from_dict with missing required field."""

        @dataclass
        class RequiredConfig(BaseConfig):
            req: int

        with pytest.raises(ValueError) as cm:
            RequiredConfig.from_dict({})
        assert "Configuration error" in str(cm.value)

    def test_to_dict_complex(self):
        """Test to_dict with nested dataclasses (not BaseConfig) and dicts."""

        @dataclass
        class PlainData:
            x: int

        @dataclass
        class ComplexConfig(BaseConfig):
            plain: PlainData
            mapping: Dict[str, PlainData]

        config = ComplexConfig(
            plain=PlainData(x=1),
            mapping={'k': PlainData(x=2)}
        )
        d = config.to_dict()
        assert d['plain']['x'] == 1
        assert d['mapping']['k']['x'] == 2

    def test_extra_fields(self):
        """Test handling of extra fields."""
        # Test disallowed extra fields
        data_extra = {'a': 10, 'extra': 'ignored'}
        with pytest.raises(ValueError) as cm:
            SimpleConfig.from_dict(data_extra)
        assert "Unknown configuration keys" in str(cm.value)

        # Test allowed extra fields
        data_allowed = {'a': 2, 'extra_val': 'value', 'extra_dict': {'x': 1}}
        config = ExtraConfig.from_dict(data_allowed)
        assert config.a == 2
        assert config.extra_val == 'value'

        # Verify nested dict in extra fields becomes BaseConfig
        assert isinstance(config.extra_dict, BaseConfig)
        assert config.extra_dict.x == 1

    def test_nested_config_conversion(self):
        """Test recursive conversion of nested configs."""
        data = {
            'simple': {'a': 5, 'b': 'nested'},
            'c': [1, 2, 3]
        }
        config = NestedConfig.from_dict(data)
        assert isinstance(config.simple, SimpleConfig)
        assert config.simple.a == 5
        assert config.c == [1, 2, 3]

    def test_list_of_dataclasses(self):
        """Test list of dataclasses conversion."""
        @dataclass
        class ListConfig(BaseConfig):
            items: List[SimpleConfig] = field(default_factory=list)

        data = {
            'items': [
                {'a': 1, 'b': 'one'},
                {'a': 2, 'b': 'two'}
            ]
        }
        config = ListConfig.from_dict(data)
        assert len(config.items) == 2
        assert isinstance(config.items[0], SimpleConfig)
        assert config.items[0].b == 'one'

    def test_load_from_yaml(self):
        """Test loading from YAML file."""
        if not os.path.exists(YAML_PATH):
            pytest.skip(f"Template yaml not found at {YAML_PATH}")

        config = TrainConfig.load_from_yaml(YAML_PATH)
        assert isinstance(config, TrainConfig)

        # Verify CheckpointConfig
        assert isinstance(config.checkpoint, CheckpointConfig)
        assert config.checkpoint.save_max == 5

        # Verify TrainingConfig
        assert isinstance(config.training, TrainingConfig)
        assert config.training.steps == 100

        # Verify OptimizerConfig
        assert isinstance(config.optimizer, OptimizerConfig)
        assert config.optimizer.type == "AdamW"
        assert config.optimizer.betas == [0.9, 0.95]

        # Verify Callbacks (List of dataclasses/BaseConfig)
        assert isinstance(config.callbacks, list)
        if config.callbacks:
            assert isinstance(config.callbacks[0], CallbackConfig)
            assert config.callbacks[0].type == "callbackA"
            assert config.callbacks[0].param == 0

    def test_load_from_yaml_errors(self):
        """Test errors when loading YAML."""
        # File not found
        with pytest.raises(FileNotFoundError):
            TrainConfig.load_from_yaml("non_existent_file.yaml")

    def test_utils_check_type_scalar(self):
        """Test check_type with scalar types."""
        check_type(1, int, "test_int")
        check_type(1.0, float, "test_float")
        check_type("s", str, "test_str")
        check_type(True, bool, "test_bool")

        # Int allows bool in python (bool is subclass of int), but check_type might be strict
        # Looking at utils.py:
        # if expected_type is int: if not isinstance(value, int) or isinstance(value, bool): raise
        with pytest.raises(TypeError):
            check_type(True, int, "bool_as_int")

        with pytest.raises(TypeError):
            check_type(1.0, int, "float_as_int")

        with pytest.raises(TypeError):
            check_type(1, str, "int_as_str")

        # Test failures for bool and float to cover lines 98 and 108
        with pytest.raises(TypeError):
            check_type("not_bool", bool, "fail_bool")

        with pytest.raises(TypeError):
            check_type("not_float", float, "fail_float")

    def test_utils_check_type_list(self):
        """Test check_type with List."""
        check_type([1, 2], List[int], "test_list")
        check_type([], List[int], "test_empty_list")

        # Test raw list
        check_type([1, "s"], list, "raw_list")
        check_type([1, "s"], List, "raw_List_typing")

        with pytest.raises(TypeError):
            check_type("not_list", List[int], "not_a_list")

        with pytest.raises(TypeError):
            check_type([1, "s"], List[int], "mixed_list")

    def test_utils_check_type_union_optional(self):
        """Test check_type with Union and Optional."""
        check_type(None, Optional[int], "test_opt_none")
        check_type(1, Optional[int], "test_opt_val")
        check_type(1, Union[int, str], "test_union_int")
        check_type("s", Union[int, str], "test_union_str")

        # Test Union with None but value is not None (covers line 59: if t is type(None): continue)
        check_type(1, Union[int, None], "test_union_none_val")
        # Try explicit NoneType first to ensure we hit the continue
        check_type(1, Union[type(None), int], "test_union_none_first")

        with pytest.raises(TypeError):
            check_type(None, Union[int, str], "union_none")

        with pytest.raises(TypeError):
            check_type(1.0, Union[int, str], "union_invalid")

    def test_utils_check_type_dataclass(self):
        """Test check_type with Dataclass."""
        config = SimpleConfig()
        check_type(config, SimpleConfig, "test_dataclass")

        with pytest.raises(TypeError):
            check_type({}, SimpleConfig, "dict_as_dataclass")

    def test_utils_check_type_any(self):
        """Test check_type with Any."""
        check_type(1, Any, "test_any")
        check_type(None, Any, "test_any_none")

    def test_utils_check_type_none(self):
        """Test check_type with None value for non-optional."""
        with pytest.raises(TypeError):
            check_type(None, int, "none_for_int")

    def test_train_config_defaults(self):
        """Test TrainConfig default values."""
        config = TrainConfig()
        assert isinstance(config.checkpoint, CheckpointConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.parallelism, ParallelismConfig)
        assert config.training.steps == 1000  # Default in class definition
