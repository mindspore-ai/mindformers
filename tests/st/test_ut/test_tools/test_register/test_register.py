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
"""test register.py"""
import sys

import pytest

from mindformers.tools.hub.dynamic_module_utils import HubConstants
from mindformers.tools.register.register import MindFormerRegister, MindFormerModuleType
from .model_class import MyModel, MyTool


class TestMindFormerRegister:
    """test MindFormerRegister"""
    def test_register_other_type_case(self):
        """Tools remain available in the shared registry."""
        assert MindFormerRegister.is_exist(MindFormerModuleType.TOOLS, "MyTool")
        assert MindFormerRegister.get_cls(MindFormerModuleType.TOOLS, "MyTool") is MyTool

    def test_register_decorator_and_is_exist_case(self):
        """Models use their declared names without a runtime namespace prefix."""
        assert MindFormerRegister.is_exist(MindFormerModuleType.MODELS, "MyModel")
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "MyModel") is MyModel
        assert "mcore_MyModel" not in MindFormerRegister.registry[MindFormerModuleType.MODELS]

    def test_register_cls_manual_case(self):
        """Manual registration resolves the same names as decorator registration."""
        class ManualModel:
            pass

        MindFormerRegister.register_cls(ManualModel, MindFormerModuleType.MODELS, search_names="manual_model")
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "ManualModel") is ManualModel
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "manual_model") is ManualModel

    def test_model_construction_without_context_patch(self):
        """Configuration and model construction use the PyNative configuration fields."""
        class AtomicTestConfig:
            def __init__(self, width, **kwargs):
                self.width = width
                self.model_type = kwargs.get("model_type")

        class AtomicTestModel:
            def __init__(self, config):
                self.config = config

        MindFormerRegister.register_cls(
            AtomicTestConfig, MindFormerModuleType.CONFIG, search_names="atomic_test_config")
        MindFormerRegister.register_cls(AtomicTestModel, MindFormerModuleType.MODELS)
        cfg = {"model_type": "atomic_test_config"}
        cfg["width"] = 8
        config = MindFormerRegister.get_instance_from_cfg(cfg, MindFormerModuleType.CONFIG)
        model_cfg = {"architectures": ["AtomicTestModel"]}
        model = MindFormerRegister.get_instance_from_cfg(
            model_cfg, MindFormerModuleType.MODELS, default_args={"config": config})
        assert model.config.width == 8

    def test_get_cls_not_exist_case(self):
        """
        Test querying a non-existent class, should raise ValueError.
        Input: Query NotExistModel.
        Output: ValueError is raised.
        Expected: Exception is thrown.
        """
        with pytest.raises(ValueError):
            MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "NotExistModel")

    def test_is_exist_without_class_name_case(self):
        """
        Test checking if the type exists in registry.
        Input: Only module_type provided.
        Output: Returns True if type exists.
        Expected: is_exist returns True.
        """
        assert MindFormerRegister.is_exist(MindFormerModuleType.MODELS)

    def test_auto_register_rejects_remote_reference_case(self, monkeypatch, tmp_path):
        """
        Test auto_register with a `repo_id--module.Class` reference.
        Input: REGISTER_PATH set, class_reference pointing at a remote repository.
        Output: ValueError is raised before any module is fetched or imported.
        Expected: get_class_from_dynamic_module is never called.
        """
        monkeypatch.setenv("REGISTER_PATH", str(tmp_path))
        calls = []
        monkeypatch.setattr("mindformers.tools.register.register.get_class_from_dynamic_module",
                            lambda *args, **kwargs: calls.append((args, kwargs)))
        with pytest.raises(ValueError, match="REGISTER_PATH"):
            MindFormerRegister.auto_register(class_reference="attacker/repo--evil.EvilTool",
                                             module_type=MindFormerModuleType.TOOLS)
        assert not calls

    def test_auto_register_local_reference_case(self, monkeypatch, tmp_path):
        """
        Test auto_register with a local `module_file.class_name` reference.
        Input: REGISTER_PATH holding the module file.
        Output: The class is loaded from REGISTER_PATH and registered.
        Expected: get_cls returns the loaded class.
        """
        register_path = tmp_path / "register_path"
        register_path.mkdir()
        (register_path / "auto_register_tool.py").write_text("class AutoRegisteredTool:\n    pass\n", encoding="utf-8")
        monkeypatch.setenv("REGISTER_PATH", str(register_path))
        monkeypatch.setattr(HubConstants, "OM_MODULES_CACHE", str(tmp_path / "modules_cache"))
        monkeypatch.setattr(sys, "path", list(sys.path))
        MindFormerRegister.auto_register(class_reference="auto_register_tool.AutoRegisteredTool",
                                         module_type=MindFormerModuleType.TOOLS)
        assert MindFormerRegister.get_cls(MindFormerModuleType.TOOLS, "AutoRegisteredTool").__name__ \
            == "AutoRegisteredTool"

    @pytest.mark.parametrize("class_reference, register_path, error", [
        (["not", "a", "str"], "tmp", ValueError),
        ("module_file.ClassName", None, EnvironmentError),
        ("module_file.ClassName", "missing", EnvironmentError),
    ], ids=["not_str", "register_path_unset", "register_path_missing"])
    def test_auto_register_invalid_input_case(self, monkeypatch, tmp_path, class_reference, register_path, error):
        """
        Test auto_register input checks.
        Input: a non-str class_reference, REGISTER_PATH unset, or REGISTER_PATH pointing at no directory.
        Output: the matching explicit error, before any module is fetched or imported.
        Expected: get_class_from_dynamic_module is never called.
        """
        if register_path is None:
            monkeypatch.delenv("REGISTER_PATH", raising=False)
        else:
            monkeypatch.setenv("REGISTER_PATH", str(tmp_path if register_path == "tmp" else tmp_path / register_path))
        calls = []
        monkeypatch.setattr("mindformers.tools.register.register.get_class_from_dynamic_module",
                            lambda *args, **kwargs: calls.append((args, kwargs)))
        with pytest.raises(error):
            MindFormerRegister.auto_register(class_reference=class_reference, module_type=MindFormerModuleType.TOOLS)
        assert not calls
