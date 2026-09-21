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

from mindformers.core.context.build_context import build_context, set_context
from mindformers.tools.hub.dynamic_module_utils import HubConstants
from mindformers.tools.register.register import MindFormerRegister, MindFormerModuleType
from .model_class import MyModel
from .model_class_legacy import MyTool, MyModel as MyModelNew


class TestMindFormerRegister:
    """test MindFormerRegister"""
    @classmethod
    def setup_class(cls):
        build_context({"use_legacy": True})

    def test_register_other_type_case(self):
        """
        Test this feature only affects models, not other types, check existence.
        Input: MyTool registered with legacy=True.
        Output: MyTool is found in registry, when use_legacy=False.
        Expected: get_cls returns MyTool.
        """
        set_context(use_legacy=False)
        assert MindFormerRegister.is_exist(MindFormerModuleType.TOOLS, "MyTool")
        assert MindFormerRegister.get_cls(MindFormerModuleType.TOOLS, "MyTool") is MyTool

    def test_register_decorator_and_is_exist_case(self):
        """
        Test registering with decorator (legacy=True), check existence and internal keys.
        Input: MyModel registered with legacy=True.
        Output: MyModel is found in registry, both 'MyModel' and 'mcore_MyModel' keys exist.
        Expected: get_cls returns MyModel, keys present.
        """
        set_context(use_legacy=True)
        assert MindFormerRegister.is_exist(MindFormerModuleType.MODELS, "MyModel")
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "MyModel") is MyModel
        keys = list(MindFormerRegister.registry[MindFormerModuleType.MODELS].keys())
        assert "MyModel" in keys
        assert "mcore_MyModel" in keys

    def test_register_decorator_legacy_false_case(self):
        """
        Test registering with decorator (legacy=False), check existence and internal keys.
        Input: MyModel registered with legacy=False, context set to use_legacy=False.
        Output: MyModel is found in registry, both 'MyModel' and 'mcore_MyModel' keys exist.
        Expected: get_cls returns MyModelNew, keys present.
        """
        set_context(use_legacy=False)
        assert MindFormerRegister.is_exist(MindFormerModuleType.MODELS, "MyModel")
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "MyModel") is MyModelNew
        keys = list(MindFormerRegister.registry[MindFormerModuleType.MODELS].keys())
        assert "mcore_MyModel" in keys
        assert "MyModel" in keys

    def test_legacy_switching_case(self):
        """
        Test switching between legacy=True and legacy=False, check correct class is returned.
        Input: Switch context between use_legacy True/False.
        Output: get_cls returns correct class for each context.
        Expected: MyModel for legacy=True, MyModelNew for legacy=False.
        """
        set_context(use_legacy=True)
        assert MindFormerRegister.is_exist(MindFormerModuleType.MODELS, "MyModel")
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "MyModel") is MyModel

        set_context(use_legacy=False)
        assert MindFormerRegister.is_exist(MindFormerModuleType.MODELS, "MyModel")
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "MyModel") is MyModelNew

        set_context(use_legacy=True)
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "MyModel") is MyModel

    def test_register_cls_manual_case(self):
        """
        Test manually registering classes with legacy=True and legacy=False, check existence and keys.
        Input: Register ManualLegacy (legacy=True), ManualNew (legacy=False).
        Output: Both classes are found in registry, correct keys exist.
        Expected: get_cls returns correct class, keys present.
        """
        class ManualLegacy:
            pass
        MindFormerRegister.register_cls(ManualLegacy, MindFormerModuleType.MODELS, legacy=True)
        set_context(use_legacy=True)
        assert MindFormerRegister.is_exist(MindFormerModuleType.MODELS, "ManualLegacy")
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "ManualLegacy") is ManualLegacy

        class ManualNew:
            pass
        MindFormerRegister.register_cls(ManualNew, MindFormerModuleType.MODELS, legacy=False)
        set_context(use_legacy=False)
        assert MindFormerRegister.is_exist(MindFormerModuleType.MODELS, "ManualNew")
        assert MindFormerRegister.get_cls(MindFormerModuleType.MODELS, "ManualNew") is ManualNew

        keys = list(MindFormerRegister.registry[MindFormerModuleType.MODELS].keys())
        assert "mcore_ManualNew" in keys
        assert "ManualLegacy" in keys

    def test_get_cls_not_exist_case(self):
        """
        Test querying a non-existent class, should raise ValueError.
        Input: Query NotExistModel.
        Output: ValueError is raised.
        Expected: Exception is thrown.
        """
        set_context(use_legacy=True)
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
