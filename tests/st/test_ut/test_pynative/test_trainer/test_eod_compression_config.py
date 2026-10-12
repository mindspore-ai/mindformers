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
"""Dataset EOD compression must select the model layout before construction."""

import pytest

from mindformers.pynative.config.config import TrainConfig
from mindformers.pynative.trainer.utils import _build_model
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    "loader,model_enabled,expected_enabled",
    [
        pytest.param({"type": "HFDataLoader", "create_compressed_eod_mask": True},
                     False, True, id="hf-top-level-enabled"),
        pytest.param({"type": "HFDataLoader", "create_compressed_eod_mask": False},
                     False, False, id="hf-disabled"),
        pytest.param({"type": "HFDataLoader"}, False, False, id="hf-missing-flag"),
        pytest.param({"type": "OrderedIndexDataLoader", "create_compressed_eod_mask": True},
                     False, True, id="ordered-index-top-level-enabled"),
        pytest.param({"type": "OrderedIndexDataLoader", "create_compressed_eod_mask": False},
                     False, False, id="ordered-index-disabled"),
        pytest.param({"type": "BlendedMegatronDatasetDataLoader",
                      "config": {"create_compressed_eod_mask": True}},
                     False, True, id="megatron-nested-enabled"),
        pytest.param({"type": "BlendedMegatronDatasetDataLoader",
                      "config": {"create_compressed_eod_mask": False}},
                     False, False, id="megatron-nested-disabled"),
        pytest.param({"type": "HFDataLoader", "create_compressed_eod_mask": False},
                     True, True, id="preserve-explicit-model-enable"),
        pytest.param({"type": "HFDataLoader", "create_compressed_eod_mask": True,
                      "config": {"create_compressed_eod_mask": False}},
                     False, True, id="hf-uses-its-top-level-setting"),
        pytest.param({"type": "HFDataLoader", "create_compressed_eod_mask": False,
                      "config": {"create_compressed_eod_mask": True}},
                     False, False, id="hf-ignores-unconsumed-nested-setting"),
        pytest.param({"type": "BlendedMegatronDatasetDataLoader",
                      "create_compressed_eod_mask": False,
                      "config": {"create_compressed_eod_mask": True}},
                     False, True, id="megatron-uses-its-nested-setting"),
    ],
)
def test_build_model_syncs_compression_before_layout_selection(
        monkeypatch, loader, model_enabled, expected_enabled):
    """Exercise the model builder/converter in PyNative mode without allocating weights."""
    monkeypatch.setattr("mindformers.tools.register.register.is_legacy_model", lambda: False)
    config = TrainConfig.from_dict({
        "model": {
            "model_type": "deepseek_v4",
            "architectures": "DeepseekV4ForCausalLM",
            "num_hidden_layers": 4,
            "experimental_attention_variant": "mla",
            "use_flash_attention": True,
            "use_eod_attn_mask_compression": model_enabled,
        },
        "lr_scheduler": {"type": "ConstantWarmUpLR"},
        "train_dataset": {"dataloader": loader},
    })
    original_factory = MindFormerRegister.get_instance_from_cfg

    def factory(cfg, module_type, **kwargs):
        if module_type == MindFormerModuleType.MODELS:
            model_config = kwargs["default_args"]["config"]
            # Conversion is performed at this same boundary by the real model constructor.
            return model_config.convert_to_transformer_config(is_mla_model=True)
        return original_factory(cfg, module_type, **kwargs)

    monkeypatch.setattr(MindFormerRegister, "get_instance_from_cfg", staticmethod(factory))
    transformer_config = _build_model(config)
    assert transformer_config.use_eod_attn_mask_compression is expected_enabled
    assert (transformer_config.input_layout == "TND") is expected_enabled
