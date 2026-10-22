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
"""Checkpoint loading must preserve restored FP32 masters, regardless of their sums."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import mindspore as ms
from mindspore import Parameter, Tensor, nn

from mindformers.checkpoint import checkpoint as checkpoint_module
from mindformers.checkpoint.checkpoint import load_parameters
from mindformers.pynative.optimizer.adamw import AdamW
from mindformers.pynative.optimizer.muon import Muon
from mindformers.pynative.optimizer import muon as muon_module


class ResumeNet(nn.Cell):
    """Two low-precision weights, allowing a checkpoint to omit one master."""

    def __init__(self):
        super().__init__()
        self.weight = Parameter(Tensor([1., 1.], ms.bfloat16), name="weight")
        self.other = Parameter(Tensor([2., 2.], ms.float16), name="other")


def _split(*_args):
    return None


_split._muon_schema = ()


@pytest.fixture(name="resume_optimizer", params=["muon", "adamw"])
def create_resume_optimizer(request, monkeypatch):
    """Exercise the shared resume code through both production optimizers."""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    monkeypatch.setattr(checkpoint_module.LayoutAdapter, "is_pynative_mode", lambda: True)
    net = ResumeNet()
    if request.param == "muon":
        monkeypatch.setattr(muon_module.core_context, "is_legacy_model", lambda: False)
        model = SimpleNamespace(
            get_gpt_transformer_config=lambda: SimpleNamespace(multi_latent_attention=True),
            make_model_muon_fns=lambda: (_split, lambda *_: None),
        )
        optimizer = Muon(net.trainable_params(), model=model, comm_strategy="allgather")
    else:
        optimizer = AdamW(net.trainable_params())
    return net, optimizer


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_resume_preserves_loaded_master_with_unchanged_sum(resume_optimizer):
    """The loader-to-optimizer path retains sub-BF16 precision when sums collide."""
    net, optimizer = resume_optimizer
    main = next(p for p in optimizer.fp32_params if p.name == "main_param.weight")
    saved = np.array([1.0001, 0.9999], dtype=np.float32)
    assert float(Tensor(saved).sum()) == float(main.sum())
    state = {
        "weight": Parameter(Tensor([1., 1.], ms.bfloat16), name="weight"),
        "main_param.weight": Parameter(Tensor(saved), name="main_param.weight"),
    }
    loaded_names = load_parameters(net, state, optimizer)
    optimizer.reload_main_params_from_model(loaded_names)
    np.testing.assert_array_equal(main.asnumpy(), saved)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_resume_refreshes_only_missing_masters(resume_optimizer):
    """A partial optimizer checkpoint preserves its master and refreshes the missing one."""
    net, optimizer = resume_optimizer
    saved = np.array([1.0001, 0.9999], dtype=np.float32)
    state = {
        "weight": Parameter(Tensor([1., 1.], ms.bfloat16), name="weight"),
        "other": Parameter(Tensor([7., 8.], ms.float16), name="other"),
        "main_param.weight": Parameter(Tensor(saved), name="main_param.weight"),
    }
    loaded_names = load_parameters(net, state, optimizer)
    assert "main_param.weight" in loaded_names
    assert "main_param.other" not in loaded_names
    optimizer.reload_main_params_from_model(loaded_names)
    masters = {p.name: p.asnumpy() for p in optimizer.fp32_params}
    np.testing.assert_array_equal(masters["main_param.weight"], saved)
    np.testing.assert_array_equal(masters["main_param.other"], [7., 8.])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_weights_only_resume_refreshes_all_masters(resume_optimizer):
    """Without optimizer loading, every low-precision master follows the loaded model."""
    net, optimizer = resume_optimizer
    load_parameters(net, {"weight": Parameter(Tensor([5., 6.], ms.bfloat16), name="weight")})
    optimizer.reload_main_params_from_model()
    main = next(p for p in optimizer.fp32_params if p.name == "main_param.weight")
    np.testing.assert_array_equal(main.asnumpy(), [5., 6.])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("has_local_optimizer_state", [False, True])
def test_loaded_names_follow_loader_and_broadcast_results(
        resume_optimizer, monkeypatch, balanced, has_local_optimizer_state):
    """Broadcast receivers participate even when no optimizer state was read locally."""
    net, optimizer = resume_optimizer
    missing = ["main_param.weight", "main_param.other"]
    monkeypatch.setattr(checkpoint_module, "load_param_into_net", Mock(side_effect=[([], []), (missing, [])]))

    def broadcast(*args):
        network, _, not_loaded, _ = args
        if network is optimizer:
            not_loaded.remove("main_param.weight")

    monkeypatch.setattr(checkpoint_module, "single_parameter_broadcast", broadcast)
    state = {"weight": Parameter(Tensor([1., 1.], ms.bfloat16), name="weight")}
    if has_local_optimizer_state:
        state["main_param.weight"] = Parameter(Tensor([1., 1.]), name="main_param.weight")
    names = load_parameters(net, state, optimizer, balanced_load=balanced, param_redundancy={})
    assert ("main_param.weight" in names) == balanced
    assert "main_param.other" not in names
