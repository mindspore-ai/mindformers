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
"""CPU unit tests for Muon fp32 master-weights (mixed precision).

Muon's master-weights methods mirror AdamW's, but Muon's ``__init__`` requires a
``model`` (MLA config + muon split/merge fns + muon filter). We feed a lightweight
mock model and make every param go through the AdamW branch (``get_muon_filter``
returns False), so construction stays on CPU and never touches Newton-Schulz.
"""

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, Parameter, nn, dtype as mstype

from mindformers.pynative.optimizer import muon as muon_mod
from mindformers.pynative.optimizer.muon import Muon


class MixedNet(nn.Cell):
    """Tiny net with one bf16, one fp16 and one fp32 parameter (fixed order)."""

    def __init__(self):
        super().__init__()
        self.w_bf16 = Parameter(Tensor(np.array([1.0, 2.0, 3.0]), mstype.bfloat16), name="w_bf16")
        self.w_fp16 = Parameter(Tensor(np.array([4.0, 5.0]), mstype.float16), name="w_fp16")
        self.w_fp32 = Parameter(Tensor(np.array([6.0, 7.0]), mstype.float32), name="w_fp32")


def _muon_split_fn(*_args, **_kwargs):
    return None


_muon_split_fn._muon_schema = ()  # Muon reads this attribute lazily; empty = no split rules.


def _muon_merge_fn(*_args, **_kwargs):
    return None


def _make_mock_model():
    """Minimal model double satisfying Muon.__init__ without a real GPT model."""

    class _MockCfg:
        multi_latent_attention = True  # Muon._verify_model requires this.

    class _MockModel:
        def get_gpt_transformer_config(self):
            return _MockCfg()

        def make_model_muon_fns(self):
            return (_muon_split_fn, _muon_merge_fn)

        def get_muon_filter(self):
            # All params take the AdamW branch -> no Newton-Schulz, CPU-safe.
            return lambda _param: False

    return _MockModel()


class TestMuonMasterWeights:
    """CPU tests for Muon fp32 master-weights creation, sync and reload."""

    def setup_method(self):
        """Fresh net + Muon per test; stub is_legacy_model to False (restored in teardown)."""
        ms.set_context(mode=ms.PYNATIVE_MODE)
        self._orig_is_legacy_model = muon_mod.core_context.is_legacy_model
        muon_mod.core_context.is_legacy_model = lambda: False
        self.net = MixedNet()
        self.opt = Muon(
            self.net.trainable_params(),
            model=_make_mock_model(),
            comm_strategy="allgather",
        )

    def teardown_method(self):
        muon_mod.core_context.is_legacy_model = self._orig_is_legacy_model

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_main_params_classification(self):
        """
        Feature: Muon._init_main_params low-precision classification.
        Description: bf16/fp16 params need an fp32 master; fp32 params do not.
        Expectation: _is_low_precision_param == (True, True, False); 3 fp32 slots.
        """
        assert self.opt._is_low_precision_param == (True, True, False)
        assert len(self.opt.fp32_params) == 3

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_main_params_lp_are_fresh_fp32_copies(self):
        """
        Feature: Muon fp32 master copies for low-precision params.
        Description: master is an independent fp32 Parameter holding the upcast value.
        Expectation: dtype is fp32, value matches the upcast model param, not the same object.
        """
        for i in (0, 1):  # bf16, fp16 slots
            master = self.opt.fp32_params[i]
            model_param = self.opt._parameters[i]
            assert master.dtype == mstype.float32
            assert master is not model_param
            assert np.allclose(
                master.asnumpy(),
                model_param.astype(mstype.float32).asnumpy(),
            )

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_main_params_fp32_reused_in_place(self):
        """
        Feature: Muon reuses fp32 params as their own master.
        Description: an already-fp32 param must not be copied — the master IS the param.
        Expectation: fp32_params[2] is the same object as _parameters[2].
        """
        assert self.opt.fp32_params[2] is self.opt._parameters[2]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_copy_main_params_to_model_params_casts_lp_only(self):
        """
        Feature: Muon._copy_main_params_to_model_params (master -> model sync).
        Description: cast fp32 master back to model dtype for low-precision params only.
        Expectation: fp16 model param equals the cast-down new value; fp32 model param untouched.
        """
        self.opt.fp32_params[1].copy_(Tensor(np.array([40.0, 50.0]), mstype.float32))
        fp32_param_before = self.opt._parameters[2].asnumpy().copy()

        self.opt._copy_main_params_to_model_params()

        assert self.opt._parameters[1].dtype == mstype.float16
        assert np.allclose(
            self.opt._parameters[1].astype(mstype.float32).asnumpy(),
            np.array([40.0, 50.0]),
        )
        assert np.allclose(self.opt._parameters[2].asnumpy(), fp32_param_before)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reload_main_params_from_model_lp_only(self):
        """
        Feature: Muon.reload_main_params_from_model (weights-only resume).
        Description: copy freshly-loaded model params back into the fp32 masters.
        Expectation: bf16/fp16 masters follow the model (upcast); fp32 master unchanged.
        """
        self.opt._parameters[0].copy_(Tensor(np.array([9.0, 9.0, 9.0]), mstype.bfloat16))
        fp32_master_before = self.opt.fp32_params[2].asnumpy().copy()

        try:
            self.opt.reload_main_params_from_model()
        except RuntimeError as exc:  # pragma: no cover - environment dependent
            if "dtensor" in str(exc).lower() or "dispatch" in str(exc).lower():
                pytest.skip(f"reload needs DTensor dispatch context: {exc}")
            raise

        assert np.allclose(self.opt.fp32_params[0].asnumpy(), np.array([9.0, 9.0, 9.0]))
        assert np.allclose(self.opt.fp32_params[2].asnumpy(), fp32_master_before)
