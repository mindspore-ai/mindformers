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
"""CPU unit tests for AdamW fp32 master-weights (mixed precision)."""

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, Parameter, nn, dtype as mstype

from mindformers.pynative.optimizer.adamw import AdamW, _run_adamw_opt


class MixedNet(nn.Cell):
    """Tiny net with one bf16, one fp16 and one fp32 parameter (fixed order)."""

    def __init__(self):
        super().__init__()
        # Integer values are exactly representable in bf16/fp16, so casts are lossless.
        self.w_bf16 = Parameter(Tensor(np.array([1.0, 2.0, 3.0]), mstype.bfloat16), name="w_bf16")
        self.w_fp16 = Parameter(Tensor(np.array([4.0, 5.0]), mstype.float16), name="w_fp16")
        self.w_fp32 = Parameter(Tensor(np.array([6.0, 7.0]), mstype.float32), name="w_fp32")


class TestAdamWMasterWeights:
    """CPU tests for AdamW fp32 master-weights creation, sync and reload."""

    def setup_method(self):
        """Fresh net + optimizer per test (PYNATIVE mode)."""
        ms.set_context(mode=ms.PYNATIVE_MODE)
        self.net = MixedNet()
        # trainable_params() preserves definition order: [w_bf16, w_fp16, w_fp32].
        self.opt = AdamW(self.net.trainable_params())

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_main_params_classification(self):
        """
        Feature: AdamW._init_main_params low-precision classification.
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
        Feature: AdamW fp32 master copies for low-precision params.
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
        Feature: AdamW reuses fp32 params as their own master.
        Description: an already-fp32 param must not be copied — the master IS the param.
        Expectation: fp32_params[2] is the same object as _parameters[2].
        """
        assert self.opt.fp32_params[2] is self.opt._parameters[2]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_copy_main_params_to_model_params_casts_lp_only(self):
        """
        Feature: AdamW._copy_main_params_to_model_params (master -> model sync).
        Description: cast fp32 master back to model dtype for low-precision params only.
        Expectation: bf16 model param equals the cast-down new value; fp32 model param untouched.
        """
        # Mutate the bf16 master to a new (bf16-exact) value.
        self.opt.fp32_params[0].copy_(Tensor(np.array([16.0, 32.0, 48.0]), mstype.float32))
        fp32_param_before = self.opt._parameters[2].asnumpy().copy()

        self.opt._copy_main_params_to_model_params()

        # bf16 model param picked up the new value (cast fp32 -> bf16, integers are exact).
        assert self.opt._parameters[0].dtype == mstype.bfloat16
        assert np.allclose(
            self.opt._parameters[0].astype(mstype.float32).asnumpy(),
            np.array([16.0, 32.0, 48.0]),
        )
        # fp32 param is the reused master (no-op branch) — unchanged.
        assert np.allclose(self.opt._parameters[2].asnumpy(), fp32_param_before)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reload_main_params_from_model_lp_only(self):
        """
        Feature: AdamW.reload_main_params_from_model (weights-only resume).
        Description: copy freshly-loaded model params back into the fp32 masters.
        Expectation: bf16/fp16 masters follow the model (upcast); fp32 master unchanged.
        """
        # Simulate a checkpoint load that changed the bf16 model param.
        self.opt._parameters[0].copy_(Tensor(np.array([9.0, 9.0, 9.0]), mstype.bfloat16))
        fp32_master_before = self.opt.fp32_params[2].asnumpy().copy()

        try:
            self.opt.reload_main_params_from_model()
        except RuntimeError as exc:  # pragma: no cover - environment dependent
            if "dtensor" in str(exc).lower() or "dispatch" in str(exc).lower():
                pytest.skip(f"reload needs DTensor dispatch context: {exc}")
            raise

        assert np.allclose(self.opt.fp32_params[0].asnumpy(), np.array([9.0, 9.0, 9.0]))
        # The fp32-reused slot is not a low-precision param, so reload leaves it alone.
        assert np.allclose(self.opt.fp32_params[2].asnumpy(), fp32_master_before)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_run_adamw_opt_fp32_math(self):
        """
        Feature: _run_adamw_opt fp32 accumulation core.
        Description: momentum/variance update in fp32; return cast back to param dtype.
        Expectation: exp_avg=(1-beta1)*g, exp_avg_sq=(1-beta2)*g^2 in fp32; return dtype=fp16.
        """
        beta1, beta2 = 0.9, 0.999
        params = Tensor(np.array([2.0, 2.0]), mstype.float16)
        grads = Tensor(np.array([1.0, 1.0]), mstype.float16)
        exp_avg = Parameter(Tensor(np.zeros(2), mstype.float32), name="exp_avg")
        exp_avg_sq = Parameter(Tensor(np.zeros(2), mstype.float32), name="exp_avg_sq")

        ret = _run_adamw_opt(
            beta1, beta2, 1e-8, 0.0, 0.0, params, grads,
            exp_avg, exp_avg_sq, True, 1.0, 1.0, 1.0 - beta2,
        )

        assert ret.dtype == mstype.float16
        assert exp_avg.dtype == mstype.float32 and exp_avg_sq.dtype == mstype.float32
        assert np.allclose(exp_avg.asnumpy(), (1.0 - beta1) * np.array([1.0, 1.0]))
        assert np.allclose(exp_avg_sq.asnumpy(), (1.0 - beta2) * np.array([1.0, 1.0]))

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_run_adamw_opt_filter_false_passthrough(self):
        """
        Feature: _run_adamw_opt optim_filter=False branch.
        Description: when filtered out, just cast the grad to param dtype, no state update.
        Expectation: returns grad cast to fp16; exp_avg/exp_avg_sq stay zero.
        """
        params = Tensor(np.array([2.0, 2.0]), mstype.float16)
        grads = Tensor(np.array([1.0, 1.0]), mstype.float16)
        exp_avg = Parameter(Tensor(np.zeros(2), mstype.float32), name="exp_avg2")
        exp_avg_sq = Parameter(Tensor(np.zeros(2), mstype.float32), name="exp_avg_sq2")

        ret = _run_adamw_opt(
            0.9, 0.999, 1e-8, 0.0, 0.0, params, grads,
            exp_avg, exp_avg_sq, False, 1.0, 1.0, 0.001,
        )

        assert ret.dtype == mstype.float16
        assert np.allclose(ret.asnumpy(), np.array([1.0, 1.0]))
        assert np.allclose(exp_avg.asnumpy(), np.zeros(2))
        assert np.allclose(exp_avg_sq.asnumpy(), np.zeros(2))
