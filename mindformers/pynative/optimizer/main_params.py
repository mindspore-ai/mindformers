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
"""Shared fp32 master weight (main params) helpers for pynative optimizers."""
from typing import Optional, Set

from mindspore import _no_grad
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P

from hyper_parallel import SkipDTensorDispatch

from mindformers.pynative.dtensor_compat import inplace_copy

op_cast = P.Cast()


class MainParamsMixin:
    """Reload helpers for optimizers holding fp32 master weights.

    Expects the host optimizer to provide ``self._parameters``, ``self.fp32_params``
    and ``self._is_low_precision_param``.
    """

    def reload_main_params_from_model(self, loaded_param_names: Optional[Set[str]] = None):
        """Refresh fp32 master weights from the model parameters.

        When optimizer state was loaded, pass the names actually restored by the
        checkpoint loader. Loaded masters are preserved even if their values or
        sums equal their initialization. Missing masters (e.g. newly unfrozen
        parameters) are refreshed from the model. With ``None``, refresh every
        low-precision master for weights-only loading.

        Runs outside the optimizer's construct(), so it must enter SkipDTensorDispatch
        itself (InplaceCopy has no parallel layout infer func under DTensor dispatch).
        _no_grad avoids the "leaf tensor that requires grad in an inplace operator"
        error: the fp32 master is a leaf Parameter and this copy is not an autograd op.
        """
        needs_reload = [
            is_lp and (loaded_param_names is None or fp32_param.name not in loaded_param_names)
            for fp32_param, is_lp in zip(self.fp32_params, self._is_low_precision_param)
        ]
        self._reload_main_params(needs_reload)

    def _reload_main_params(self, reload_flags):
        """Copy model params into the fp32 masters selected by ``reload_flags``."""
        with _no_grad(), SkipDTensorDispatch():
            for model_param, fp32_param, do_reload in zip(
                    self._parameters, self.fp32_params, reload_flags):
                if do_reload:
                    inplace_copy(fp32_param, op_cast(model_param, mstype.float32))
