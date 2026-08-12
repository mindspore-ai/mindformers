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
from mindspore import _no_grad
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P

from hyper_parallel import SkipDTensorDispatch

from mindformers.pynative.dtensor_compat import inplace_copy

op_cast = P.Cast()

_CHECKSUM_TOLERANCE = 1e-6


class MainParamsMixin:
    """Snapshot / reload helpers for optimizers holding fp32 master weights.

    Expects the host optimizer to provide ``self._parameters``, ``self.fp32_params``
    and ``self._is_low_precision_param``, and to initialize
    ``self._main_params_snapshot = None`` in its constructor.
    """

    def save_main_params_snapshot(self):
        """Save lightweight checksums of fp32 master weights before checkpoint load."""
        self._main_params_snapshot = []
        for fp32_param, is_lp in zip(self.fp32_params, self._is_low_precision_param):
            if is_lp:
                self._main_params_snapshot.append(float(fp32_param.value().sum()))
            else:
                self._main_params_snapshot.append(None)

    def reload_main_params_from_model(self):
        """Refresh fp32 master weights from the model parameters.

        When ``save_main_params_snapshot`` was called before checkpoint load, only
        params whose fp32 master checksum did *not* change (i.e. not loaded from
        checkpoint, e.g. newly unfrozen params) are refreshed — others are preserved.
        Without a snapshot, does a full reload (weights-only resume / no_load_optim).

        Runs outside the optimizer's construct(), so it must enter SkipDTensorDispatch
        itself (InplaceCopy has no parallel layout infer func under DTensor dispatch).
        _no_grad avoids the "leaf tensor that requires grad in an inplace operator"
        error: the fp32 master is a leaf Parameter and this copy is not an autograd op.
        """
        snapshot = getattr(self, '_main_params_snapshot', None)
        if snapshot is None:
            self._reload_main_params(self._is_low_precision_param)
            return

        needs_reload = []
        with _no_grad():
            for idx, (fp32_param, is_lp) in enumerate(
                    zip(self.fp32_params, self._is_low_precision_param)):
                if not is_lp:
                    needs_reload.append(False)
                    continue
                current = float(fp32_param.value().sum())
                needs_reload.append(abs(current - snapshot[idx]) < _CHECKSUM_TOLERANCE)

        self._reload_main_params(needs_reload)
        self._main_params_snapshot = None

    def _reload_main_params(self, reload_flags):
        """Copy model params into the fp32 masters selected by ``reload_flags``."""
        with _no_grad(), SkipDTensorDispatch():
            for model_param, fp32_param, do_reload in zip(
                    self._parameters, self.fp32_params, reload_flags):
                if do_reload:
                    inplace_copy(fp32_param, op_cast(model_param, mstype.float32))
