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
"""Reject invalid mHC clamp bounds and incomplete fused-kernel builds."""
import pytest

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.transformers import hyper_connection as hc


@pytest.mark.parametrize("clamp", [float("nan"), float("inf"), -float("inf"),
                                   [float("nan"), 1], [-1, float("nan")],
                                   [-float("inf"), 1], [-1, float("inf")]])
def test_mhc_rejects_nonfinite_clamp(clamp):
    """A valid interval must not propagate NaN or select an unbounded kernel."""
    with pytest.raises(ValueError, match="finite"):
        TransformerConfig(hc_sinkhorn_clamp=clamp)


@pytest.mark.parametrize("clamp", [None, 1.0])
def test_mhc_rejects_missing_fused_post(monkeypatch, clamp):
    """Either pre-kernel may be present without the separately shipped post-kernel."""
    monkeypatch.setattr(hc, "npu_mhc_pre_sinkhorn", lambda *_: None)
    monkeypatch.setattr(hc, "npu_mhc_pre_clamp_sinkhorn", lambda *_: None)
    monkeypatch.setattr(hc, "npu_mhc_post", None)
    config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2,
                               enable_hyper_connections=True, num_residual_streams=2,
                               hc_sinkhorn_clamp=clamp)
    with pytest.raises(ImportError, match="npu_mhc_post"):
        hc.FusedHyperConnectionModule(config, 1)
