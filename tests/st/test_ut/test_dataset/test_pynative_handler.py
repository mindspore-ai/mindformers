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
"""Instruction handlers select PyNative tokenization without a Trainer patch."""
from types import SimpleNamespace

import pytest

from mindformers.dataset.handler.alpaca_handler import AlpacaInstructDataHandler


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("padding, source_length, expected_length", [(True, 3, 8), (True, 9, 8), (False, 3, 3)])
def test_handler_tokenizer_and_sequence_length(monkeypatch, padding, source_length, expected_length):
    """Handlers use PyNative tokenization without adding an extra sequence token."""
    calls = []
    tokenizer = SimpleNamespace(pad_token_id=0, padding_side="left")

    def build_tokenizer(config, **kwargs):
        calls.append((config, kwargs))
        return tokenizer

    monkeypatch.setattr("mindformers.dataset.handler.base_handler.build_tokenizer", build_tokenizer)
    handler = AlpacaInstructDataHandler(
        seq_length=8, padding=padding,
        tokenizer={"pretrained_model_dir": "/local/tokenizer", "trust_remote_code": True})
    assert calls == [({}, {"pretrained_model_dir": "/local/tokenizer",
                          "trust_remote_code": True})]
    inputs, labels = handler._process_sequence_with_length(  # pylint: disable=protected-access
        list(range(source_length)), list(range(source_length)))
    assert len(inputs) == len(labels) == expected_length
    if padding and source_length < expected_length:
        assert inputs[source_length:] == [0] * (expected_length - source_length)
        assert labels[source_length:] == [-100] * (expected_length - source_length)
