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
"""Tests for the HuggingFace-only tokenizer builder."""
from importlib import import_module
from unittest.mock import Mock

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from mindformers.models import build_tokenizer

builder = import_module("mindformers.models.build_tokenizer")


def test_load_local_tokenizer(tmp_path):
    """Load and use real tokenizer artifacts without a legacy selector."""
    backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]").save_pretrained(tmp_path)
    tokenizer = build_tokenizer(pretrained_model_dir=str(tmp_path))
    assert tokenizer.encode("hello world") == [1, 2]


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_context_fallback_and_remote_code(monkeypatch, tmp_path, trust_remote_code):
    """Resolve the context directory and preserve the remote code option."""
    monkeypatch.setattr(builder, "get_context", lambda key: str(tmp_path))
    factory = Mock()
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", factory)
    tokenizer = build_tokenizer({}, trust_remote_code=trust_remote_code)
    assert tokenizer is factory.return_value
    factory.assert_called_once_with(pretrained_model_name_or_path=str(tmp_path.resolve()),
                                    trust_remote_code=trust_remote_code)


def test_missing_tokenizer_directory(monkeypatch):
    """A missing explicit and context directory must fail before loading."""
    monkeypatch.setattr(builder, "get_context", lambda key: None)
    with pytest.raises(ValueError, match="empty or None"):
        build_tokenizer()


def test_invalid_tokenizer_directory(tmp_path):
    """A regular file cannot be used as a pretrained tokenizer directory."""
    path = tmp_path / "tokenizer.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid directory"):
        build_tokenizer(pretrained_model_dir=str(path))
