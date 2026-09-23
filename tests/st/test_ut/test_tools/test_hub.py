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
"""test PushToHubMixin in hub.py"""
import sys
import types

import pytest

from mindformers.tools.hub.hub import PushToHubMixin


class _Uploader(PushToHubMixin):
    """Minimal PushToHubMixin user."""


@pytest.fixture(name="fake_openmind_hub")
def fixture_fake_openmind_hub(monkeypatch):
    """Replace openmind_hub so no network is touched; record create_commit calls."""
    commits = []
    fake = types.ModuleType("openmind_hub")
    fake.CommitOperationAdd = lambda **kwargs: kwargs
    fake.create_branch = lambda **kwargs: None
    fake.create_commit = lambda **kwargs: commits.append(kwargs) or "commit-ok"
    monkeypatch.setitem(sys.modules, "openmind_hub", fake)
    return commits


SHORT_SECRET = "om_x"
LONG_SECRET = "om_" + "s" * 125


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("token, expected", [
    ("".join(["om_", "x"]), SHORT_SECRET),
    ("".join(["om_", "s" * 125]), LONG_SECRET),
    (True, True),
    (False, False),
    (None, None),
])
def test_upload_modified_files_leaves_token_intact(tmp_path, fake_openmind_hub, token, expected):
    """
    Feature: PushToHubMixin._upload_modified_files.
    Description: upload twice with the same token object, as push_to_hub callers do for model and tokenizer.
    Expectation: the token reaches create_commit unchanged both times and stays usable afterwards.
    """
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    for _ in range(2):
        assert _Uploader()._upload_modified_files(str(tmp_path), "me/repo", {}, token=token) == "commit-ok"
    assert [commit["token"] for commit in fake_openmind_hub] == [expected, expected]
    assert token == expected
    assert bool(1) is True and bool(0) is False


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_upload_modified_files_failed_commit_leaves_token_intact(tmp_path, monkeypatch, fake_openmind_hub):
    """
    Feature: PushToHubMixin._upload_modified_files.
    Description: create_commit raises (e.g. a token without write access).
    Expectation: the error reaches the caller and the token is still usable for a retry.
    """
    def _reject(**kwargs):
        raise PermissionError("403: token has no write access")

    monkeypatch.setattr(sys.modules["openmind_hub"], "create_commit", _reject)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    token = "".join(["om_", "x"])
    with pytest.raises(PermissionError):
        _Uploader()._upload_modified_files(str(tmp_path), "me/repo", {}, token=token)
    assert token == SHORT_SECRET
    assert not fake_openmind_hub
