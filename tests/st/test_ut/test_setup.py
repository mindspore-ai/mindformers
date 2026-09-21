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
"""UTs for the commit-id recording in setup.py."""
import importlib.util
import os
import shutil
import subprocess

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FALLBACK = b"git is not available while building.\n"


def _load_setup():
    """Import setup.py as a module; its setup() call is behind ``__main__``."""
    spec = importlib.util.spec_from_file_location("mindformers_setup", os.path.join(REPO_ROOT, "setup.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_commit_id(build_dir):
    with open(os.path.join(build_dir, "mindformers", ".commit_id"), "rb") as f:
        return f.read()


@pytest.fixture(name="build_dir")
def fixture_build_dir(tmp_path, monkeypatch):
    """A source tree to run write_commit_id from, as setup.py is run from the repo root."""
    (tmp_path / "mindformers").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_write_commit_id_records_branch_and_last_commit(build_dir):
    """
    Feature: setup.write_commit_id
    Description: Run it at the root of a git work tree.
    Expectation: .commit_id holds the branch name followed by `git log --abbrev-commit -1`.
    """
    git = ["git", "-C", str(build_dir), "-c", "user.name=ut", "-c", "user.email=ut@example.com"]
    subprocess.run(git[:3] + ["init", "-q", "-b", "feat/commit-id"], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "subject with $(id) and `id`"], check=True)

    _load_setup().write_commit_id()

    lines = _read_commit_id(build_dir).decode("utf-8").splitlines()
    assert lines[0] == "feat/commit-id"
    assert lines[1].startswith("commit ")
    assert "    subject with $(id) and `id`" in lines


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_write_commit_id_falls_back_outside_a_git_tree(build_dir):
    """
    Feature: setup.write_commit_id
    Description: Run it from a directory that is not a git work tree (e.g. an unpacked sdist).
    Expectation: .commit_id holds the fallback message.
    """
    _load_setup().write_commit_id()
    assert _read_commit_id(build_dir) == FALLBACK


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_write_commit_id_falls_back_without_git(build_dir, monkeypatch):
    """
    Feature: setup.write_commit_id
    Description: Run it where git is not installed.
    Expectation: No command is started and .commit_id holds the fallback message.
    """
    setup_module = _load_setup()
    monkeypatch.setattr(setup_module.shutil, "which", lambda _: None)
    monkeypatch.setattr(setup_module.subprocess, "run",
                        lambda *args, **kwargs: pytest.fail("no command may run without git"))

    setup_module.write_commit_id()
    assert _read_commit_id(build_dir) == FALLBACK


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_write_commit_id_runs_git_without_a_shell(build_dir, monkeypatch):
    """
    Feature: setup.write_commit_id
    Description: Capture every command it starts.
    Expectation: Each is an argument list for the resolved git binary, never a shell string.
    """
    setup_module = _load_setup()
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"out\n")

    monkeypatch.setattr(setup_module.shutil, "which", lambda _: "/usr/bin/git")
    monkeypatch.setattr(setup_module.subprocess, "run", fake_run)

    setup_module.write_commit_id()

    assert [cmd for cmd, _ in calls] == [
        ["/usr/bin/git", "rev-parse", "--abbrev-ref", "HEAD"],
        ["/usr/bin/git", "log", "--abbrev-commit", "-1"],
    ]
    assert all(not kwargs.get("shell", False) for _, kwargs in calls)
    assert _read_commit_id(build_dir) == b"out\nout\n"
