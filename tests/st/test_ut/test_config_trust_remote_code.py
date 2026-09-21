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
"""Shipped configs must not turn on trust_remote_code by default."""
import os
import re

import pytest

from mindformers.mindformer_book import MindFormerBook

TRUST_REMOTE_CODE_ON = re.compile(r"^\s*trust_remote_code\s*:\s*(true|yes|on)\b", re.IGNORECASE)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_shipped_configs_do_not_trust_remote_code():
    """
    Feature: configs/ and research/ yaml files.
    Description: scan every shipped yaml for `trust_remote_code: True`.
    Expectation: none; running a model directory's custom code must stay an explicit user choice.
    """
    root = MindFormerBook.get_project_path()
    offenders = []
    for top in ("configs", "research"):
        for dirpath, _, filenames in os.walk(os.path.join(root, top)):
            for name in filenames:
                if not name.endswith((".yaml", ".yml")):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, "r", encoding="utf-8") as f:
                    offenders += [f"{os.path.relpath(path, root)}:{lineno}"
                                  for lineno, line in enumerate(f, 1) if TRUST_REMOTE_CODE_ON.match(line)]
    assert not offenders, f"trust_remote_code is enabled by default in: {offenders}"
