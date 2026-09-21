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
"""UTs for the built-in sentence-level BLEU-4 used by ADGENMetric."""
import importlib.util
import os
import random
import subprocess
import sys

import pytest

from mindformers.core.metric.bleu import sentence_bleu

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Expected values computed with nltk 3.10.3:
# sentence_bleu(references, hypothesis, smoothing_function=SmoothingFunction().method3)
NLTK_CASES = [
    ("identical, long", ["the cat sat on the mat"], "the cat sat on the mat", 1.0),
    ("identical, 2 tokens (orders 3-4 smoothed)", ["你好"], "你好", 0.5946035575013605),
    ("partial overlap, shorter hypothesis", ["今天天气不错，适合出门散步。"], "今天天气很好，适合出去散步。",
     0.43748114312246444),
    ("hypothesis longer than reference", ["模型训练"], "模型训练已经完成了", 0.2984745896009823),
    ("no common unigram", ["毫无关系"], "完全不同的句子", 0),
    ("empty hypothesis", ["abc"], "", 0),
    ("repeated tokens are clipped", ["ab"], "aaaa", 0.15973577606156814),
    ("closest reference length, tie to shorter", ["abcd", "abcdefgh"], "abcdef", 1.0),
    ("two references, counts clipped per reference", ["aab", "abb"], "aabb", 0.8408964152537145),
]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("name, references, hypothesis, expected", NLTK_CASES, ids=[c[0] for c in NLTK_CASES])
def test_sentence_bleu_matches_nltk_values(name, references, hypothesis, expected):
    """
    Feature: mindformers.core.metric.bleu.sentence_bleu
    Description: Score character sequences covering smoothing, brevity penalty, clipping and multiple references.
    Expectation: The same value nltk's sentence_bleu gives with SmoothingFunction().method3.
    """
    del name
    score = sentence_bleu([list(ref) for ref in references], list(hypothesis))
    assert score == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.skipif(importlib.util.find_spec("nltk") is None, reason="needs nltk as the reference implementation")
def test_sentence_bleu_matches_nltk_on_random_inputs():
    """
    Feature: mindformers.core.metric.bleu.sentence_bleu
    Description: Compare against nltk on random token sequences of varied lengths, alphabets and reference counts.
    Expectation: Identical scores.
    """
    # pylint: disable=import-outside-toplevel
    from nltk.translate.bleu_score import SmoothingFunction
    from nltk.translate.bleu_score import sentence_bleu as nltk_sentence_bleu

    smoothing = SmoothingFunction().method3
    rng = random.Random(0)
    for _ in range(2000):
        alphabet = rng.choice(["ab", "abcd", "abcdefgh", "今天天气很好适合出门散步模型训练完成"])
        max_len = rng.choice([3, 8, 30])
        references = [[rng.choice(alphabet) for _ in range(rng.randint(1, max_len))]
                      for _ in range(rng.choice([1, 2, 3]))]
        hypothesis = [rng.choice(alphabet) for _ in range(rng.randint(0, max_len))]
        expected = nltk_sentence_bleu(references, hypothesis, smoothing_function=smoothing)
        assert sentence_bleu(references, hypothesis) == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_adgen_metric_scores_bleu_without_nltk():
    """
    Feature: ADGENMetric
    Description: Score one prediction in a fresh interpreter where importing nltk fails.
    Expectation: bleu-4 is computed without nltk and equals the nltk value.
    """
    code = (
        "import sys\n"
        "sys.modules['nltk'] = None\n"
        "from mindformers.core.metric.metric import ADGENMetric\n"
        "metric = ADGENMetric()\n"
        "metric.update(['今天天气很好，适合出去散步。'], ['今天天气不错，适合出门散步。'])\n"
        "print('bleu-4', metric.eval()['bleu-4'])\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")])))
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                            timeout=600, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    # round(0.43748114312246444 * 100, 4), the value nltk gives for this pair
    assert "bleu-4 43.7481" in result.stdout
