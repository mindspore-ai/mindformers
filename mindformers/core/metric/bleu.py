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
"""
Sentence-level BLEU-4 with NIST geometric sequence smoothing.

Computes the same value as ``nltk.translate.bleu_score.sentence_bleu`` with its default uniform
4-gram weights and ``SmoothingFunction().method3`` (Chen and Cherry, 2014, "A Systematic
Comparison of Smoothing Techniques for Sentence-Level BLEU"), which is what ADGENMetric used, so
the metric no longer needs nltk.
"""
import math
from collections import Counter

BLEU_MAX_ORDER = 4


def _ngram_counts(tokens, n):
    """Count the n-grams of ``tokens``; empty when there are fewer than ``n`` tokens."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _clipped_precision(references, hypothesis, n):
    """
    Modified n-gram precision of ``hypothesis``.

    Returns:
        tuple(int, int), the hypothesis n-grams also found in a reference (each clipped to its
        largest count in any single reference), and the number of hypothesis n-grams (at least 1).
    """
    hyp_counts = _ngram_counts(hypothesis, n)
    max_ref_counts = Counter()
    for reference in references:
        max_ref_counts |= _ngram_counts(reference, n)
    matches = sum(min(count, max_ref_counts[ngram]) for ngram, count in hyp_counts.items())
    return matches, max(1, sum(hyp_counts.values()))


def sentence_bleu(references, hypothesis):
    """
    Sentence-level BLEU-4 of ``hypothesis`` against ``references``.

    Args:
        references (list[Sequence]): The reference token sequences.
        hypothesis (Sequence): The hypothesis token sequence.

    Returns:
        float, the BLEU score in [0, 1]; 0 when no unigram of the hypothesis is in a reference.
    """
    precisions = [_clipped_precision(references, hypothesis, n) for n in range(1, BLEU_MAX_ORDER + 1)]
    if precisions[0][0] == 0:
        return 0

    hyp_len = len(hypothesis)
    # Closest reference length; a tie goes to the shorter reference.
    ref_len = min((len(reference) for reference in references),
                  key=lambda length: (abs(length - hyp_len), length))
    brevity_penalty = 1 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)

    weight = 1 / BLEU_MAX_ORDER
    log_precisions = []
    smoothing_power = 1
    for matches, total in precisions:
        if matches == 0:
            # Orders without any match get 1 / (2^k * total), k counting up from 1.
            precision = 1 / (2 ** smoothing_power * total)
            smoothing_power += 1
        else:
            precision = matches / total
        log_precisions.append(weight * math.log(precision))
    return brevity_penalty * math.exp(math.fsum(log_precisions))
