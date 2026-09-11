# Copyright 2026 Huawei Technologies Co, Ltd
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
"""Test PyNative inference with various parallel strategies on 8 cards."""

import pytest

from tests.st.test_multi_cards_cases.utils import TaskType
from tests.st.test_multi_cards_cases.test_pynative.test_inference.utils import (
    assert_parallel_matches_reference,
    run_inference_case,
)

_LEVEL_0_TASK_TIME = 0
_LEVEL_1_TASK_TIME = 320
_TASK_TYPE = TaskType.EIGHT_CARDS_TASK


@pytest.mark.level1
def test_tp8_sp_inference():
    """
    Feature: PyNative inference with tensor parallel + sequence parallel
    Description: Run PyNative inference with TP=8 (+SP) on 8 cards using the
                 seed-deterministic miniature model; vocab-parallel logits are
                 gathered before the greedy argmax.
    Expectation: Every generated token id sequence matches the single-card
                 reference run token by token.
    """
    multi_ids, reference_ids, _ = run_inference_case(
        "infer_tp8_sp.yaml",
        "log_infer_tp8_sp",
        worker_num=8,
        updates={
            "parallelism": {
                "tensor_parallel": 8,
                "sequence_parallel": True,
            },
        },
    )
    assert_parallel_matches_reference(multi_ids, reference_ids)


@pytest.mark.level1
def test_cp2_dp4_inference():
    """
    Feature: PyNative inference with context parallel
    Description: Run PyNative inference with CP=2 x DP=4 on 8 cards using the
                 seed-deterministic miniature model; sequence-sharded logits
                 go through the CP row selection and gather before argmax.
    Expectation: Every generated token id sequence matches the single-card
                 reference run token by token.
    """
    multi_ids, reference_ids, _ = run_inference_case(
        "infer_cp2_dp4.yaml",
        "log_infer_cp2_dp4",
        worker_num=8,
        updates={
            "parallelism": {"context_parallel": 2},
            "model": {
                "apply_rope_fusion": False,
                "use_attn_mask_compression": True,
            },
        },
    )
    assert_parallel_matches_reference(multi_ids, reference_ids)


@pytest.mark.level1
def test_ep4_inference():
    """
    Feature: PyNative inference with expert parallel
    Description: Run PyNative inference with EP=4 on 8 cards using the
                 seed-deterministic miniature model; the 8 routed experts are
                 sharded 2-per-rank and MoE dispatch/gather must stay numerically
                 identical to the single-card reference.
    Expectation: Every generated token id sequence matches the single-card
                 reference run token by token.
    """
    multi_ids, reference_ids, _ = run_inference_case(
        "infer_ep4.yaml",
        "log_infer_ep4",
        worker_num=8,
        updates={
            "parallelism": {"expert_parallel": 4},
        },
    )
    assert_parallel_matches_reference(multi_ids, reference_ids)


@pytest.mark.level1
def test_pp2_fsdp4_inference():
    """
    Feature: PyNative inference with pipeline parallel + FSDP
    Description: Run PyNative inference with PP=2 x FSDP4 on 8 cards using the
                 seed-deterministic miniature model; forward-only pipeline stages
                 exchange activations and the sampled token is broadcast back
                 within each pipeline group.
    Expectation: Every generated token id sequence matches the single-card
                 reference run token by token.
    """
    multi_ids, reference_ids, _ = run_inference_case(
        "infer_pp2_fsdp4.yaml",
        "log_infer_pp2_fsdp4",
        worker_num=8,
        updates={
            "parallelism": {
                "data_parallel_shard": 4,
                "pipeline_parallel": 2,
            },
        },
    )
    assert_parallel_matches_reference(multi_ids, reference_ids)


@pytest.mark.level1
def test_tp2_pp2_inference():
    """
    Feature: PyNative inference with tensor + pipeline parallel
    Description: Run PyNative inference with TP=2(+SP) x PP=2 on 8 cards using
                 the seed-deterministic miniature model; vocab-parallel logits
                 gather and the pipeline token broadcast coexist on the last stage.
    Expectation: Every generated token id sequence matches the single-card
                 reference run token by token.
    """
    multi_ids, reference_ids, _ = run_inference_case(
        "infer_tp2_pp2.yaml",
        "log_infer_tp2_pp2",
        worker_num=8,
        updates={
            "parallelism": {
                "data_parallel_shard": 2,
                "tensor_parallel": 2,
                "sequence_parallel": True,
                "pipeline_parallel": 2,
            },
        },
    )
    assert_parallel_matches_reference(multi_ids, reference_ids)
