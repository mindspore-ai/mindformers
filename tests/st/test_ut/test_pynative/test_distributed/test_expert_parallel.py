#!/usr/bin/env python3
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
"""Tests for expert parallel in pynative mode."""
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, context, nn, ops

from hyper_parallel.core.dtensor.placement_types import Shard

from mindformers.pynative.distributed import utils
import mindformers.pynative.distributed.expert_parallel as ep_mod
from mindformers.pynative.distributed.activation_checkpoint import recompute_context_fn
from mindformers.pynative.distributed.ep_overlap import OverlapExpertParallel
from mindformers.pynative.distributed.expert_parallel import ExpertParallel
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.layers.linear import Linear
from mindformers.pynative.transformers.mlp import MLP, MLPSubmodules
from mindformers.pynative.transformers.moe.experts import GroupedMLP

HIDDEN_SIZE = 32
EXPERT_NUM = 4

@pytest.fixture(name="device_mesh")
def fixture_device_mesh():
    """Provide a lightweight fake device mesh."""
    return object()

class TestExpertParallel:
    """Tests for ExpertParallel."""

    def setup_method(self):
        """Set up test fixtures for expert parallel MoE tests."""
        context.set_context(mode=context.PYNATIVE_MODE)

        self.device_mesh = object()
        self.config = TransformerConfig(
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=4,
            num_layers=1,
            hidden_act="fusedswiglu",
            num_moe_experts=EXPERT_NUM,
            add_bias_linear=False,
            # MoE specific configs
            moe_ffn_hidden_size=HIDDEN_SIZE * 4, # Standard MLP expansion
            moe_apply_probs_on_input=False,
            gated_linear_unit=True,
        )
        self.set_expert_parallel()

    def set_expert_parallel(self):
        self.expert_parallel = ExpertParallel()

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_module_valid(self):
        # Pass in valid module GroupedMLP
        module = self.expert_parallel._apply(GroupedMLP(self.config), self.device_mesh)

        assert isinstance(module, GroupedMLP)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_module_invalid(self):
        # Pass in invalid module MLP
        mlp = MLP(submodules=MLPSubmodules(linear_fc1=Linear, linear_fc2=Linear,),
                  config=self.config,
                  input_size=HIDDEN_SIZE)
        with pytest.raises(TypeError):
            self.expert_parallel._apply(mlp, self.device_mesh)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @pytest.mark.parametrize("use_safe_tokens, expected_pad_size", [(True, EXPERT_NUM), (False, 0)])
    def test_add_safe_tokens_respects_config(self, use_safe_tokens, expected_pad_size):
        """Safe tokens are only prepended when use_safe_tokens is enabled."""
        tokens = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=ms.float32)
        probs = Tensor([[0.4, 0.6], [0.7, 0.3]], dtype=ms.float32)
        topk_indices = Tensor([[0, 1], [2, 3]], dtype=ms.int32)
        expert_parallel = ExpertParallel(use_safe_tokens=use_safe_tokens)

        padded_tokens, padded_probs, padded_indices, pad_size = expert_parallel._add_safe_tokens(
            tokens, probs, topk_indices, EXPERT_NUM, topk_indices.shape[-1]
        )

        assert pad_size == expected_pad_size
        assert padded_tokens.shape[0] == tokens.shape[0] + expected_pad_size
        assert padded_probs.shape[0] == probs.shape[0] + expected_pad_size
        assert padded_indices.shape[0] == topk_indices.shape[0] + expected_pad_size

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_sync_d2h_splits_are_reused_during_recompute(self):
        """Replay returns forward host splits without touching device tensors."""
        local_counts = Tensor([2, 1, 3, 2], dtype=ms.float32)
        grouped_counts = Tensor([1, 2, 2, 3], dtype=ms.float32)
        forward_ctx, replay_ctx = recompute_context_fn()

        with forward_ctx:
            expected = self.expert_parallel._host_token_splits(
                local_counts, grouped_counts, ep_degree=2)

        # None inputs prove replay returns before concat/cast/tolist.
        with replay_ctx:
            actual = self.expert_parallel._host_token_splits(
                None, None, ep_degree=2)

        assert actual == expected
        assert actual == ([3, 5], [3, 5], [1, 2, 2, 3])

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_async_d2h_is_issued_only_by_forward(self, monkeypatch):
        """Async replay recomputes counts but skips the D2H launch and event wait."""
        class FakeHostBuffer:
            def __init__(self, values):
                self.values = values
                self.tolist_calls = 0

            def tolist(self):
                self.tolist_calls += 1
                return self.values

        class FakeEvent:
            def __init__(self):
                self.synchronize_calls = 0

            def synchronize(self):
                self.synchronize_calls += 1

        local_counts = Tensor([2, 1, 3, 2], dtype=ms.float32)
        grouped_counts = Tensor([1, 2, 2, 3], dtype=ms.float32)
        host_buf = FakeHostBuffer([2, 1, 3, 2, 1, 2, 2, 3])
        event = FakeEvent()
        issue_calls = []

        monkeypatch.setattr(ep_mod, "get_rank", lambda: 0)
        monkeypatch.setattr(ep_mod, "get_ep_group_name", lambda rank, degree: f"ep-{rank}-{degree}")
        monkeypatch.setattr(
            self.expert_parallel, "_count_tokens_per_expert",
            lambda topk_indices, num_experts: local_counts)
        monkeypatch.setattr(
            self.expert_parallel, "_counts_a2a",
            lambda counts, ep_degree: grouped_counts)

        def fake_issue(counts):
            issue_calls.append(counts)
            return host_buf, event

        monkeypatch.setattr(self.expert_parallel, "_issue_async_d2h", fake_issue)
        forward_ctx, replay_ctx = recompute_context_fn()

        with forward_ctx:
            fwd_host, fwd_event, fwd_grouped = self.expert_parallel._dispatch_preprocess(
                Tensor([[0, 1]], dtype=ms.int32), num_experts=4, ep_degree=2)
            expected = self.expert_parallel._finish_async_d2h(
                fwd_host, fwd_event, num_experts=4, ep_degree=2)

        with replay_ctx:
            replay_host, replay_event, replay_grouped = self.expert_parallel._dispatch_preprocess(
                Tensor([[0, 1]], dtype=ms.int32), num_experts=4, ep_degree=2)
            actual = self.expert_parallel._finish_async_d2h(
                replay_host, replay_event, num_experts=4, ep_degree=2)

        assert len(issue_calls) == 1
        assert event.synchronize_calls == 1
        assert host_buf.tolist_calls == 1
        assert replay_host is None
        assert replay_event is None
        assert actual == expected
        np.testing.assert_array_equal(fwd_grouped.asnumpy(), replay_grouped.asnumpy())

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_fused_resort_matches_chunk_fallback(self, monkeypatch):
        """Fused local-expert resort is identical to the chunk permutation."""
        group_counts = [2, 1, 1, 2]
        grouped_counts = Tensor(group_counts, dtype=ms.float32)
        received = Tensor(np.arange(12).reshape(1, 6, 2), dtype=ms.float32)

        fused = ExpertParallel(moe_permute_fusion=True)
        routing_map = fused._build_resort_routing_map(
            grouped_counts, group_counts, ep_degree=2)
        fused_output, restore_map = fused._resort_after_dispatch(
            received, routing_map=routing_map)

        def fail_unpermute(*args, **kwargs):
            raise AssertionError("local EP restore must not call moe_token_unpermute")

        monkeypatch.setattr(ops, "moe_token_unpermute", fail_unpermute)
        fused_restored = fused._unsort_for_combine(
            fused_output, restore_map, hidden_size=2)

        fallback = ExpertParallel(moe_permute_fusion=False)
        sort_idx, restore_idx = fallback._chunk_perm(2, 2)
        fallback_output, _ = fallback._resort_after_dispatch(
            received, group_counts, sort_idx)
        combine_counts = [group_counts[i] for i in sort_idx]
        fallback_restored = fallback._unsort_for_combine(
            fallback_output, (combine_counts, restore_idx), hidden_size=2)

        np.testing.assert_array_equal(fused_output.asnumpy(), fallback_output.asnumpy())
        np.testing.assert_array_equal(fused_restored.asnumpy(), received.asnumpy())
        np.testing.assert_array_equal(fallback_restored.asnumpy(), received.asnumpy())

        def fused_roundtrip_sum(value):
            output, output_restore_map = fused._resort_after_dispatch(
                value, routing_map=routing_map)
            output = fused._unsort_for_combine(
                output, output_restore_map, hidden_size=2)
            return output.sum()

        grad = ms.grad(fused_roundtrip_sum)(received)
        np.testing.assert_array_equal(
            grad.asnumpy(), np.ones(received.shape, dtype=np.float32))

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_fused_finalize_combine_with_probs(self):
        """Final EP restore fuses unpermute, routing weights and top-k reduction."""
        permuted = Tensor(np.arange(8).reshape(4, 2), dtype=ms.float32)
        restore_map = Tensor([[2, 0], [3, 1]], dtype=ms.int32)
        probs = Tensor([[0.25, 0.75], [0.6, 0.4]], dtype=ms.float32)
        restore_index = restore_map.asnumpy().reshape(-1)
        restored = permuted.asnumpy()[restore_index]
        probs_np = probs.asnumpy()
        expected = (restored.reshape(2, 2, 2) * probs_np[..., None]).sum(axis=1)
        expected_input_grad = np.zeros_like(permuted.asnumpy())
        expected_input_grad[restore_index] = probs_np.reshape(-1, 1)
        expected_probs_grad = restored.sum(axis=-1).reshape(2, 2)

        expert_parallel = ExpertParallel(moe_permute_fusion=True)
        actual = expert_parallel._finalize_combine(
            permuted, restore_map, probs, pad_size=0)
        np.testing.assert_array_equal(actual.asnumpy(), expected)

        def loss(value, weights):
            return expert_parallel._finalize_combine(
                value, restore_map, weights, pad_size=0).sum()

        input_grad, probs_grad = ms.grad(loss, grad_position=(0, 1))(
            permuted, probs)
        np.testing.assert_array_equal(input_grad.asnumpy(), expected_input_grad)
        np.testing.assert_array_equal(probs_grad.asnumpy(), expected_probs_grad)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @pytest.mark.parametrize("async_d2h", [False, True])
    def test_overlap_builds_routing_map_during_main_a2a(self, monkeypatch, async_d2h):
        """Launch the async token a2a before building its local resort map."""
        expert_parallel = OverlapExpertParallel(
            coordinator=object(), moe_permute_fusion=True)
        calls = []

        monkeypatch.setattr(
            expert_parallel, "_sync_hook",
            lambda value, name: calls.append(name) or value)
        monkeypatch.setattr(
            expert_parallel, "_compute_group_list",
            lambda grouped_counts, ep_degree: calls.append("group_list") or "group_list")
        monkeypatch.setattr(
            expert_parallel, "_main_a2a",
            lambda flat_in, input_splits, output_splits, block_size:
            calls.append("main_a2a") or "flat_out")
        monkeypatch.setattr(
            expert_parallel, "_build_resort_routing_map",
            lambda grouped_counts, group_counts, ep_degree:
            calls.append("routing_map") or "routing_map")

        if async_d2h:
            class FakeEvent:
                """Record the deferred D2H synchronization."""

                @staticmethod
                def synchronize():
                    calls.append("d2h_sync")

            class FakeHostBuffer:
                """Provide the host counts consumed after synchronization."""

                @staticmethod
                def tolist():
                    calls.append("host_tolist")
                    return [1, 1]

            monkeypatch.setattr(
                expert_parallel, "_derive_splits",
                lambda both_host, num_experts, ep_degree:
                calls.append("host_splits") or ([1], [1], [1]))
            result = expert_parallel._dispatch_a2a(
                "flat_in", FakeHostBuffer(), FakeEvent(), "grouped_counts",
                num_experts=1, ep_degree=1, block_size=4)
        else:
            monkeypatch.setattr(
                expert_parallel, "_counts_a2a",
                lambda counts, ep_degree: calls.append("counts_a2a") or "grouped_counts")
            monkeypatch.setattr(
                expert_parallel, "_host_token_splits",
                lambda counts, grouped_counts, ep_degree:
                calls.append("host_splits") or ([1], [1], [1]))
            result = expert_parallel._dispatch_comm(
                "flat_in", "counts", ep_degree=1, block_size=4)

        assert calls.index("main_a2a") < calls.index("routing_map") < calls.index("B")
        assert result[-1] == "routing_map"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_parameter_sharding_plan(self, monkeypatch):
        """Test that expert parallel applies the expected parameter sharding plan."""
        captured = {}
        def _fake_shard_module(module, device_mesh, parameter_shard_plan):
            captured["module"] = module
            captured["device_mesh"] = device_mesh
            captured["sharding_plan"] = parameter_shard_plan
            # No actual partitioning is performed to avoid introducing communication/distributed dependencies.
            return module

        monkeypatch.setattr(utils, "shard_module", _fake_shard_module)

        module = self.expert_parallel._apply(GroupedMLP(self.config), self.device_mesh)# pylint: disable=unused-variable
        sharding_plan = captured["sharding_plan"]
        plan = sharding_plan.plan

        # expect：weight1/weight2 is Shard(0)
        assert set(plan.keys()) == {"weight1", "weight2"}
        assert plan["weight1"][0] == Shard(0)
        assert plan["weight2"][0] == Shard(0)
