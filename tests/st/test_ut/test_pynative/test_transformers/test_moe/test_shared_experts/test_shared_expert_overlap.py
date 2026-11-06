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
"""Unit tests for split shared-expert forward/backward overlap."""

from types import SimpleNamespace

import numpy as np
import pytest
import mindspore as ms

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.layers.linear import Linear
from mindformers.pynative.transformers.mlp import MLPSubmodules
import mindformers.pynative.transformers.moe.shared_experts as shared_experts_mod
from mindformers.pynative.transformers.moe.shared_experts import SharedExpertMLP


def _shared_expert_config(**overrides):
    """Build the minimal shared-expert configuration used by overlap tests."""
    values = {
        "num_layers": 1,
        "num_attention_heads": 1,
        "hidden_size": 4,
        "ffn_hidden_size": 4,
        "moe_ffn_hidden_size": 4,
        "moe_shared_expert_intermediate_size": 4,
        "shared_expert_num": 1,
        "hidden_act": "silu",
        "add_bias_linear": False,
        "compute_dtype": "float32",
        "params_dtype": "float32",
        "moe_shared_expert_overlap": True,
    }
    values.update(overrides)
    return TransformerConfig(**values)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_split_shared_expert_forward_uses_side_stream(monkeypatch):
    """The invocation-local FC stages stay ordered and join without a global sync."""
    events = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, stream):
            events.append((self.name, "wait", stream.name))

    class FakeStreamCtx:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            events.append((self.stream.name, "enter"))

        def __exit__(self, exc_type, exc_value, traceback):
            del exc_type, exc_value, traceback
            events.append((self.stream.name, "exit"))

    main_stream = FakeStream("main")
    side_stream = FakeStream("shared")
    monkeypatch.setattr(shared_experts_mod.ms.runtime, "Stream", lambda: side_stream)
    monkeypatch.setattr(shared_experts_mod.ms.runtime, "StreamCtx", FakeStreamCtx)
    monkeypatch.setattr(shared_experts_mod.ms.runtime, "current_stream", lambda: main_stream)
    monkeypatch.setattr(
        SharedExpertMLP,
        "_forward_fc1_and_act",
        lambda self, hidden_states: hidden_states + 1,
    )
    monkeypatch.setattr(
        SharedExpertMLP,
        "_forward_fc2",
        lambda self, intermediate: intermediate * 2,
    )
    monkeypatch.setattr(SharedExpertMLP, "_overlap_stream", None)

    shared_experts = SharedExpertMLP(
        _shared_expert_config(),
        MLPSubmodules(linear_fc1=Linear, linear_fc2=Linear),
    )
    hidden_states = ms.Tensor([[1.0, 2.0, 3.0, 4.0]], ms.float32)

    ctx = shared_experts.overlap_pre_forward(hidden_states)
    shared_experts.wait_current_stream()
    shared_experts.overlap_fc1(ctx)
    shared_experts.wait_current_stream()
    shared_experts.overlap_fc2(ctx)
    output = shared_experts.overlap_finish(ctx)

    np.testing.assert_array_equal(
        output.asnumpy(),
        np.array([[4.0, 6.0, 8.0, 10.0]], dtype=np.float32),
    )
    assert ctx.state == "finished"
    assert events.count(("shared", "wait", "main")) == 3
    assert events[-1] == ("main", "wait", "shared")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_shared_expert_overlap_config_requires_alltoall():
    """Reject a requested overlap with an unsupported token dispatcher."""
    with pytest.raises(ValueError, match="only supports.*alltoall"):
        _shared_expert_config(moe_token_dispatcher_type="alltoall_deredundancy")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_shared_expert_backward_switches_side_and_main_streams(monkeypatch):
    """Backward stream markers wait only at the output and final input boundaries."""
    events = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            events.append((self.name, "wait_event", event))

        def wait_stream(self, stream):
            events.append((self.name, "wait_stream", stream.name))

    main_stream = FakeStream("main")
    shared_stream = FakeStream("shared")
    ready_event = object()
    overlap_ctx = SimpleNamespace(
        main_stream=main_stream,
        shared_stream=shared_stream,
        backward_grad_ready_event=ready_event,
    )
    monkeypatch.setattr(
        shared_experts_mod.ms.runtime,
        "set_cur_stream",
        lambda stream: events.append(("set", stream.name)),
    )
    grad = ms.Tensor([1.0], ms.float32)

    shared_fn_ctx = SimpleNamespace(
        overlap_ctx=overlap_ctx,
        wait_for_output_grad=True,
    )
    returned = shared_experts_mod._BackwardUseSharedStream.backward(
        shared_fn_ctx, grad)

    main_fn_ctx = SimpleNamespace(
        overlap_ctx=overlap_ctx,
        wait_for_shared=True,
        anchor_count=0,
    )
    restored = shared_experts_mod._BackwardUseMainStream.backward(
        main_fn_ctx, grad)

    assert returned == (grad, None, None)
    assert restored == (grad, None, None)
    assert events == [
        ("shared", "wait_event", ready_event),
        ("set", "shared"),
        ("set", "main"),
        ("main", "wait_stream", "shared"),
    ]
