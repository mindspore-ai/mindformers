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
"""Tests for GPT model parallelization helpers."""

from contextlib import nullcontext
from types import SimpleNamespace
import threading

import pytest
import mindspore as ms

from mindformers.pynative.base_models.gpt import parallelize


def _build_moe_model():
    """Build the minimal model tree consumed by the MoE parallelize helpers."""
    layer = SimpleNamespace(mlp=SimpleNamespace(experts=object()))
    config = SimpleNamespace(moe_permute_fusion=False)
    inner_model = SimpleNamespace(
        config=config,
        decoder=SimpleNamespace(layers=[layer]),
        mtp=None,
    )
    return SimpleNamespace(model=inner_model)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("dispatcher", ["alltoall", "alltoall_deredundancy"])
def test_apply_moe_ep_tp_propagates_use_safe_tokens(monkeypatch, dispatcher):
    """The safe-token setting reaches both PyNative EP dispatchers."""
    captured = {}

    def _capture_strategy(_, **kwargs):
        captured["use_safe_tokens"] = kwargs["use_safe_tokens"]
        return object()

    monkeypatch.setattr(parallelize, "ExpertParallel", _capture_strategy)
    monkeypatch.setattr(parallelize, "DeredundancyExpertParallel", _capture_strategy)
    monkeypatch.setattr(parallelize, "set_comm_ops_inplace", lambda _: None)
    monkeypatch.setattr(parallelize, "parallelize_module", lambda **_: None)

    parallelize.apply_moe_ep_tp(
        _build_moe_model(),
        ep_mesh=object(),
        moe_token_dispatcher_type=dispatcher,
        use_safe_tokens=False,
    )

    assert captured["use_safe_tokens"] is False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_moe_ep_overlap_tp_propagates_use_safe_tokens(monkeypatch):
    """The overlap EP strategy receives the same safe-token config."""
    captured = {}

    def _capture_strategy(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(parallelize, "OverlapExpertParallel", _capture_strategy)
    monkeypatch.setattr(parallelize, "parallelize_module", lambda **_: None)
    monkeypatch.setattr(parallelize.ms, "DeviceCtx", lambda *_: nullcontext())

    parallelize.apply_moe_ep_overlap_tp(
        _build_moe_model(),
        overlap=SimpleNamespace(coordinator=object()),
        ep_mesh=object(),
        use_safe_tokens=False,
    )

    assert captured["use_safe_tokens"] is False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_moe_ep_tp_wires_shared_expert_overlap(monkeypatch):
    """An all-to-all EP strategy owns shared experts only when overlap is requested."""
    shared_experts = object()
    overlap_flags = []
    mlp = SimpleNamespace(
        experts=object(),
        shared_experts=shared_experts,
        config=SimpleNamespace(moe_shared_expert_overlap=True),
        set_shared_expert_overlap=overlap_flags.append,
    )
    model = SimpleNamespace(model=SimpleNamespace(
        config=SimpleNamespace(moe_permute_fusion=False),
        decoder=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]),
        mtp=None,
    ))
    captured = {}

    def _capture_strategy(*args, **kwargs):
        del args
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(parallelize, "ExpertParallel", _capture_strategy)
    monkeypatch.setattr(parallelize, "set_comm_ops_inplace", lambda _: None)
    monkeypatch.setattr(parallelize, "parallelize_module", lambda **_: None)

    parallelize.apply_moe_ep_tp(
        model,
        ep_mesh=object(),
        moe_token_dispatcher_type="alltoall",
    )

    assert captured["shared_experts"] is shared_experts
    assert overlap_flags == [True]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_shared_expert_overlap_rejects_non_alltoall_dispatcher():
    """The optimization cannot be silently enabled on a different dispatcher."""
    moe_layer = SimpleNamespace(
        config=SimpleNamespace(moe_shared_expert_overlap=True),
        shared_experts=object(),
    )

    with pytest.raises(ValueError, match="only supports.*alltoall"):
        parallelize._shared_experts_for_a2a_overlap(
            moe_layer, "alltoall_deredundancy")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_shared_expert_overlap_rejects_pipeline_b_f_overlap():
    """The two independent side-stream overlap protocols must not be composed."""
    shared_experts = object()
    mlp = SimpleNamespace(
        experts=object(),
        shared_experts=shared_experts,
        config=SimpleNamespace(moe_shared_expert_overlap=True),
    )
    model = SimpleNamespace(model=SimpleNamespace(
        config=SimpleNamespace(moe_permute_fusion=False),
        decoder=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]),
        mtp=None,
    ))

    with pytest.raises(ValueError, match="cannot be combined.*pipeline_parallel_overlap_b_f"):
        parallelize.apply_moe_ep_overlap_tp(
            model,
            overlap=SimpleNamespace(coordinator=object()),
            ep_mesh=object(),
        )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tag_dsv4_tp_replicated_grad_norm_params_excludes_sharded_and_indexer_weights():
    """Only already-global full-attention gradients are TP replica-counted."""
    parameters = {
        "linear_proj.weight": SimpleNamespace(),
        "linear_q_up_proj.weight": SimpleNamespace(),
        "core_attention.attn_sink": SimpleNamespace(),
        "core_attention.indexer.weight": SimpleNamespace(),
    }
    unrelated = SimpleNamespace()

    marked_attention = SimpleNamespace(
        _tp_full_attention_replica_count=2,
        parameters_and_names=parameters.items,
    )
    unmarked_cell = SimpleNamespace(
        parameters_and_names=lambda: (("weight", unrelated),)
    )
    model = SimpleNamespace(
        cells_and_names=lambda: (
            ("self_attention", marked_attention),
            ("unrelated", unmarked_cell),
        )
    )
    tag_replicated_params = getattr(
        parallelize, "_tag_dsv4_tp_replicated_grad_norm_params"
    )
    tag_replicated_params(model)

    assert getattr(parameters["linear_proj.weight"], "_grad_norm_replica_count") == 2
    assert getattr(parameters["core_attention.attn_sink"], "_grad_norm_replica_count") == 2
    assert not hasattr(parameters["linear_q_up_proj.weight"], "_grad_norm_replica_count")
    assert not hasattr(parameters["core_attention.indexer.weight"], "_grad_norm_replica_count")
    assert not hasattr(unrelated, "_grad_norm_replica_count")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_parallelize_rejects_reentrant_dxdw_split():
    """Reject the unsupported reentrant + pipeline dxdw-split combination."""
    with pytest.raises(ValueError, match="enable_dxdw_split=True"):
        parallelize.parallelize_gptmodel(
            model=object(),
            parallel_dims=SimpleNamespace(pp_enabled=False),
            parallelism=SimpleNamespace(
                pipeline_parallel_enable_dxdw_split=True),
            recompute=SimpleNamespace(use_reentrant=True),
            recompute_comm=object(),
            swap=object(),
        )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_overlap_b_f_runs_reentrant_replay_lazily_on_backward_worker():
    """Reentrant replay overlaps the paired forward instead of being prefired."""
    fwd_started = threading.Event()
    replay_started = threading.Event()
    records = []

    class ReplayLoss:
        """Model the lazy replay entered from pipeline backward."""

        @staticmethod
        def backward():
            records.append(("replay", threading.current_thread().name))
            replay_started.set()
            if not fwd_started.wait(timeout=5):
                raise RuntimeError("paired forward did not overlap reentrant replay")

    loss = ReplayLoss()

    class Coordinator:
        """Minimal disabled overlap coordinator used by the schedule stub."""

        @staticmethod
        def is_enabled():
            return False

        @staticmethod
        def rendezvous(_):
            return None

        @staticmethod
        def notify_dispatched(_):
            return None

    class Overlap:
        """Run paired forward and backward functions on separate threads."""

        coordinator = Coordinator()

        @staticmethod
        def run(fwd_fn, bwd_fn):
            """Overlap the supplied forward with the backward worker."""
            errors = []

            def _run_backward():
                try:
                    bwd_fn()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    errors.append(exc)

            worker = threading.Thread(
                target=_run_backward, name="hp-overlap-bwd-worker")
            worker.start()
            fwd_fn()
            worker.join(timeout=10)
            if worker.is_alive():
                raise RuntimeError("backward worker did not finish")
            if errors:
                raise errors[0]

    class BackwardStage:
        """Run the loss backward call for the pipeline schedule stub."""

        stage_index = 0

        @staticmethod
        def recompute_one_chunk(_):
            records.append(("prefire", threading.current_thread().name))

        @staticmethod
        def backward_one_chunk(_):
            loss.backward()

    class ForwardStage:
        """Signal when the paired pipeline forward starts."""

        stage_index = 1

        @staticmethod
        def forward_one_chunk(_, args, kwargs):
            del kwargs
            records.append(("paired_forward", threading.current_thread().name))
            fwd_started.set()
            if not replay_started.wait(timeout=5):
                raise RuntimeError("reentrant replay did not start")
            return args[0] * 2

    class Schedule:
        """Provide the pipeline schedule protocol consumed by the callback."""

        _stage_dict = {0: BackwardStage(), 1: ForwardStage()}

        @staticmethod
        def wait_bwd_recv(*_):
            return None

        @staticmethod
        def wait_fwd_recv(*_):
            return None

        @staticmethod
        def update_losses(*_):
            return None

    step = SimpleNamespace(sub_steps=(
        SimpleNamespace(stage_index=0, micro_index=0),
        SimpleNamespace(stage_index=1, micro_index=1),
    ))
    ctx = SimpleNamespace(
        schedule=Schedule(),
        arg_mbs=(None, (ms.Tensor([1.0], ms.float32),)),
        kwarg_mbs=(None, {}),
        losses=[],
    )

    parallelize._make_overlap_b_f_callback(Overlap())(step, ctx)

    assert ("prefire", "MainThread") in records
    assert ("paired_forward", "MainThread") in records
    assert ("replay", "hp-overlap-bwd-worker") in records
