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
"""Test recompute functional scenarios."""
import importlib

import numpy as np
import pytest
import mindspore as ms
from mindspore import mint, nn, ops
from hyper_parallel.platform.mindspore.activation_checkpoint import (
    CheckpointExcludeWrapper,
    CheckpointWrapper,
    SwapWrapper,
)
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from hyper_parallel.core.activation_checkpoint import CheckpointPolicy

from mindformers.pynative.config.config import (
    RecomputeCommConfig,
    RecomputeConfig,
    SwapConfig,
)
import mindformers.pynative.distributed.activation_checkpoint as ac_mod
from mindformers.pynative.distributed import checkpoint_backend as backend_mod
from mindformers.pynative.distributed.activation_checkpoint import (
    apply_ac,
    apply_recompute,
    apply_swap,
    get_recompute_metadata,
    is_in_recompute,
    recompute_context_fn,
    save_for_recompute,
)


def _reentrant_api():
    """Load the optional HyperParallel reentrant API only for its tests."""
    return importlib.import_module(
        "hyper_parallel.platform.mindspore.activation_checkpoint.reentrant_checkpoint"
    )


@pytest.fixture(autouse=True)
def _reset_config_list():
    """Isolate the module-global whitelist cache between tests."""
    ac_mod._config_list = {}
    yield
    ac_mod._config_list = {}


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestCheckpointBackendIsolation:
    """Checkpoint implementations are loaded and selected independently."""

    def test_non_reentrant_backend_does_not_import_reentrant_api(
            self, monkeypatch):
        """Non-reentrant construction must not load the optional API."""
        real_import_module = importlib.import_module

        def reject_reentrant_import(module_name):
            if module_name == backend_mod.REENTRANT_CHECKPOINT_MODULE:
                raise AssertionError("non-reentrant backend loaded reentrant API")
            return real_import_module(module_name)

        monkeypatch.setattr(
            backend_mod.importlib, "import_module", reject_reentrant_import)

        backend = ac_mod._create_checkpoint_backend(False)

        assert backend.name == "non-reentrant"

    def test_reentrant_backend_reports_missing_optional_api(self, monkeypatch):
        """Reentrant construction reports an actionable missing dependency."""
        real_import_module = importlib.import_module

        def missing_reentrant_api(module_name):
            if module_name == backend_mod.REENTRANT_CHECKPOINT_MODULE:
                raise ModuleNotFoundError(module_name)
            return real_import_module(module_name)

        monkeypatch.setattr(
            backend_mod.importlib, "import_module", missing_reentrant_api)

        with pytest.raises(ImportError, match="MR 1291 or later"):
            ac_mod._create_checkpoint_backend(True)


class MockAttention(nn.Cell):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Cell()
        self.proj = nn.Cell()

    def construct(self, x):
        return x


class MockMLP(nn.Cell):
    def __init__(self):
        super().__init__()
        self.gate = nn.Cell()
        self.proj = nn.Cell()

    def construct(self, x):
        return x


class MockTransformerLayer(nn.Cell):
    def __init__(self):
        super().__init__()
        self.attention = MockAttention()
        self.mlp = MockMLP()

    def construct(self, x):
        return x


class MockModel(nn.Cell):
    """Transformer-block mock with a configurable number of layers."""

    def __init__(self, num_layers=2):
        super().__init__()
        self.layers = nn.CellList([MockTransformerLayer() for _ in range(num_layers)])
        self.config = type("Config", (), {"num_layers": num_layers})()
        self.layer_start = 0
        self.layer_end = num_layers - 1

    # pylint: disable=unused-argument
    def construct(self, hidden_states, attention_mask=None, rotary_pos_emb=None,
                  prefix_keys_values=None, actual_seq_len=None, input_ids=None,
                  mscale=1.0, rotary_cos_sin=None):
        return hidden_states
    # pylint: enable=unused-argument


class ReorderedMockModel(MockModel):
    """Model whose hidden-state argument is not the first positional input."""

    def construct(self, attention_mask, hidden_states, input_ids=None):  # pylint: disable=arguments-differ
        return hidden_states


class MockMtpLayer(nn.Cell):
    """Mirrors MultiTokenPredictionLayer: the heavy transformer is nested under
    ``transformer_layer`` (alongside enorm/hnorm/eh_proj/final_layernorm)."""

    def __init__(self):
        super().__init__()
        self.enorm = nn.Cell()
        self.hnorm = nn.Cell()
        self.eh_proj = nn.Cell()
        self.transformer_layer = MockTransformerLayer()
        self.final_layernorm = nn.Cell()

    def construct(self, x):
        return x


class MockMtpBlock(nn.Cell):
    """Mirrors MultiTokenPredictionBlock: a plain (local-indexed) CellList."""

    def __init__(self, mtp_num_layers=1):
        super().__init__()
        self.layers = nn.CellList([MockMtpLayer() for _ in range(mtp_num_layers)])

    # pylint: disable=unused-argument
    def construct(self, input_ids, position_ids, hidden_states, attention_mask,
                  rotary_pos_emb=None, extra_block_kwargs=None, embedding=None,
                  actual_seq_len=None, mscale=1.0):
        return hidden_states
    # pylint: enable=unused-argument


class MockMtpDecoder(MockModel):
    """Decoder mock whose config also advertises ``mtp_num_layers``."""

    def __init__(self, num_layers=2, mtp_num_layers=1):
        super().__init__(num_layers=num_layers)
        self.config = type("Config", (), {"num_layers": num_layers,
                                           "mtp_num_layers": mtp_num_layers})()


def _make_recompute_config(mode="None", full_recompute_layer=None,
                 select_module=None, comm_enable=False, comm_select_module=None, exclude_op=None,
                 use_reentrant=False):
    """Build recompute and recompute_comm configs."""
    rc = RecomputeConfig(mode=mode, full_recompute_layer=full_recompute_layer,
                         select_module=select_module, exclude_op=exclude_op,
                         use_reentrant=use_reentrant)
    rc_comm = RecomputeCommConfig(enable=comm_enable, select_module=comm_select_module)
    return rc, rc_comm


# ======================== Recompute ========================


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestFullRecompute:
    """Test full recompute: entire layers are wrapped with CheckpointWrapper."""

    def test_full_recompute_single_layer(self):
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(mode="full", full_recompute_layer=["0"])
        apply_recompute(model, rc, rc_comm)
        assert isinstance(model.layers[0], CheckpointWrapper)
        assert not isinstance(model.layers[1], CheckpointWrapper)

    def test_full_recompute_all_layers(self):
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(mode="full", full_recompute_layer=["0-1"])
        apply_recompute(model, rc, rc_comm)
        assert isinstance(model.layers[0], CheckpointWrapper)
        assert isinstance(model.layers[1], CheckpointWrapper)

    def test_full_recompute_single_layer_reentrant(self):
        reentrant_api = _reentrant_api()
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(
            mode="full", full_recompute_layer=["0"], use_reentrant=True)
        apply_recompute(model, rc, rc_comm)
        assert isinstance(
            model.layers[0], reentrant_api.ReentrantCheckpointWrapper)
        assert not isinstance(
            model.layers[1], reentrant_api.ReentrantCheckpointWrapper)

    def test_reentrant_exclude_only_wraps_active_checkpoint_boundaries(self):
        """Do not install a reentrant exclude outside a checkpoint boundary."""
        reentrant_api = _reentrant_api()
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(
            mode="full",
            full_recompute_layer=["0"],
            exclude_op={"attention": ["0-1"]},
            use_reentrant=True,
        )

        apply_recompute(model, rc, rc_comm)

        assert isinstance(
            model.layers[0], reentrant_api.ReentrantCheckpointWrapper)
        assert isinstance(
            model.layers[0].attention,
            reentrant_api.ReentrantCheckpointExcludeWrapper,
        )
        assert isinstance(model.layers[1].attention, MockAttention)

    def test_reentrant_exclude_routes_callable_and_comm_registry(self):
        """Callable attributes and registered comm ops use the same routing."""
        reentrant_api = _reentrant_api()
        backend = ac_mod._create_checkpoint_backend(True)

        class CallableLayer(nn.Cell):
            def __init__(self):
                super().__init__()
                self.operator = lambda value: value * 2
                self._comm_ops = {"input.allgather": {"fn": lambda value: value}}

            def construct(self, value):
                return self.operator(value)

        layer = CallableLayer()
        ac_mod._set_pattern_exclude(layer, ["operator"], backend)
        ac_mod._set_pattern_exclude(
            layer, ["input", "allgather"], backend)

        assert isinstance(
            layer.operator, reentrant_api.ReentrantCheckpointExcludeWrapper)
        assert isinstance(
            layer._comm_ops["input.allgather"]["fn"],
            reentrant_api.ReentrantCheckpointExcludeWrapper,
        )

    def test_reentrant_exclude_preserves_gradients_and_skips_replay(self):
        """The excluded child runs once while the surrounding layer replays."""
        enable_mindspore_backward_compat()

        class CountingMiddle(nn.Cell):
            def __init__(self):
                super().__init__()
                self.weight = ms.Parameter(
                    ms.Tensor([3.0], ms.float32), name="weight")
                self.calls = 0

            def construct(self, value):
                self.calls += 1
                return value * self.weight

        class CountingLayer(nn.Cell):
            def __init__(self):
                super().__init__()
                self.middle = CountingMiddle()
                self.calls = 0

            def construct(self, value):
                self.calls += 1
                hidden = value * value
                return self.middle(hidden) * hidden

        class CountingModel(nn.Cell):
            def __init__(self):
                super().__init__()
                self.layers = nn.CellList([CountingLayer()])
                self.config = type("Config", (), {"num_layers": 1})()
                self.layer_start = 0
                self.layer_end = 0

        reference = CountingModel()
        actual = CountingModel()
        actual_layer = actual.layers[0]
        rc, rc_comm = _make_recompute_config(
            mode="full",
            full_recompute_layer=["0"],
            exclude_op={"middle": ["0"]},
            use_reentrant=True,
        )
        apply_recompute(actual, rc, rc_comm)

        reference_input = ms.Tensor([2.0], ms.float32)
        actual_input = ms.Tensor([2.0], ms.float32)
        reference_input.requires_grad = True
        actual_input.requires_grad = True
        reference_loss = reference.layers[0](reference_input).sum()
        actual_loss = actual.layers[0](actual_input).sum()
        reference_loss.backward()
        actual_loss.backward()

        np.testing.assert_allclose(actual_loss.asnumpy(), reference_loss.asnumpy())
        np.testing.assert_allclose(
            actual_input.grad.asnumpy(), reference_input.grad.asnumpy())
        np.testing.assert_allclose(
            actual_layer.middle.weight.grad.asnumpy(),
            reference.layers[0].middle.weight.grad.asnumpy(),
        )
        assert actual_layer.calls == 2
        assert actual_layer.middle.calls == 1

    def test_reentrant_exclude_accumulates_multiple_microbatch_gradients(self):
        """Each backward replays independently and accumulates parameter grads."""
        enable_mindspore_backward_compat()
        backend = ac_mod._create_checkpoint_backend(True)

        class CountingMiddle(nn.Cell):
            def __init__(self):
                super().__init__()
                self.weight = ms.Parameter(
                    ms.Tensor([3.0], ms.float32), name="weight")
                self.calls = 0

            def construct(self, value):
                self.calls += 1
                return value * self.weight

        class CountingLayer(nn.Cell):
            def __init__(self):
                super().__init__()
                self.middle = backend.wrap_exclude(CountingMiddle())
                self.calls = 0

            def construct(self, value):
                self.calls += 1
                hidden = value * value
                return self.middle(hidden) * hidden

        layer = CountingLayer()
        wrapped = backend.wrap_cell(layer)
        input_grads = []
        for value in (2.0, 3.0):
            microbatch = ms.Tensor([value], ms.float32)
            microbatch.requires_grad = True
            wrapped(microbatch).sum().backward()
            input_grads.append(microbatch.grad.asnumpy().item())

        assert input_grads == [96.0, 324.0]
        assert layer.middle.weight.grad.asnumpy().tolist() == [97.0]
        assert layer.calls == 4
        assert layer.middle.calls == 2

    def test_reentrant_recompute_gradients_and_metadata(self):
        """Custom backward replays once and preserves input/weight gradients."""
        enable_mindspore_backward_compat()

        class MetadataParamCell(nn.Cell):
            """Count original and replay calls while exercising metadata."""

            def __init__(self):
                super().__init__()
                self.weight = ms.Parameter(ms.Tensor([3.0], ms.float32), name="weight")
                self.forward_calls = 0
                self.replay_calls = 0
                self.key = ("reentrant_recompute", id(self))

            def construct(self, x):
                cached = get_recompute_metadata(self.key)
                if cached is None:
                    self.forward_calls += 1
                    save_for_recompute(self.key, "forward-value")
                else:
                    assert cached == "forward-value"
                    self.replay_calls += 1
                return x * x * self.weight, None

        cell = MetadataParamCell()
        wrapped = ac_mod._create_checkpoint_backend(True).wrap_cell(cell)
        x = ms.Tensor([2.0], ms.float32)
        x.requires_grad = True

        output = wrapped(x)[0].sum()
        output.backward()

        assert output.asnumpy().item() == 12.0
        assert x.grad.asnumpy().tolist() == [12.0]
        assert cell.weight.grad.asnumpy().tolist() == [4.0]
        assert cell.forward_calls == 1
        assert cell.replay_calls == 1

    def test_reentrant_recompute_restores_rng(self):
        """Replay must use the same random values without perturbing the caller RNG."""
        enable_mindspore_backward_compat()

        class RandomCell(nn.Cell):
            """Capture random values produced by original and replay forwards."""

            def __init__(self):
                super().__init__()
                self.forward_random = None
                self.replay_random = None

            def construct(self, x):
                random_value = mint.rand_like(x)
                if is_in_recompute():
                    self.replay_random = random_value
                else:
                    self.forward_random = random_value
                return x * random_value

        ms.set_seed(42)
        cell = RandomCell()
        wrapped = ac_mod._create_checkpoint_backend(True).wrap_cell(cell)
        x = ms.Tensor([1.0, 1.0, 1.0, 1.0], ms.float32)
        x.requires_grad = True

        wrapped(x).sum().backward()

        assert cell.forward_random is not None
        assert cell.replay_random is not None
        assert np.array_equal(cell.forward_random.asnumpy(), cell.replay_random.asnumpy())
        assert np.array_equal(x.grad.asnumpy(), cell.forward_random.asnumpy())

    def test_reentrant_recompute_supports_pipeline_prefire(self):
        """A collected replay is recorded before backward and reused there."""
        reentrant_api = _reentrant_api()
        from hyper_parallel.platform.mindspore.pipeline_parallel.backward import (  # pylint: disable=C0415
            forward_and_gradfn,
        )
        enable_mindspore_backward_compat()

        class CountingCell(nn.Cell):
            def __init__(self):
                super().__init__()
                self.weight = ms.Parameter(ms.Tensor([3.0], ms.float32), name="weight")
                self.calls = 0

            def construct(self, x):
                self.calls += 1
                return x * self.weight

        cell = CountingCell()
        wrapped = ac_mod._create_checkpoint_backend(True).wrap_cell(cell)
        x = ms.Tensor([2.0], ms.float32)

        with reentrant_api.reentrant_recompute_handle_collector_ctx() as handles:
            output, grad_fn = forward_and_gradfn(
                lambda value: wrapped(value).sum(),
                x,
                weights=tuple(wrapped.trainable_params()),
                grad_position=-1,
            )
        assert output.asnumpy().item() == 6.0
        assert cell.calls == 1
        assert len(handles) == 1

        session_id = (0, 0)
        handles[0].recompute(session_id)
        assert cell.calls == 2
        with reentrant_api.reentrant_recompute_session_ctx(session_id=session_id):
            input_grads, weight_grads = grad_fn(sens=ms.Tensor(1.0, ms.float32))
        reentrant_api.clear_reentrant_recompute_session(session_id)

        assert input_grads[0].asnumpy().tolist() == [3.0]
        assert weight_grads == (None,)
        assert cell.weight.grad.asnumpy().tolist() == [2.0]
        assert cell.calls == 2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestRecomputeInvocationMetadata:
    """Test checkpoint-local Python metadata shared by forward and replay."""

    def test_forward_values_are_replayed_in_call_order(self):
        """Replay forward metadata in FIFO call order."""
        forward_ctx, replay_ctx = recompute_context_fn()
        key = ("ep_splits", 0)

        with forward_ctx:
            assert not is_in_recompute()
            assert save_for_recompute(key, ("first",))
            assert save_for_recompute(key, ("second",))
            assert get_recompute_metadata(key) is None

        with replay_ctx:
            assert is_in_recompute()
            assert get_recompute_metadata(key) == ("first",)
            assert get_recompute_metadata(key) == ("second",)

        assert not is_in_recompute()
        assert get_recompute_metadata(key) is None

    def test_checkpoint_invocations_keep_isolated_metadata(self):
        """Keep metadata isolated between checkpoint invocations."""
        first_forward, first_replay = recompute_context_fn()
        second_forward, second_replay = recompute_context_fn()
        key = ("ep_splits", 0)

        with first_forward:
            save_for_recompute(key, "first")
        with second_forward:
            save_for_recompute(key, "second")

        with second_replay:
            assert get_recompute_metadata(key) == "second"
        with first_replay:
            assert get_recompute_metadata(key) == "first"

    def test_missing_replay_metadata_raises(self):
        _, replay_ctx = recompute_context_fn()
        with replay_ctx:
            with pytest.raises(RuntimeError, match="No forward metadata"):
                get_recompute_metadata(("missing", 0))

    def test_checkpoint_wrapper_shares_metadata_with_actual_replay(self):
        """MindSpore checkpoint enters the paired contexts around forward/replay."""
        class MetadataCell(nn.Cell):
            """Cell that saves metadata in forward and consumes it during replay."""

            def __init__(self):
                super().__init__()
                self.host_calls = 0
                self.replay_calls = 0
                self.key = ("metadata_cell", id(self))

            def construct(self, x):
                cached = get_recompute_metadata(self.key)
                if cached is None:
                    self.host_calls += 1
                    save_for_recompute(self.key, "forward-value")
                else:
                    assert cached == "forward-value"
                    self.replay_calls += 1
                hidden = x * x
                return hidden * hidden

        cell = MetadataCell()
        wrapped = ac_mod._create_checkpoint_backend(False).wrap_cell(cell)
        x = ops.ones((1,), ms.float32) * 2

        grad = ms.grad(lambda value: wrapped(value).sum())(x)

        assert grad.asnumpy().tolist() == [32.0]
        assert cell.host_calls == 1
        assert cell.replay_calls == 1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestSelectRecompute:
    """Test select recompute: specific modules within layers are wrapped."""

    def test_select_by_wildcard(self):
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(mode="select", select_module={".*\\.proj": ["0"]})
        apply_recompute(model, rc, rc_comm)
        assert isinstance(model.layers[0].attention.proj, CheckpointWrapper)
        assert isinstance(model.layers[0].mlp.proj, CheckpointWrapper)

    def test_select_parent_covers_children(self):
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(mode="select", select_module={"attention": ["0"], "attention.qkv": ["0"]})
        apply_recompute(model, rc, rc_comm)
        assert isinstance(model.layers[0].attention, CheckpointWrapper)
        assert not isinstance(model.layers[0].attention.qkv, CheckpointWrapper)

    def test_select_multiple_modules(self):
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(mode="select", select_module={"attention": ["0"], "mlp": ["1"]})
        apply_recompute(model, rc, rc_comm)
        assert isinstance(model.layers[0].attention, CheckpointWrapper)
        assert isinstance(model.layers[1].mlp, CheckpointWrapper)

    def test_select_recompute_reentrant(self):
        """Selected child Cells use independent reentrant boundaries."""
        reentrant_api = _reentrant_api()
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(
            mode="select",
            select_module={"attention": ["0"], "mlp": ["1"]},
            use_reentrant=True,
        )

        apply_recompute(model, rc, rc_comm)

        assert isinstance(
            model.layers[0].attention,
            reentrant_api.ReentrantCheckpointWrapper,
        )
        assert isinstance(
            model.layers[1].mlp,
            reentrant_api.ReentrantCheckpointWrapper,
        )

    def test_select_callable_reentrant_preserves_gradient(self):
        """Instance-owned callable targets also use reentrant replay."""
        reentrant_api = _reentrant_api()
        enable_mindspore_backward_compat()

        class CountingCallable:
            def __init__(self):
                self.calls = 0

            def __call__(self, value):
                self.calls += 1
                return value * value

        class CallableLayer(nn.Cell):
            def __init__(self):
                super().__init__()
                self.operator = CountingCallable()

            def construct(self, value):
                return self.operator(value)

        model = MockModel(num_layers=1)
        model.layers[0] = CallableLayer()
        operator = model.layers[0].operator
        rc, rc_comm = _make_recompute_config(
            mode="select",
            select_module={"operator": ["0"]},
            use_reentrant=True,
        )

        apply_recompute(model, rc, rc_comm)

        assert isinstance(
            model.layers[0].operator,
            reentrant_api.ReentrantCheckpointWrapper,
        )
        value = ms.Tensor([3.0], ms.float32)
        value.requires_grad = True
        model.layers[0](value).sum().backward()
        assert operator.calls == 2
        assert value.grad.asnumpy().tolist() == [6.0]

    def test_mixed_full_select_reentrant_keeps_router_out_of_replay(self):
        """Full and selected activation replay while every router executes once."""
        reentrant_api = _reentrant_api()
        enable_mindspore_backward_compat()

        class CountingOp(nn.Cell):
            def __init__(self, factor):
                super().__init__()
                self.factor = factor
                self.calls = 0

            def construct(self, value):
                self.calls += 1
                return value * self.factor

        class CountingLayer(nn.Cell):
            def __init__(self):
                super().__init__()
                self.router = CountingOp(2.0)
                self.activation = CountingOp(3.0)
                self.calls = 0

            def construct(self, value):
                self.calls += 1
                return self.activation(self.router(value))

        class CountingModel(nn.Cell):
            def __init__(self):
                super().__init__()
                self.layers = nn.CellList([CountingLayer(), CountingLayer()])
                self.config = type("Config", (), {"num_layers": 2})()
                self.layer_start = 0
                self.layer_end = 1

        model = CountingModel()
        original_layers = tuple(model.layers)
        rc, rc_comm = _make_recompute_config(
            mode="select",
            full_recompute_layer=["0"],
            select_module={"activation": ["0-1"]},
            exclude_op={"router": ["0-1"]},
            use_reentrant=True,
        )

        apply_ac(model, rc, rc_comm, SwapConfig(enable=False), 1)

        assert isinstance(
            model.layers[0], reentrant_api.ReentrantCheckpointWrapper)
        assert isinstance(
            original_layers[0].router,
            reentrant_api.ReentrantCheckpointExcludeWrapper,
        )
        assert isinstance(
            original_layers[1].activation,
            reentrant_api.ReentrantCheckpointWrapper,
        )
        assert isinstance(original_layers[1].router, CountingOp)

        value = ms.Tensor([1.0], ms.float32)
        value.requires_grad = True
        output = value
        for layer in model.layers:
            output = layer(output)
        output.sum().backward()

        assert original_layers[0].calls == 2
        assert original_layers[0].router.calls == 1
        assert original_layers[0].activation.calls == 2
        assert original_layers[1].calls == 1
        assert original_layers[1].router.calls == 1
        assert original_layers[1].activation.calls == 2
        assert value.grad.asnumpy().tolist() == [36.0]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestReentrantCompatibilityValidation:
    """Application-level guards for callers that do not build TrainConfig."""

    def test_reentrant_rejects_recompute_comm(self):
        """Reentrant checkpointing rejects communication recompute."""
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(
            mode="full",
            full_recompute_layer=["0"],
            comm_enable=True,
            comm_select_module={"attention": ["1"]},
            use_reentrant=True,
        )

        with pytest.raises(ValueError, match="recompute_comm.enable=True"):
            apply_ac(model, rc, rc_comm, SwapConfig(enable=False), 1)

    def test_reentrant_rejects_swap(self):
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(
            mode="full", full_recompute_layer=["0"], use_reentrant=True)

        with pytest.raises(ValueError, match="swap.enable=True"):
            apply_ac(
                model,
                rc,
                rc_comm,
                SwapConfig(enable=True, layer_swap=[{"layers": ["1"]}]),
                1,
            )


# ======================== Swap ========================


def _make_swap_config(enable=True, default_prefetch=1, layer_swap=None, op_swap=None):
    """Build a SwapConfig."""
    return SwapConfig(enable=enable, default_prefetch=default_prefetch,
                      layer_swap=layer_swap, op_swap=op_swap)


def _decoder_input_boundary(model):
    """Build the MindFormers decoder input-boundary rule used by swap."""
    return [ac_mod._SwapInputBoundary(
        model,
        range(model.layer_start, model.layer_end + 1),
        exclude_arg_names={"hidden_states"},
    )]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestSwapPolicy:
    """Test tensor selection in the activation swap policy."""

    def test_external_storage_alias_is_saved_without_shape_matching(self):
        """Storage identity protects aliases but not unrelated same-shape tensors."""
        policy = ac_mod._build_policy_fn_swap()
        external = ops.zeros((16, 4), ms.float32)
        alias = external.reshape((8, 8))
        same_shape = ops.zeros((8, 8), ms.float32)

        policy.capture_external_inputs((external,))

        assert policy(alias) == CheckpointPolicy.MUST_SAVE
        assert policy(same_shape) == CheckpointPolicy.MUST_SWAP

    def test_decoder_hook_saves_all_external_inputs_except_hidden_states(self):
        """Decoder mask, RoPE, ids and prefix stay while hidden states still swap."""
        model = MockModel(num_layers=2)
        sc = _make_swap_config(layer_swap=[{"layers": ["0"]}])
        apply_swap(model, sc, input_boundaries=_decoder_input_boundary(model))
        hidden_states = ops.zeros((8, 4), ms.float32)
        attention_mask = ops.zeros((1, 1, 8, 8), ms.uint8)
        rotary_pos_emb = ops.zeros((8, 1, 1, 64), ms.float32)
        prefix_key = ops.zeros((1, 8, 64), ms.float32)
        actual_seq_len = ops.zeros((1,), ms.int32)
        input_ids = ops.zeros((1, 8), ms.int32)
        cos = ops.zeros((8, 1, 1, 64), ms.float32)
        sin = ops.ones((8, 1, 1, 64), ms.float32)
        unrelated = ops.zeros((8, 1, 1, 64), ms.float32)
        policy = model.layers[0].policy_fn

        model(hidden_states, attention_mask, rotary_pos_emb, ((prefix_key,),), actual_seq_len,
              input_ids=input_ids, rotary_cos_sin=(cos, sin))

        assert policy(hidden_states) == CheckpointPolicy.MUST_SWAP
        for tensor in (attention_mask, rotary_pos_emb, prefix_key, actual_seq_len, input_ids, cos, sin):
            assert policy(tensor) == CheckpointPolicy.MUST_SAVE
        assert policy(unrelated) == CheckpointPolicy.MUST_SWAP

    def test_decoder_hook_refreshes_external_storage_each_forward(self):
        """A new decoder forward drops storage identities from the previous batch."""
        model = MockModel(num_layers=2)
        sc = _make_swap_config(layer_swap=[{"layers": ["0"]}])
        apply_swap(model, sc, input_boundaries=_decoder_input_boundary(model))
        hidden_states = ops.zeros((8, 4), ms.float32)
        first_mask = ops.zeros((1, 1, 8, 8), ms.uint8)
        second_mask = ops.ones((1, 1, 8, 8), ms.uint8)
        policy = model.layers[0].policy_fn

        model(hidden_states, first_mask)
        assert policy(first_mask) == CheckpointPolicy.MUST_SAVE

        model(hidden_states, second_mask)
        assert policy(first_mask) == CheckpointPolicy.MUST_SWAP
        assert policy(second_mask) == CheckpointPolicy.MUST_SAVE

    def test_excluded_input_is_bound_by_name_not_position(self):
        """Changing construct argument order does not change the exclusion rule."""
        model = ReorderedMockModel(num_layers=2)
        sc = _make_swap_config(layer_swap=[{"layers": ["0"]}])
        apply_swap(model, sc, input_boundaries=_decoder_input_boundary(model))
        attention_mask = ops.zeros((1, 1, 8, 8), ms.uint8)
        hidden_states = ops.zeros((8, 4), ms.float32)
        input_ids = ops.zeros((1, 8), ms.int32)
        policy = model.layers[0].policy_fn

        model(attention_mask, hidden_states, input_ids=input_ids)

        assert policy(hidden_states) == CheckpointPolicy.MUST_SWAP
        assert policy(attention_mask) == CheckpointPolicy.MUST_SAVE
        assert policy(input_ids) == CheckpointPolicy.MUST_SAVE

    def test_mtp_hook_saves_all_external_tensor_inputs(self):
        """MTP keeps every tensor entering the enclosing block, including hidden states."""
        decoder = MockMtpDecoder(num_layers=2, mtp_num_layers=2)
        mtp = MockMtpBlock(mtp_num_layers=2)
        rc, rc_comm = _make_recompute_config()
        sc = _make_swap_config(layer_swap=[{"layers": ["2"]}])
        apply_ac(decoder, rc, rc_comm, sc, 1, mtp_block=mtp)
        input_ids = ops.zeros((1, 8), ms.int32)
        position_ids = ops.zeros((1, 8), ms.int32)
        hidden_states = ops.zeros((8, 1, 64), ms.float32)
        attention_mask = ops.zeros((1, 1, 8, 8), ms.uint8)
        rotary_pos_emb = ops.zeros((8, 1, 1, 64), ms.float32)
        actual_seq_len = ops.zeros((1,), ms.int32)
        policy = mtp.layers[0].policy_fn

        mtp(input_ids, position_ids, hidden_states, attention_mask,
            rotary_pos_emb=rotary_pos_emb, actual_seq_len=actual_seq_len)

        for tensor in (input_ids, position_ids, hidden_states, attention_mask, rotary_pos_emb, actual_seq_len):
            assert policy(tensor) == CheckpointPolicy.MUST_SAVE

    def test_mtp_hook_is_not_registered_without_mtp_swap_target(self):
        """An existing MTP block adds no hook when only decoder layers are swapped."""
        decoder = MockMtpDecoder(num_layers=2, mtp_num_layers=1)
        mtp = MockMtpBlock(mtp_num_layers=1)
        rc, rc_comm = _make_recompute_config()
        sc = _make_swap_config(layer_swap=[{"layers": ["0"]}])

        apply_ac(decoder, rc, rc_comm, sc, 1, mtp_block=mtp)

        assert hasattr(decoder, "_swap_input_storage_hook_handle")
        assert not hasattr(mtp, "_swap_input_storage_hook_handle")

    def test_only_mtp_hook_is_registered_for_mtp_op_swap_target(self):
        """An MTP-only op target activates the MTP boundary but not the decoder boundary."""
        decoder = MockMtpDecoder(num_layers=2, mtp_num_layers=2)
        mtp = MockMtpBlock(mtp_num_layers=2)
        rc, rc_comm = _make_recompute_config()
        sc = _make_swap_config(op_swap=[{"op_name": "transformer_layer", "layers": ["2"]}])

        apply_ac(decoder, rc, rc_comm, sc, 1, mtp_block=mtp)

        assert not hasattr(decoder, "_swap_input_storage_hook_handle")
        assert hasattr(mtp, "_swap_input_storage_hook_handle")
        assert isinstance(mtp.layers[0].transformer_layer, SwapWrapper)

    def test_decoder_and_mtp_boundaries_keep_isolated_policy_state(self):
        """Entering MTP does not overwrite the decoder boundary's storage set."""
        decoder = MockMtpDecoder(num_layers=2, mtp_num_layers=2)
        mtp = MockMtpBlock(mtp_num_layers=2)
        rc, rc_comm = _make_recompute_config()
        sc = _make_swap_config(layer_swap=[{"layers": ["0", "2"]}])
        apply_ac(decoder, rc, rc_comm, sc, 1, mtp_block=mtp)
        decoder_hidden = ops.zeros((8, 1, 64), ms.float32)
        decoder_mask = ops.zeros((1, 1, 8, 8), ms.uint8)
        mtp_input_ids = ops.zeros((1, 8), ms.int32)
        mtp_position_ids = ops.zeros((1, 8), ms.int32)
        mtp_hidden = ops.ones((8, 1, 64), ms.float32)
        decoder_policy = decoder.layers[0].policy_fn
        mtp_policy = mtp.layers[0].policy_fn

        decoder(decoder_hidden, decoder_mask)
        mtp(mtp_input_ids, mtp_position_ids, mtp_hidden, decoder_mask)

        assert decoder_policy is not mtp_policy
        assert decoder_policy(decoder_mask) == CheckpointPolicy.MUST_SAVE
        assert decoder_policy(mtp_hidden) == CheckpointPolicy.MUST_SWAP
        assert mtp_policy(mtp_hidden) == CheckpointPolicy.MUST_SAVE


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestLayerSwap:
    """Test layer swap: entire layers are wrapped with swap_wrapper."""

    def test_layer_swap_single_layer(self):
        model = MockModel(num_layers=3)
        sc = _make_swap_config(layer_swap=[{"layers": ["0"]}])
        apply_swap(model, sc)
        assert isinstance(model.layers[0], SwapWrapper)
        assert not isinstance(model.layers[1], SwapWrapper)
        assert not isinstance(model.layers[2], SwapWrapper)

    def test_layer_swap_range(self):
        model = MockModel(num_layers=3)
        sc = _make_swap_config(layer_swap=[{"layers": ["0-1"]}])
        apply_swap(model, sc)
        assert isinstance(model.layers[0], SwapWrapper)
        assert isinstance(model.layers[1], SwapWrapper)
        assert not isinstance(model.layers[2], SwapWrapper)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestOpSwap:
    """Test op swap: specific modules within layers are wrapped."""

    def test_op_swap_by_name(self):
        model = MockModel(num_layers=3)
        sc = _make_swap_config(op_swap=[{"op_name": "attention", "layers": ["0"]}])
        apply_swap(model, sc)
        assert isinstance(model.layers[0].attention, SwapWrapper)
        assert not isinstance(model.layers[0].mlp, SwapWrapper)

    def test_op_swap_multiple_modules(self):
        model = MockModel(num_layers=3)
        sc = _make_swap_config(op_swap=[
            {"op_name": "attention", "layers": ["0"]},
            {"op_name": "mlp", "layers": ["1"]},
        ])
        apply_swap(model, sc)
        assert isinstance(model.layers[0].attention, SwapWrapper)
        assert isinstance(model.layers[1].mlp, SwapWrapper)


# ======================== MTP recompute ========================


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestMtpRecompute:
    """MTP layers are addressable as the last layers (MTP layer i == layer num_layers + i)."""

    def test_full_recompute_mtp_layer_only(self):
        decoder = MockMtpDecoder(num_layers=2, mtp_num_layers=1)
        mtp = MockMtpBlock(mtp_num_layers=1)
        rc, rc_comm = _make_recompute_config(mode="full", full_recompute_layer=["2"])
        apply_ac(decoder, rc, rc_comm, _make_swap_config(enable=False), 1, mtp_block=mtp)
        assert not isinstance(decoder.layers[0], CheckpointWrapper)
        assert not isinstance(decoder.layers[1], CheckpointWrapper)
        assert isinstance(mtp.layers[0], CheckpointWrapper)

    def test_full_recompute_decoder_and_mtp(self):
        decoder = MockMtpDecoder(num_layers=2, mtp_num_layers=2)
        mtp = MockMtpBlock(mtp_num_layers=2)
        rc, rc_comm = _make_recompute_config(mode="full", full_recompute_layer=["0-3"])
        apply_ac(decoder, rc, rc_comm, _make_swap_config(enable=False), 1, mtp_block=mtp)
        assert isinstance(decoder.layers[0], CheckpointWrapper)
        assert isinstance(decoder.layers[1], CheckpointWrapper)
        assert isinstance(mtp.layers[0], CheckpointWrapper)
        assert isinstance(mtp.layers[1], CheckpointWrapper)

    def test_select_recompute_mtp_transformer_layer(self):
        decoder = MockMtpDecoder(num_layers=2, mtp_num_layers=1)
        mtp = MockMtpBlock(mtp_num_layers=1)
        rc, rc_comm = _make_recompute_config(mode="select", select_module={"transformer_layer": ["2"]})
        apply_ac(decoder, rc, rc_comm, _make_swap_config(enable=False), 1, mtp_block=mtp)
        assert isinstance(mtp.layers[0].transformer_layer, CheckpointWrapper)
        # decoder layers are untouched
        assert not isinstance(decoder.layers[0].attention, CheckpointWrapper)

    def test_select_recompute_mtp_nested_attention(self):
        decoder = MockMtpDecoder(num_layers=2, mtp_num_layers=1)
        mtp = MockMtpBlock(mtp_num_layers=1)
        rc, rc_comm = _make_recompute_config(
            mode="select", select_module={"transformer_layer.attention": ["2"]})
        apply_ac(decoder, rc, rc_comm, _make_swap_config(enable=False), 1, mtp_block=mtp)
        assert isinstance(mtp.layers[0].transformer_layer.attention, CheckpointWrapper)
        assert not isinstance(mtp.layers[0].transformer_layer.mlp, CheckpointWrapper)

    def test_mtp_layer_id_out_of_range_raises(self):
        decoder = MockMtpDecoder(num_layers=2, mtp_num_layers=1)
        mtp = MockMtpBlock(mtp_num_layers=1)
        # valid ids are 0,1 (decoder) and 2 (MTP); 3 is out of range
        rc, rc_comm = _make_recompute_config(mode="full", full_recompute_layer=["3"])
        with pytest.raises(ValueError):
            apply_ac(decoder, rc, rc_comm, _make_swap_config(enable=False), 1, mtp_block=mtp)

    def test_no_mtp_block_keeps_decoder_only_namespace(self):
        # Without an MTP block, layer id 2 (== the would-be MTP layer) is out of range.
        decoder = MockMtpDecoder(num_layers=2, mtp_num_layers=0)
        rc, rc_comm = _make_recompute_config(mode="full", full_recompute_layer=["2"])
        with pytest.raises(ValueError):
            apply_ac(decoder, rc, rc_comm, _make_swap_config(enable=False), 1, mtp_block=None)
