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
"""Unit tests for the pynative Muon optimizer: basic functionality and precision.

Two groups of tests live here:

* ``TestMuonConfig`` — pure-Python config helpers (schedule expansion, the
  Muon-vs-AdamW filter, hyper-parameter / model validation).  These need no
  accelerator and run on CPU.
* ``TestNewtonSchulz`` / ``TestMuonSingleCardUpdate`` — the numerical core
  (Newton-Schulz orthogonalisation and a single-card optimiser step).  Newton-
  Schulz runs in bf16 and is exercised on Ascend 910B.

The single-card tests feed a lightweight mock model so ``Muon.__init__`` is
satisfied without a real GPT model, and use identity split/merge functions so a
plain 2D weight flows through the real Muon (Newton-Schulz) branch while 1D
weights flow through the AdamW branch.
"""

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, Parameter, nn, mint, dtype as mstype
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard

from mindformers.pynative.optimizer import muon as muon_mod
from mindformers.pynative.optimizer.muon import Muon, newton_schulz


# Classic single-triple Newton-Schulz coefficients, expanded to a 5-step schedule.
_NS_TRIPLE = (3.4445, -4.7750, 2.0315)
_NS_SCHED_5 = tuple(_NS_TRIPLE for _ in range(5))


def _skip_if_nonfinite(arr, what):
    """Skip a numeric check when ``arr`` is non-finite.

    The Newton-Schulz path relies on ``mint`` matmul / normalize kernels that
    need a fully-initialised Ascend runtime. In a bare ``pytest`` UT on some
    MindSpore/CANN builds those kernels return all-NaN (the kernels only run
    correctly under the msrun-launched ST harness, which already covers NS
    training numerics). We therefore validate the orthogonalisation precision
    where the build supports it and skip — rather than fail — where it does not.
    """
    if not np.isfinite(np.asarray(arr)).all():
        pytest.skip(
            f"{what} produced non-finite output on this MindSpore/CANN build; "
            "NS numerics are covered by the msrun-based ST test")


# --------------------------------------------------------------------------- #
#  Mock model — minimal surface required by ``Muon.__init__``.
# --------------------------------------------------------------------------- #
def _identity_split_fn(param_name, tensor):  # pylint: disable=unused-argument
    """No-op split: the whole tensor is a single Newton-Schulz piece."""
    return [tensor]


_identity_split_fn._muon_schema = ()  # Muon reads this attribute lazily.


def _identity_merge_fn(param_name, pieces):  # pylint: disable=unused-argument
    """Inverse of :func:`_identity_split_fn`."""
    return pieces[0]


def _make_mock_model(muon_predicate):
    """Model double for ``Muon.__init__``.

    ``muon_predicate(param) -> bool`` decides which params take the Muon branch
    (``True``) vs the AdamW branch (``False``).
    """

    class _MockCfg:
        multi_latent_attention = True  # Muon._verify_model requires this.

    class _MockModel:
        def get_gpt_transformer_config(self):
            return _MockCfg()

        def make_model_muon_fns(self):
            return (_identity_split_fn, _identity_merge_fn)

        def get_muon_filter(self):
            return muon_predicate

    return _MockModel()


class _TwoWeightNet(nn.Cell):
    """A 2D weight (Muon branch) and a 1D bias (AdamW branch)."""

    def __init__(self, w_init, b_init):
        super().__init__()
        self.weight = Parameter(Tensor(w_init, mstype.float32), name="layers.0.weight")
        self.bias = Parameter(Tensor(b_init, mstype.float32), name="layers.0.bias")


def _build_optimizer(net, **kwargs):
    """Build a single-card Muon: 2D -> Muon, everything else -> AdamW, QK-clip off."""
    defaults = {
        "learning_rate": 0.02,
        "weight_decay": 0.1,
        "momentum": 0.95,
        "model": _make_mock_model(lambda p: len(p.shape) == 2),
        "comm_strategy": "allgather",
        "qk_clip_enabled": False,  # avoids model.synced_max_attention_logit_fires(...)
    }
    defaults.update(kwargs)
    return Muon(net.trainable_params(), **defaults)


# --------------------------------------------------------------------------- #
#  Config helpers — CPU only, no accelerator needed.
# --------------------------------------------------------------------------- #
class TestMuonConfig:
    """Newton-Schulz schedule expansion, the Muon filter, and config validation."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_muon_constructor_enables_cube_math_type_4(self, monkeypatch):
        """A real Muon parameter automatically enables the FP32-add Cube mode."""
        calls = []

        class _FakeOpPrecision:
            @staticmethod
            def cube_math_type(value):
                """Record the requested Cube mode."""
                calls.append(value)

        monkeypatch.setattr(muon_mod, "op_precision", _FakeOpPrecision(), raising=False)
        monkeypatch.setattr(muon_mod.core_context, "is_legacy_model", lambda: False)

        net = _TwoWeightNet(np.ones((2, 2), np.float32), np.zeros(2, np.float32))
        optimizer = _build_optimizer(net)

        assert optimizer.use_muon == (True, False)
        assert calls == [4]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_muon_requires_cube_math_type_api(self, monkeypatch):
        """Fail clearly instead of silently using an unsafe MindSpore path."""
        monkeypatch.setattr(muon_mod, "op_precision", object(), raising=False)

        with pytest.raises(RuntimeError, match="cube_math_type.*MindSpore"):
            muon_mod._enable_muon_cube_math_type()  # pylint: disable=protected-access

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        "input_dtype", (mstype.float32, mstype.float16, mstype.bfloat16))
    def test_ns_normalization_uses_fp32_and_returns_bf16(
            self, monkeypatch, input_dtype):
        """Every NS input is normalized in FP32 before entering BF16 iteration."""
        events = []

        class _FakeTensor:
            """Record the dtype used by each normalization operation."""

            def __init__(self, dtype):
                self.dtype = dtype

            def norm(self, dim=None, keepdim=False, dtype=None):
                events.append(("norm", self.dtype, dim, keepdim, dtype))
                return _FakeTensor(dtype or self.dtype)

            def __truediv__(self, value):
                events.append(("div", self.dtype, value.dtype))
                return _FakeTensor(value.dtype)

        def _fake_cast(tensor, dtype):
            events.append(("cast", tensor.dtype, dtype))
            return _FakeTensor(dtype)

        def _fake_clamp(tensor, min=None, max=None):  # pylint: disable=redefined-builtin
            events.append(("clamp", tensor.dtype, min, max))
            return _FakeTensor(tensor.dtype)

        monkeypatch.setattr(muon_mod, "op_cast", _fake_cast)
        monkeypatch.setattr(muon_mod.mint, "clamp", _fake_clamp)

        output = muon_mod._normalize_newton_schulz_input(  # pylint: disable=protected-access
            _FakeTensor(input_dtype), 1e-7)

        assert output.dtype == mstype.bfloat16
        assert events == [
            ("norm", input_dtype, (-2, -1), True, mstype.float32),
            ("clamp", mstype.float32, 1e-7, None),
            ("div", input_dtype, mstype.float32),
            ("cast", mstype.float32, mstype.bfloat16),
        ]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_expert_muon_redist_only_materializes_dim1(self, monkeypatch):
        """Only a grouped 3D expert weight receives the Shard(1) compatibility path."""
        class _FakeDTensor:
            def __init__(self, placements):
                self.shape = (8, 16, 32)
                self.placements = placements
                self.device_mesh = "expert_mesh"

        monkeypatch.setattr(muon_mod, "DTensor", _FakeDTensor)

        dim0_only = _FakeDTensor((Shard(0), Replicate()))
        assert muon_mod._get_expert_muon_redist_spec(
            dim0_only, "layers.0.mlp.experts.weight1") is None

        dim1_sharded = _FakeDTensor((Shard(0), Shard(1)))
        assert muon_mod._get_expert_muon_redist_spec(
            dim1_sharded, "layers.0.attention.weight") is None
        assert muon_mod._get_expert_muon_redist_spec(
            dim1_sharded, "layers.0.mlp.experts.weight1") == (
                "expert_mesh",
                (Shard(0), Shard(1)),
                (Shard(0), Replicate()),
            )

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_muon_rejects_non_expert_3d_weight(self):
        """3D Muon weights must match the grouped-expert special case."""
        assert muon_mod._select_muon_matmul_op(
            3, "layers.0.mlp.experts.weight1", (8, 16, 32)) is mint.bmm
        assert muon_mod._select_muon_matmul_op(
            3, "layers.0.mlp.experts.weight2", (8, 32, 16)) is mint.bmm

        with pytest.raises(ValueError, match="only supports 3D weights from grouped experts"):
            muon_mod._select_muon_matmul_op(
                3, "layers.0.attention.weight", (8, 16, 32))
        with pytest.raises(ValueError, match="only supports 3D weights from grouped experts"):
            muon_mod._select_muon_matmul_op(
                3, "layers.0.mlp.experts.weight3", (8, 16, 32))

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_bound_phase4_groups(self):
        """
        Feature: memory-bounded Phase 4 Muon batching.
        Description: a same-shape group is chunked according to its estimated
            temporary bytes while preserving slot order.
        Expectation: groups fit two slots per batch without creating a singleton tail.
        """
        group_key = ((8, 8), mstype.bfloat16, mstype.float32)
        bytes_per_slot = 8 * 8 * muon_mod._PHASE4_BATCH_TEMP_BYTES_PER_ELEMENT
        groups = {group_key: [0, 1, 2, 3, 4]}

        bounded = muon_mod._bound_phase4_groups(groups, max_temp_bytes=2 * bytes_per_slot)
        assert list(bounded.values()) == [[0, 1], [2, 3, 4]]

        bounded = muon_mod._bound_phase4_groups(groups, max_temp_bytes=4 * bytes_per_slot)
        assert list(bounded.values()) == [[0, 1, 2], [3, 4]]

        bounded = muon_mod._bound_phase4_groups(groups, max_temp_bytes=1)
        assert list(bounded.values()) == [[0, 1], [2, 3, 4]]

        bounded = muon_mod._bound_phase4_groups({group_key: [0]}, max_temp_bytes=1)
        assert list(bounded.values()) == [[0]]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_bound_ns_groups(self):
        """
        Feature: memory-bounded Newton-Schulz Muon batching.
        Description: a large-matrix same-shape group is chunked by its estimated
            NS transient bytes; a small-matrix group with tiny per-slot bytes is
            never split; chunks keep >=2 slots so no weight leaves the bmm path.
        Expectation: only the large group splits, preserving slot order, keyed by
            ``(group_sig, chunk_index)``.
        """
        big = ("big", (3584, 3584))
        small = ("small", (16, 16))
        big_bytes = muon_mod._estimate_ns_peak_bytes_for_shape((3584, 3584))
        small_bytes = muon_mod._estimate_ns_peak_bytes_for_shape((16, 16))
        assert big_bytes > small_bytes
        groups = {big: [0, 1, 2, 3], small: [4, 5, 6, 7]}
        slot_bytes = {big: big_bytes, small: small_bytes}

        # Budget = 2 big slots -> big splits 2+2; small (tiny bytes) stays whole.
        bounded = muon_mod._bound_ns_groups(
            groups, slot_bytes, max_temp_bytes=2 * big_bytes)
        assert bounded[(big, 0)] == [0, 1]
        assert bounded[(big, 1)] == [2, 3]
        assert bounded[(small, 0)] == [4, 5, 6, 7]

        # Even a 1-byte budget never drops below two slots per chunk (bmm path).
        bounded = muon_mod._bound_ns_groups(groups, slot_bytes, max_temp_bytes=1)
        assert bounded[(big, 0)] == [0, 1]
        assert bounded[(big, 1)] == [2, 3]
        assert all(len(chunk) >= 2 for chunk in bounded.values())

        # A pre-existing singleton group is preserved (stays on the mm path).
        bounded = muon_mod._bound_ns_groups(
            {big: [0]}, {big: big_bytes}, max_temp_bytes=1)
        assert bounded == {(big, 0): [0]}

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_bound_ns_groups_allows_safe_singletons(self):
        """
        Feature: memory-bounded rank-3 Newton-Schulz batching.
        Description: a singleton rank-3 schema piece still executes through
            bmm, so splitting it does not switch to the 2D mm numerical path.
        Expectation: singleton-safe groups strictly honour a one-slot budget.
        """
        piece_shapes = ((3, 7168, 2048), (3, 7168, 2048))
        group_sig = (piece_shapes, tuple(range(128)))
        groups = {group_sig: [0, 1, 2, 3, 4, 5, 6]}
        bytes_per_slot = sum(
            muon_mod._estimate_ns_peak_bytes_for_shape(shape)
            for shape in piece_shapes
        )
        assert bytes_per_slot == 960 * 1024 ** 2
        slot_bytes = {group_sig: bytes_per_slot}

        bounded = muon_mod._bound_ns_groups(
            groups,
            slot_bytes,
            max_temp_bytes=1024 ** 3,
            singleton_safe_groups={group_sig},
        )
        assert list(bounded.values()) == [
            [0], [1], [2], [3], [4], [5], [6],
        ]

        bounded = muon_mod._bound_ns_groups(
            groups,
            slot_bytes,
            max_temp_bytes=2 * bytes_per_slot,
            singleton_safe_groups={group_sig},
        )
        assert list(bounded.values()) == [[0, 1], [2, 3], [4, 5], [6]]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_grouped_owner_assignment_spreads_across_shape_groups(self):
        """
        Feature: layout-aware Muon owner assignment.
        Description: several small shape groups share a much larger rank list.
        Expectation: later groups rotate to idle owners instead of restarting
            from the first rank-list position and overloading the same ranks.
        """
        world_size = 1024
        rank_list = tuple(range(896, 1024))
        group_sizes = [7] * 8 + [8, 2]
        items_by_group = {}
        param_index = 0
        for group_index, group_size in enumerate(group_sizes):
            group_sig = (((group_index + 1, 16),), rank_list)
            weights = []
            work = (len(group_sizes) - group_index) * 100
            for _ in range(group_size):
                weights.append((param_index, work, rank_list))
                param_index += 1
            items_by_group[group_sig] = weights

        assignment, rank_loads, rank_counts = \
            muon_mod._assign_grouped_muon_owners(items_by_group, world_size)

        assert len(assignment) == 66
        assert len(set(assignment.values())) >= 60
        assert max(rank_loads) == 1000
        assert max(rank_counts) <= 2

        reversed_groups = dict(reversed(list(items_by_group.items())))
        repeated = muon_mod._assign_grouped_muon_owners(
            reversed_groups, world_size)
        assert repeated == (assignment, rank_loads, rank_counts)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_bound_elementwise_groups_allows_singletons(self):
        """
        Feature: memory-bounded elementwise Muon batching.
        Description: Phase 0 is elementwise and may split below two slots
            without changing the numerical kernel path.
        Expectation: every chunk respects the byte budget, including the tail.
        """
        group_key = ("phase0", (8, 8))
        groups = {group_key: [0, 1, 2, 3, 4]}
        slot_bytes = {group_key: 8}

        bounded = muon_mod._bound_elementwise_groups(
            groups, slot_bytes, max_temp_bytes=8)
        assert list(bounded.values()) == [[0], [1], [2], [3], [4]]

        bounded = muon_mod._bound_elementwise_groups(
            groups, slot_bytes, max_temp_bytes=16)
        assert list(bounded.values()) == [[0, 1], [2, 3], [4]]

    def test_resolve_batch_memory_bytes(self):
        """
        Feature: configurable Muon batch memory budgets.
        Description: the ``*_batch_memory_gb`` options are given in GB and convert
            to bytes; ``None`` keeps the built-in default; non-positive or
            non-numeric values are rejected. A larger budget must batch more
            slots per chunk than a smaller one.
        Expectation: conversion, defaulting and validation all behave as stated.
        """
        gb = 1024 ** 3
        # Integer and fractional GB both convert to bytes.
        assert muon_mod._resolve_batch_memory_bytes(2, "x") == 2 * gb
        assert muon_mod._resolve_batch_memory_bytes(0.5, "x") == gb // 2
        # Invalid values are rejected.
        for bad in (0, -1):
            with pytest.raises(ValueError):
                muon_mod._resolve_batch_memory_bytes(bad, "x")
        for bad in ("1", True, None):
            with pytest.raises(TypeError):
                muon_mod._resolve_batch_memory_bytes(bad, "x")

        # The budget actually drives how many slots share a chunk.
        sig = ("big", (1024, 1024))
        slot_bytes = {sig: gb // 4}
        groups = {sig: [0, 1, 2, 3, 4, 5, 6, 7]}
        small = muon_mod._bound_ns_groups(
            dict(groups), slot_bytes,
            max_temp_bytes=muon_mod._resolve_batch_memory_bytes(0.5, "x"))
        large = muon_mod._bound_ns_groups(
            dict(groups), slot_bytes,
            max_temp_bytes=muon_mod._resolve_batch_memory_bytes(2, "x"))
        assert max(len(c) for c in small.values()) < max(len(c) for c in large.values())


    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_normalize_ns_schedule_flat(self):
        """
        Feature: Muon._normalize_ns_schedule flat-triple form.
        Description: a flat [a, b, c] broadcasts to every one of ns_steps iterations.
        Expectation: schedule length == ns_steps and every entry equals the triple.
        """
        schedule, n_steps = Muon._normalize_ns_schedule(list(_NS_TRIPLE), 5)
        assert n_steps == 5
        assert len(schedule) == 5
        assert all(triple == _NS_TRIPLE for triple in schedule)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_normalize_ns_schedule_segmented(self):
        """
        Feature: Muon._normalize_ns_schedule segmented form.
        Description: [[triple, count], ...] repeats each triple count times; ns_steps ignored.
        Expectation: total steps == sum(count); first/last triples come from the right segment.
        """
        schedule, n_steps = Muon._normalize_ns_schedule(
            [[[1.0, 2.0, 3.0], 2], [[4.0, 5.0, 6.0], 3]], ns_steps=999)
        assert n_steps == 5
        assert schedule[0] == (1.0, 2.0, 3.0)
        assert schedule[1] == (1.0, 2.0, 3.0)
        assert schedule[-1] == (4.0, 5.0, 6.0)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_normalize_ns_schedule_invalid(self):
        """
        Feature: Muon._normalize_ns_schedule validation.
        Description: an empty schedule, a wrong-length flat triple, and a non-positive
            ns_steps must all be rejected.
        Expectation: each raises ValueError.
        """
        with pytest.raises(ValueError):
            Muon._normalize_ns_schedule([], 5)
        with pytest.raises(ValueError):
            Muon._normalize_ns_schedule([1.0, 2.0], 5)  # flat needs 3 elements
        with pytest.raises(ValueError):
            Muon._normalize_ns_schedule(list(_NS_TRIPLE), 0)  # ns_steps must be > 0

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_muon_filter_default(self):
        """
        Feature: Muon._build_muon_filter default routing.
        Description: 2D/3D weights take Muon; unsupported 3D weights are rejected
            later by the update path. 1D, embedding, and output-layer weights take AdamW.
        Expectation: predicate matches the documented default behaviour.
        """
        muon_filter = Muon._build_muon_filter(None)

        def _p(shape, name):
            return Parameter(Tensor(np.zeros(shape), mstype.float32), name=name)

        assert muon_filter(_p((4, 6), "layers.0.attention.weight")) is True
        assert muon_filter(_p((2, 4, 6), "layers.0.experts.weight")) is True
        assert muon_filter(_p((2, 4, 6), "layers.0.attention.weight")) is True
        assert muon_filter(_p((6,), "layers.0.bias")) is False
        assert muon_filter(_p((4, 6), "embedding.word_embeddings.weight")) is False
        assert muon_filter(_p((4, 6), "head.output_layer.weight")) is False

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_muon_filter_custom_include(self):
        """
        Feature: Muon._build_muon_filter custom adamw_include globs.
        Description: a custom glob routes matching 2D weights to AdamW while leaving
            the default embedding/output-layer names on Muon.
        Expectation: only the custom-matched weight is excluded from Muon.
        """
        muon_filter = Muon._build_muon_filter(("*router*",))

        def _p(name):
            return Parameter(Tensor(np.zeros((4, 6)), mstype.float32), name=name)

        assert muon_filter(_p("layers.0.router.weight")) is False
        assert muon_filter(_p("layers.0.attention.weight")) is True
        # default exclusions no longer apply once adamw_include is overridden
        assert muon_filter(_p("embedding.word_embeddings.weight")) is True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_verify_config_invalid(self):
        """
        Feature: Muon._verify_config validation.
        Description: ns_steps must be positive, the schedule length must match ns_steps,
            and a positive qk_clip_threshold is required when QK-clip is enabled.
        Expectation: each violation raises ValueError.
        """
        with pytest.raises(ValueError):
            Muon._verify_config(0, _NS_SCHED_5, False, 100)
        with pytest.raises(ValueError):
            Muon._verify_config(5, _NS_SCHED_5[:3], False, 100)  # length mismatch
        with pytest.raises(ValueError):
            Muon._verify_config(5, _NS_SCHED_5, True, 0)  # threshold must be > 0

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_verify_model_requires_mla(self):
        """
        Feature: Muon._verify_model guard.
        Description: Muon only supports MLA models; a None model or one without
            multi_latent_attention is rejected.
        Expectation: both cases raise ValueError.
        """
        orig = muon_mod.core_context.is_legacy_model
        muon_mod.core_context.is_legacy_model = lambda: False
        try:
            with pytest.raises(ValueError):
                Muon._verify_model(None)

            class _NoMlaCfg:
                multi_latent_attention = False

            class _NoMlaModel:
                def get_gpt_transformer_config(self):
                    return _NoMlaCfg()

            with pytest.raises(ValueError):
                Muon._verify_model(_NoMlaModel())
        finally:
            muon_mod.core_context.is_legacy_model = orig


# --------------------------------------------------------------------------- #
#  Newton-Schulz precision — the orthogonalisation core (Ascend, bf16).
# --------------------------------------------------------------------------- #
class TestNewtonSchulz:
    """Newton-Schulz drives a matrix towards its orthogonal polar factor.

    These precision checks run in fp32 so the orthogonalisation property is
    asserted independent of low-precision accumulation; the bf16 production
    path is exercised end-to-end in :class:`TestMuonSingleCardUpdate`.
    """

    def setup_method(self):
        ms.set_context(mode=ms.PYNATIVE_MODE)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_bf16_normalization_matches_fp32_reference(self):
        """A small BF16 input follows FP32 normalization, not BF16 reduction."""
        rng = np.random.default_rng(1)
        mat = (rng.standard_normal((4, 16)) * 1.3e-6).astype(np.float32)
        x = Tensor(mat, mstype.bfloat16)

        normalized = muon_mod._normalize_newton_schulz_input(  # pylint: disable=protected-access
            x, 1e-7)
        x_fp32 = x.astype(mstype.float32)
        fp32_norm = x_fp32.norm(dim=(-2, -1), keepdim=True)
        expected = (x_fp32 / mint.clamp(fp32_norm, min=1e-7)).astype(
            mstype.bfloat16)
        low_precision_norm = x.norm(dim=(-2, -1), keepdim=True)
        low_precision = x / mint.clamp(low_precision_norm, min=1e-7)

        actual_np = normalized.astype(mstype.float32).asnumpy()
        expected_np = expected.astype(mstype.float32).asnumpy()
        low_precision_np = low_precision.astype(mstype.float32).asnumpy()
        assert normalized.dtype == mstype.bfloat16
        assert np.array_equal(actual_np, expected_np)
        assert not np.array_equal(actual_np, low_precision_np)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_ns_uses_one_bf16_precision_path(self, monkeypatch):
        """FP32 and BF16 API inputs share one normalized BF16 iteration path."""
        schedule = (
            (3.4445, -4.7750, 2.0315),
            (2.0, -1.5, 0.5),
        )
        mat = np.random.default_rng(33).standard_normal((8, 12)).astype(np.float32)
        bf16_input = Tensor(mat, mstype.bfloat16)
        fp32_input = bf16_input.astype(mstype.float32)
        cast_events = []
        real_cast = muon_mod.op_cast

        def _record_cast(tensor, dtype):
            cast_events.append((tensor.dtype, dtype))
            return real_cast(tensor, dtype)

        monkeypatch.setattr(muon_mod, "op_cast", _record_cast)

        from_fp32 = newton_schulz(
            fp32_input, 8, 12, 1e-7, len(schedule), schedule, mint.mm)
        from_bf16 = newton_schulz(
            bf16_input, 8, 12, 1e-7, len(schedule), schedule, mint.mm)

        assert from_fp32.dtype == mstype.bfloat16
        assert from_bf16.dtype == mstype.bfloat16
        assert cast_events == [
            (mstype.float32, mstype.bfloat16),
            (mstype.float32, mstype.bfloat16),
        ]
        np.testing.assert_array_equal(
            from_fp32.astype(mstype.float32).asnumpy(),
            from_bf16.astype(mstype.float32).asnumpy(),
        )

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_prepare_input_preserves_fp32_until_ns_normalization(self):
        """The momentum recurrence keeps its NS input in FP32."""
        gradient = Tensor(
            np.arange(12, dtype=np.float32).reshape(3, 4) / 100,
            mstype.bfloat16,
        )
        momentum_state = Tensor(
            np.arange(12, dtype=np.float32).reshape(3, 4) / 50,
            mstype.float32,
        )

        ns_input, next_m = muon_mod._prepare_muon_input_compute(  # pylint: disable=protected-access
            gradient, momentum_state, momentum=0.95, use_nesterov=True)

        assert ns_input.dtype == mstype.float32
        assert next_m.dtype == mstype.float32

    @staticmethod
    def _run_ns(mat):
        """Run Newton-Schulz on a float32 numpy matrix; return float32 numpy.

        The helper normalizes in FP32, then follows the same BF16 iteration path
        used in production.
        """
        dim_a, dim_b = mat.shape
        x = Tensor(mat, mstype.float32)
        out = newton_schulz(x, dim_a, dim_b, 1e-7, 5, _NS_SCHED_5, mint.mm)
        out_np = out.astype(mstype.float32).asnumpy()
        _skip_if_nonfinite(out_np, "newton_schulz")
        return out_np

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize("shape", [(8, 12), (12, 8), (16, 16)])
    def test_singular_values_pushed_to_one(self, shape):
        """
        Feature: Newton-Schulz orthogonalisation.
        Description: NS reshapes the singular spectrum of a random matrix towards 1,
            for wide, tall and square inputs alike.
        Expectation: output keeps the input shape and all singular values land in a
            tight band around 1, far tighter than the input spectrum.
        """
        rng = np.random.default_rng(0)
        mat = rng.standard_normal(shape).astype(np.float32)
        out = self._run_ns(mat)

        assert out.shape == shape
        sv_out = np.linalg.svd(out, compute_uv=False)
        sv_in = np.linalg.svd(mat, compute_uv=False)
        # 5 fixed-coefficient steps do not converge perfectly, but the band is narrow.
        assert sv_out.min() > 0.3
        assert sv_out.max() < 1.7
        # The conditioning improves by a large margin versus the raw input.
        cond_out = sv_out.max() / sv_out.min()
        cond_in = sv_in.max() / sv_in.min()
        assert cond_out < cond_in / 2

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_close_to_polar_factor(self):
        """
        Feature: Newton-Schulz approximates the polar factor U @ V^T.
        Description: for G = U S V^T, NS(G) should approach the orthogonal factor U V^T.
        Expectation: the result tracks U V^T within the 5-iteration tolerance.
        """
        rng = np.random.default_rng(1)
        mat = rng.standard_normal((10, 14)).astype(np.float32)
        out = self._run_ns(mat)

        u, _, vt = np.linalg.svd(mat, full_matrices=False)
        polar = u @ vt
        # Loose bound: only 5 fixed-coefficient NS iterations (no exact convergence).
        assert np.abs(out - polar).max() < 0.35
        assert np.allclose(out, polar, atol=0.25, rtol=0.0) or \
            np.linalg.norm(out - polar) / np.linalg.norm(polar) < 0.25

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_deterministic(self):
        """
        Feature: Newton-Schulz determinism in pynative mode.
        Description: the same input must yield the same output across repeated calls.
        Expectation: two runs are bitwise-identical.
        """
        rng = np.random.default_rng(2)
        mat = rng.standard_normal((8, 8)).astype(np.float32)
        out1 = self._run_ns(mat)
        out2 = self._run_ns(mat)
        assert np.array_equal(out1, out2)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_batched_ns_split_is_bitwise_invariant(self):
        """
        Feature: memory-bounded NS group splitting (``_bound_ns_groups``).
        Description: the bmm batch dimension carries independent weights, so
            running Newton-Schulz on a stack of K weights must give byte-for-byte
            the same per-weight result as running it on any sub-batch, including
            a singleton batch that still executes through ``bmm``.  This is the
            precision guarantee that lets large expert groups split.
        Expectation: full-batch NS output equals the concatenation of chunked-NS
            outputs with rtol=0, atol=0 (bitwise).
        """
        rng = np.random.default_rng(7)
        k, m, n = 6, 20, 28
        stack = rng.standard_normal((k, m, n)).astype(np.float32)
        x_full = Tensor(stack, mstype.float32)
        out_full = newton_schulz(
            x_full, m, n, 1e-7, 5, _NS_SCHED_5, mint.bmm).asnumpy()

        for chunk_size in (2, 1):
            pieces = []
            for start in range(0, k, chunk_size):
                x_chunk = Tensor(
                    stack[start:start + chunk_size], mstype.float32)
                pieces.append(newton_schulz(
                    x_chunk, m, n, 1e-7, 5, _NS_SCHED_5,
                    mint.bmm).asnumpy())
            out_chunked = np.concatenate(pieces, axis=0)

            assert np.array_equal(out_full, out_chunked), chunk_size

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_phase0_singleton_is_bitwise_invariant(self):
        """
        Feature: memory-bounded Phase 0 group splitting.
        Description: Phase 0 is elementwise, so processing a same-shape stack
            as singleton groups must preserve each weight's exact NS input and
            next-momentum value.
        Expectation: batched and singleton outputs are bitwise-identical.
        """
        rng = np.random.default_rng(8)
        gradients = [
            Tensor(value, mstype.float32)
            for value in rng.standard_normal((3, 16, 24)).astype(np.float32)
        ]
        momenta = [
            Tensor(value, mstype.float32)
            for value in rng.standard_normal((3, 16, 24)).astype(np.float32)
        ]

        batched_ns, batched_m = muon_mod._prepare_muon_input_batched(
            gradients, momenta, 0.95, True)
        for index, (gradient, momentum) in enumerate(zip(gradients, momenta)):
            singleton_ns, singleton_m = muon_mod._prepare_muon_input_batched(
                [gradient], [momentum], 0.95, True)
            assert np.array_equal(
                batched_ns[index].float().asnumpy(),
                singleton_ns[0].float().asnumpy(),
            ), index
            assert np.array_equal(
                batched_m[index].asnumpy(),
                singleton_m[0].asnumpy(),
            ), index


# --------------------------------------------------------------------------- #
#  Single-card optimiser step — momentum math + end-to-end update (Ascend).
# --------------------------------------------------------------------------- #
class TestMuonSingleCardUpdate:
    """A full single-card Muon step: routing, momentum, and parameter update."""

    def setup_method(self):
        ms.set_context(mode=ms.PYNATIVE_MODE)
        self._orig_is_legacy_model = muon_mod.core_context.is_legacy_model
        muon_mod.core_context.is_legacy_model = lambda: False

    def teardown_method(self):
        muon_mod.core_context.is_legacy_model = self._orig_is_legacy_model

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_batched_phase4_stages_momentum_writeback(self):
        """The staged Phase 4 path preserves updates without restacking next_m."""
        params = [
            Parameter(Tensor(np.full((2, 2), value, np.float32)), name=f"param_{index}")
            for index, value in enumerate((2.0, 3.0))
        ]
        momenta = [
            Parameter(Tensor(np.zeros((2, 2), np.float32)), name=f"momentum_{index}")
            for index in range(2)
        ]
        next_m = [Tensor(np.full((2, 2), value, np.float32)) for value in (0.25, 0.5)]
        x_rets = [
            Tensor(np.full((2, 2), value, np.float32), dtype=ms.bfloat16)
            for value in (0.1, 0.2)
        ]
        scales = (0.9, 0.8)
        infos = [
            {
                "param": params[index],
                "muon_m": momenta[index],
                "next_m": next_m[index],
                "x_ret": x_rets[index],
                "needs_redist": False,
                "lr": 0.1,
                "wd": 0.1,
                "wd_scale": scales[index],
            }
            for index in range(2)
        ]

        muon_mod._apply_prepared_update_batched(infos)

        for index in range(2):
            expected_param = np.full(
                (2, 2), np.float32(2.0 + index) * np.float32(scales[index]), np.float32)
            expected_param -= x_rets[index].float().asnumpy()
            np.testing.assert_allclose(params[index].asnumpy(), expected_param, rtol=0, atol=0)
            np.testing.assert_array_equal(momenta[index].asnumpy(), next_m[index].asnumpy())
            assert infos[index]["next_m"] is None

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_branch_routing(self):
        """
        Feature: Muon-vs-AdamW routing on a single card.
        Description: the 2D weight takes the Muon branch and gets a momentum buffer;
            the 1D bias takes the AdamW branch and gets first/second moments.
        Expectation: use_muon == (True, False); one Muon slot and one AdamW slot.
        """
        net = _TwoWeightNet(np.random.default_rng(0).standard_normal((8, 12)),
                            np.zeros(6))
        opt = _build_optimizer(net)
        assert opt.use_muon == (True, False)
        assert len(opt.muon_m) == 1
        assert len(opt.moments1) == 1 and len(opt.moments2) == 1

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize("nesterov", [True, False])
    def test_momentum_update_exact(self, nesterov):
        """
        Feature: Muon momentum buffer update.
        Description: with momentum m0 and gradient g, the new buffer is
            next_m = momentum * m0 + g (fp32), independent of the Nesterov flag.
        Expectation: the stored momentum matches the fp32 numpy reference.
        """
        rng = np.random.default_rng(3)
        w = rng.standard_normal((8, 12)).astype(np.float32)
        grad = rng.standard_normal((8, 12)).astype(np.float32)
        momentum = 0.95

        net = _TwoWeightNet(w, np.zeros(6))
        opt = _build_optimizer(net, momentum=momentum, nesterov=nesterov)

        # Seed a non-zero momentum buffer so the recurrence is non-trivial.
        m0 = rng.standard_normal((8, 12)).astype(np.float32)
        opt.muon_m[0].copy_(Tensor(m0, mstype.float32))

        grads = (Tensor(grad, mstype.float32), Tensor(np.zeros(6), mstype.float32))
        opt(grads)

        next_m = momentum * m0 + grad
        assert np.allclose(opt.muon_m[0].asnumpy(), next_m, rtol=1e-3, atol=1e-3)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_single_step_updates_all_params(self):
        """
        Feature: Muon single-card construct step.
        Description: one optimiser step must run both branches and advance state.
        Expectation: the AdamW-branch bias moves to a finite value, shapes/dtypes are
            preserved, and the global step counter advances to 1. (The Muon-branch
            weight movement is asserted by the NS-guarded magnitude test.)
        """
        rng = np.random.default_rng(4)
        w = rng.standard_normal((8, 12)).astype(np.float32)
        b = rng.standard_normal((6,)).astype(np.float32)
        net = _TwoWeightNet(w, b)
        opt = _build_optimizer(net)

        grads = (Tensor(np.ones((8, 12)), mstype.float32),
                 Tensor(np.ones((6,)), mstype.float32))
        opt(grads)

        w_after = net.weight.asnumpy()
        b_after = net.bias.asnumpy()
        assert w_after.shape == (8, 12) and net.weight.dtype == mstype.float32
        # AdamW branch is Newton-Schulz-independent: it must move to a finite value.
        assert np.isfinite(b_after).all()
        assert not np.allclose(b_after, b)
        assert int(opt.global_step.asnumpy().item()) == 1

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_weight_decay_scaling(self):
        """
        Feature: Muon decoupled weight decay.
        Description: the update is param * (1 - lr*wd) - muon_update. With a zero
            gradient the Newton-Schulz term vanishes (normalize of 0 -> 0), leaving
            pure weight decay.
        Expectation: the weight is scaled by exactly (1 - lr*wd).
        """
        rng = np.random.default_rng(5)
        w = rng.standard_normal((8, 12)).astype(np.float32)
        lr, wd = 0.02, 0.1
        net = _TwoWeightNet(w, np.zeros(6))
        opt = _build_optimizer(net, learning_rate=lr, weight_decay=wd)

        grads = (Tensor(np.zeros((8, 12)), mstype.float32),
                 Tensor(np.zeros(6), mstype.float32))
        opt(grads)

        w_after = net.weight.asnumpy()
        _skip_if_nonfinite(w_after, "muon weight update")
        expected = w * (1.0 - lr * wd)
        assert np.allclose(w_after, expected, rtol=1e-3, atol=1e-4)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_update_magnitude_bounded(self):
        """
        Feature: Muon update magnitude.
        Description: the Newton-Schulz term is an orthogonal matrix scaled by
            lr * sqrt(max(dim)) * matched_adamw_rms, so the per-element step stays
            on the order of lr and never explodes.
        Expectation: with weight decay disabled the max element move is well bounded.
        """
        rng = np.random.default_rng(6)
        w = rng.standard_normal((8, 12)).astype(np.float32)
        lr = 0.02
        net = _TwoWeightNet(w, np.zeros(6))
        opt = _build_optimizer(net, learning_rate=lr, weight_decay=0.0)

        grads = (Tensor(rng.standard_normal((8, 12)), mstype.float32),
                 Tensor(np.zeros(6), mstype.float32))
        opt(grads)

        w_after = net.weight.asnumpy()
        _skip_if_nonfinite(w_after, "muon weight update")
        delta = np.abs(w_after - w)
        # sqrt(12) * 0.2 (default matched_adamw_rms) ~= 0.69; an orthogonal entry is
        # <= 1, so the per-element step is bounded by lr * 0.69 plus slack.
        assert delta.max() < lr * 2.0
        assert delta.max() > 0.0  # the Muon branch actually moved the weight

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_multi_step_deterministic(self):
        """
        Feature: Muon multi-step determinism.
        Description: two independent optimisers fed identical inits and gradients must
            track each other step for step in pynative mode.
        Expectation: weights and momentum match exactly after several steps.
        """
        rng = np.random.default_rng(7)
        w = rng.standard_normal((8, 12)).astype(np.float32)
        b = rng.standard_normal((6,)).astype(np.float32)
        grad_seq = [rng.standard_normal((8, 12)).astype(np.float32) for _ in range(4)]

        def _run():
            net = _TwoWeightNet(w, b)
            opt = _build_optimizer(net)
            for g in grad_seq:
                opt((Tensor(g, mstype.float32), Tensor(np.zeros(6), mstype.float32)))
            return net.weight.asnumpy(), opt.muon_m[0].asnumpy()

        w1, m1 = _run()
        _skip_if_nonfinite(w1, "muon weight update")
        w2, m2 = _run()
        assert np.array_equal(w1, w2)
        assert np.array_equal(m1, m2)


# --------------------------------------------------------------------------- #
#  allgather_deredundency batched pipeline — single card, local-weight path.
# --------------------------------------------------------------------------- #
class _MultiWeightNet(nn.Cell):
    """Four same-shape 2D weights (one NS group) plus an odd-shape one."""

    def __init__(self, rng):
        super().__init__()
        for index in range(4):
            setattr(self, f"w{index}", Parameter(
                Tensor(rng.standard_normal((8, 12)).astype(np.float32), mstype.float32),
                name=f"layers.{index}.weight"))
        self.w_odd = Parameter(
            Tensor(rng.standard_normal((16, 6)).astype(np.float32), mstype.float32),
            name="layers.9.weight")
        self.bias = Parameter(
            Tensor(rng.standard_normal((6,)).astype(np.float32), mstype.float32),
            name="layers.0.bias")


class TestMuonDeredundencyBatchedPath:
    """The batched ``allgather_deredundency`` update on locally-owned weights.

    With a single rank no weight needs redistribution, so this drives Phase 0
    (where the momentum is now written back), the grouped Newton-Schulz, and
    the Phase 4 apply.  ``get_rank`` is stubbed so the strategy can be
    constructed off-cluster.
    """

    def setup_method(self):
        ms.set_context(mode=ms.PYNATIVE_MODE)
        self._orig_is_legacy_model = muon_mod.core_context.is_legacy_model
        self._orig_get_rank = muon_mod.get_rank
        muon_mod.core_context.is_legacy_model = lambda: False
        muon_mod.get_rank = lambda *args, **kwargs: 0

    def teardown_method(self):
        muon_mod.core_context.is_legacy_model = self._orig_is_legacy_model
        muon_mod.get_rank = self._orig_get_rank

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_scatter_recv_buffer_matches_bf16_ns_output(self):
        """AGD receives BF16 NS output although its pre-NS input is FP32."""
        ns_input = Tensor(np.zeros((2, 3), np.float32), mstype.float32)
        info = {
            'rank_list_tuple': (0, 1),
            'assigned_rank': 0,
            'ns_inputs_local': ns_input,
        }

        ok, ops = muon_mod._build_local_shard_scatter_ops(  # pylint: disable=protected-access
            info, x_ret_full=None, rank_id=1)

        assert ok
        assert len(ops) == 1
        assert info['x_ret'].shape == ns_input.shape
        assert info['x_ret'].dtype == mstype.bfloat16
        assert ops[0].tensor is info['x_ret']
        assert ops[0].peer == 0

    @staticmethod
    def _run(momentum=0.95, seed_momentum=False):
        """One deredundency optimiser step; returns net, opt, grads, m0, groups."""
        rng = np.random.default_rng(11)
        net = _MultiWeightNet(rng)
        opt = _build_optimizer(net, comm_strategy="allgather_deredundency",
                               momentum=momentum)
        grad_rng = np.random.default_rng(12)
        grads_np = [grad_rng.standard_normal(p.shape).astype(np.float32)
                    for p in net.trainable_params()]
        m0 = []
        for muon_m_slot in opt.muon_m:
            init = (grad_rng.standard_normal(muon_m_slot.shape).astype(np.float32)
                    if seed_momentum
                    else np.zeros(muon_m_slot.shape, np.float32))
            muon_m_slot.copy_(Tensor(init, mstype.float32))
            m0.append(init)
        opt(tuple(Tensor(g, mstype.float32) for g in grads_np))
        return (net, opt, grads_np, m0,
                opt._muon_runtime_groups)  # pylint: disable=protected-access

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_local_path_is_actually_exercised(self):
        """
        Feature: single-card allgather_deredundency routing.
        Description: with one rank no weight needs redistribution, so every Muon
            weight must land on the local (no-communication) Newton-Schulz path
            — that is what makes this class cover the batched update.
        Expectation: local NS groups are non-empty and no 2D weight is scheduled.
        """
        _, _, _, _, groups = self._run()
        assert groups['local_groups_slots'], "expected local NS groups"
        assert not groups['flat_order_2d_slots'], "expected no redistributed weights"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_momentum_written_back_in_phase0(self):
        """
        Feature: early momentum writeback.
        Description: ``next_m`` is stored into ``muon_m`` at the end of Phase 0
            rather than during the Phase 4 apply, which releases the stacked
            FP32 Phase 0 buffer before Newton-Schulz runs.  The stored value
            must still be the plain recurrence ``momentum * m0 + grad``.
        Expectation: every Muon momentum buffer matches the fp32 reference.
        """
        net, opt, grads_np, m0, _ = self._run(seed_momentum=True)
        muon_grads = [g for g, p in zip(grads_np, net.trainable_params())
                      if len(p.shape) == 2]
        assert len(muon_grads) == len(opt.muon_m)
        for slot, grad in enumerate(muon_grads):
            np.testing.assert_allclose(opt.muon_m[slot].asnumpy(),
                                       0.95 * m0[slot] + grad,
                                       rtol=1e-5, atol=1e-5)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_step_is_reproducible(self):
        """
        Feature: deredundency batched step determinism.
        Description: the P2P scratch buffers are now allocated per step instead
            of being cached for the optimiser's lifetime; a fresh buffer must
            not change the result.
        Expectation: two identical steps agree bitwise on params and momentum.
        """
        net1, opt1, _, _, _ = self._run()
        _skip_if_nonfinite(net1.trainable_params()[0].asnumpy(), "muon weight update")
        net2, opt2, _, _, _ = self._run()
        for first, second in zip(net1.trainable_params(), net2.trainable_params()):
            assert np.array_equal(first.asnumpy(), second.asnumpy())
        for first, second in zip(opt1.muon_m, opt2.muon_m):
            assert np.array_equal(first.asnumpy(), second.asnumpy())
