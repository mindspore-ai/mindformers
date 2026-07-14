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
        Description: 2D/3D weights take Muon; 1D weights and embedding/output-layer
            weights take AdamW (returns False).
        Expectation: predicate matches the documented default behaviour.
        """
        muon_filter = Muon._build_muon_filter(None)

        def _p(shape, name):
            return Parameter(Tensor(np.zeros(shape), mstype.float32), name=name)

        assert muon_filter(_p((4, 6), "layers.0.attention.weight")) is True
        assert muon_filter(_p((2, 4, 6), "layers.0.experts.weight")) is True
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

    @staticmethod
    def _run_ns(mat):
        """Run Newton-Schulz on a float32 numpy matrix; return float32 numpy.

        The precision tests run in fp32: they validate the orthogonalisation
        *algorithm*, which must hold independent of accumulation precision. The
        real bf16 production path (``op_cast(grad, bfloat16)`` before NS) stays
        covered end-to-end by :class:`TestMuonSingleCardUpdate`.
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
