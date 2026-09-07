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
"""Tests for the ``calculate_per_token_loss`` plumbing: the shape contract between the loss
and the trainer's backward sense, and the auxiliary-loss scale."""
from types import SimpleNamespace

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, mint

from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.pynative.distributed.utils import get_loss_sense
from mindformers.pynative.trainer import trainer as trainer_module
from mindformers.pynative.trainer.trainer import Trainer
from mindformers.pynative.loss.loss import ChunkCrossEntropyLoss, CrossEntropyLoss

TOTAL_VOCAB_SIZE = 32
BATCH_TOKENS = 16


def _inputs():
    np.random.seed(0)
    logits = Tensor(np.random.randn(BATCH_TOKENS, TOTAL_VOCAB_SIZE).astype(np.float32))
    labels = Tensor(np.random.randint(0, TOTAL_VOCAB_SIZE, (BATCH_TOKENS,)).astype(np.int32))
    mask = np.ones((BATCH_TOKENS,), np.float32)
    mask[-3:] = 0
    return logits, labels, Tensor(mask)


class TestShapeContract:
    """The trainer back-propagates with ``sense = Tensor([s])``, i.e. shape (1,). Anything the
    loss hands it must match, or backward dies with
    ``Mismatch in shape: grad_output[0] has a shape of (1,) and output[0] has a shape of ()``.
    """

    def test_backward_sense_is_rank_one(self):
        sense = get_loss_sense(parallelism=SimpleNamespace(pipeline_parallel=1))
        assert sense.shape == (1,)

    def test_plain_loss_is_rank_one(self):
        loss = CrossEntropyLoss(calculate_per_token_loss=False)(*_inputs())
        assert loss.shape == (1,)

    def test_per_token_numerator_matches_the_sense(self):
        """The numerator is a 0-d reduction; only the denominator picks up (1,) from the
        broadcast epsilon. Without the explicit reshape the plain path is accidentally right
        and the per-token path crashes."""
        numerator, denominator = CrossEntropyLoss(calculate_per_token_loss=True)(*_inputs())
        assert numerator.shape == (1,)
        assert denominator.shape == (1,)

    def test_per_token_ratio_equals_the_plain_loss(self):
        """numerator/denominator must reproduce the averaged loss exactly."""
        plain = CrossEntropyLoss(calculate_per_token_loss=False)(*_inputs())
        numerator, denominator = CrossEntropyLoss(calculate_per_token_loss=True)(*_inputs())
        assert np.allclose(float(mint.div(numerator, denominator)[0]), float(plain[0]),
                           rtol=1e-6, atol=0)

    def test_denominator_counts_unmasked_tokens(self):
        _, denominator = CrossEntropyLoss(calculate_per_token_loss=True)(*_inputs())
        assert pytest.approx(BATCH_TOKENS - 3, abs=1e-3) == float(denominator[0])


class TestAuxiliaryLossScale:
    """Per-token accumulates micro-batches instead of averaging them, so the 1/grad_accum
    factor must leave the auxiliary autoscalers too -- the token weighting in the router /
    MTP branches replaces it."""

    @staticmethod
    def _sense(per_token, accum):
        return float(get_loss_sense(
            parallelism=SimpleNamespace(pipeline_parallel=1),
            gradient_accumulation_steps=accum,
            apply_gradient_accumulation=not per_token,
        )[0])

    def test_plain_mode_folds_in_grad_accum(self):
        assert self._sense(per_token=False, accum=4) == pytest.approx(0.25)

    def test_per_token_mode_does_not(self):
        assert self._sense(per_token=True, accum=4) == pytest.approx(1.0)


class TestChunkedLossIsAllowed:
    """The two were mutually exclusive because the fused chunk kernels baked ``1/denominator``
    into forward and backward; they now take a per-token switch."""

    def test_chunked_per_token_config_is_accepted(self):
        config = MLATransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, ffn_hidden_size=128,
            chunk_loss_num=4, calculate_per_token_loss=True,
        )
        assert config.chunk_loss_num == 4
        assert config.calculate_per_token_loss is True


def _chunk_inputs():
    """The same tokens as ``_inputs()``, shaped ``[1, seq, vocab]`` for the chunked path."""
    logits, labels, mask = _inputs()
    return (mint.reshape(logits, (1, BATCH_TOKENS, TOTAL_VOCAB_SIZE)),
            mint.reshape(labels, (1, BATCH_TOKENS)),
            mint.reshape(mask, (1, BATCH_TOKENS)))


class TestChunkedPerTokenMatchesPlain:
    """The chunked kernels carry the per-token contract independently of the plain path.

    ``_ChunkCrossEntropyLoss`` skips the baked-in ``1/denominator`` in *both* forward and
    backward, and each chunk contributes an un-normalised numerator that must add up to the
    plain path's. Nothing else pins that down: the plain-path tests above all pass even if a
    chunk keeps its own divisor, if the numerator loses its ``(1,)`` shape, or if forward and
    backward disagree about the divisor -- the loss log stays plausible while every gradient
    is silently rescaled by a token count.
    """

    @staticmethod
    def _plain():
        """The reference ``(numerator, denominator)`` from the non-chunked kernel."""
        return CrossEntropyLoss(calculate_per_token_loss=True)(*_inputs())

    @staticmethod
    def _chunked(chunk_loss_num=4):
        """The same pair from the chunked kernel, over the identical tokens."""
        loss = ChunkCrossEntropyLoss(calculate_per_token_loss=True, chunk_loss_num=chunk_loss_num)
        return loss(*_chunk_inputs())

    @pytest.mark.parametrize("chunk_loss_num", [2, 4, 16, 32])
    def test_pair_matches_the_plain_path(self, chunk_loss_num):
        """Holds for chunk counts that divide the sequence, and for ones that do not."""
        num, den = self._chunked(chunk_loss_num)
        plain_num, plain_den = self._plain()
        assert num.shape == (1,) and den.shape == (1,)
        np.testing.assert_allclose(num.asnumpy(), plain_num.asnumpy(), rtol=1e-5)
        np.testing.assert_allclose(den.asnumpy(), plain_den.asnumpy(), rtol=0)

    def test_denominator_counts_unmasked_tokens(self):
        _, den = self._chunked()
        _, _, mask = _inputs()
        assert float(den) == float(mint.sum(mask))

    def test_gradient_matches_the_plain_path(self):
        """The divisor must be dropped symmetrically: forward alone cannot catch a backward
        that still divides by the chunk's own token count."""
        _, labels, mask = _inputs()

        def plain_fn(logits):
            num, _ = CrossEntropyLoss(calculate_per_token_loss=True)(logits, labels, mask)
            return num

        _, chunk_labels, chunk_mask = _chunk_inputs()

        def chunked_fn(logits):
            loss = ChunkCrossEntropyLoss(calculate_per_token_loss=True, chunk_loss_num=4)
            num, _ = loss(logits, chunk_labels, chunk_mask)
            return num

        logits = _inputs()[0]
        plain_grad = ms.value_and_grad(plain_fn)(logits)[1]
        chunked_grad = ms.value_and_grad(chunked_fn)(
            mint.reshape(logits, (1, BATCH_TOKENS, TOTAL_VOCAB_SIZE)))[1]
        np.testing.assert_allclose(
            mint.reshape(chunked_grad, (BATCH_TOKENS, TOTAL_VOCAB_SIZE)).asnumpy(),
            plain_grad.asnumpy(), rtol=1e-5, atol=1e-6)

    def test_plain_mode_still_normalises(self):
        """Guard the other direction: with the switch off the chunked path must keep
        returning a single normalised loss, not the pair."""
        loss = ChunkCrossEntropyLoss(calculate_per_token_loss=False, chunk_loss_num=4)
        out = loss(*_chunk_inputs())
        assert not isinstance(out, (tuple, list))
        plain_num, plain_den = self._plain()
        np.testing.assert_allclose(
            out.asnumpy(), (plain_num / plain_den).asnumpy(), rtol=1e-5)

    def test_vocab_parallel_branch_is_selected_by_the_tp_group(self):
        """The TP branch needs a real communication group, so a single-card test can only
        pin the dispatch: with no TP group the plain chunk kernel must be the one used."""
        loss = ChunkCrossEntropyLoss(calculate_per_token_loss=True, chunk_loss_num=4)
        assert loss._tp_group is None
        assert hasattr(loss, "compensate_loss_sense_tp")


def setup_module():
    ms.set_context(mode=ms.PYNATIVE_MODE)


class _FakeParam:
    """A parameter whose gradient ``scale_grads_`` can mutate in place."""

    def __init__(self, value):
        """Hold ``value`` as this parameter's gradient."""
        self.grad = Tensor(np.array(value, np.float32))


class TestTrainerLevelNormalisation:
    """Pin the divisor ``_finalize_per_token_grads`` applies to the whole gradient buffer.

    This is the step the shape tests above cannot reach: micro-batches back-propagate raw
    token-loss sums, and the single division happens once per optimizer step. If the divisor
    drifts -- a stale loss sense, a missing ``1/(dp*tp*cp)``, a runtime rebuild that
    re-applies ``1/grad_accum`` -- every gradient is rescaled while the reported loss
    ``numerator/denominator`` stays exactly right, so nothing else fails.
    """

    @staticmethod
    def _trainer(**attrs):
        """A bare trainer carrying only what ``_finalize_per_token_grads`` reads."""
        trainer = Trainer.__new__(Trainer)
        trainer.calculate_per_token_loss = True
        trainer.enable_parallel = False
        for key, value in attrs.items():
            setattr(trainer, key, value)
        return trainer

    def test_single_card_divides_by_the_token_count(self):
        """Every gradient is divided once by the accumulated token count."""
        params = [_FakeParam([4.0, 8.0]), _FakeParam(2.0)]
        trainer = self._trainer(optimizer=SimpleNamespace(parameters=params))

        num, den = trainer._finalize_per_token_grads(Tensor([26.0]), Tensor([13.0]))

        np.testing.assert_allclose(params[0].grad.asnumpy(), [4.0 / 13.0, 8.0 / 13.0], rtol=1e-6)
        np.testing.assert_allclose(params[1].grad.asnumpy(), 2.0 / 13.0, rtol=1e-6)
        # The reported step loss is the ratio, unaffected by the gradient scaling.
        assert float(num) / float(den) == 2.0

    def test_a_none_gradient_is_skipped(self):
        """Parameters that never received a gradient must not raise."""
        param = _FakeParam(1.0)
        param.grad = None
        trainer = self._trainer(optimizer=SimpleNamespace(parameters=[param]))

        trainer._finalize_per_token_grads(Tensor([1.0]), Tensor([1.0]))

        assert param.grad is None


class TestRuntimeSeedMatchesTheInitialSeed:
    """A runtime micro-batch change must not re-introduce the ``1/grad_accum`` factor.

    ``apply_pp`` leaves that factor out of the last-stage seed under per-token loss, because
    the single division happens in the trainer. ``_update_pipeline_microbatch`` rebuilds the
    same seed when the accumulation count changes at runtime, and hard-coding
    ``apply_gradient_accumulation=True`` there scales every gradient by ``1/accum`` relative
    to the initial seed -- invisible in the reported loss, so only this test catches it.
    """

    @staticmethod
    def _run(per_token, monkeypatch, new_accum=4):
        """Rebuild the schedule for ``new_accum`` and return the seed it wrote to the stage."""
        monkeypatch.setattr(trainer_module, "_MicroBatch", lambda *a, **k: object())
        stage = SimpleNamespace(loss_scale=None)
        schedule = SimpleNamespace(
            micro_batch_num=1, stages=[stage],
            _args_batch_dim=0, _kwargs_batch_dim=0,
            build_exec_order=lambda: None,
        )
        trainer = Trainer.__new__(Trainer)
        trainer.schedule = schedule
        trainer.enable_parallel = False
        trainer.calculate_per_token_loss = per_token
        trainer.config = SimpleNamespace(parallelism=SimpleNamespace(
            data_parallel_shard=1, tensor_parallel=1, context_parallel=1, pipeline_parallel=1))

        trainer._update_pipeline_microbatch(new_accum)
        return stage.loss_scale

    def test_per_token_seed_keeps_the_raw_sum(self, monkeypatch):
        """No 1/accum: the micro-batch token-loss sums must reach the trainer un-averaged."""
        assert self._run(True, monkeypatch) == pytest.approx(1.0)

    def test_plain_seed_still_folds_in_grad_accum(self, monkeypatch):
        """The other direction must not regress: without per-token loss the seed keeps 1/accum."""
        assert self._run(False, monkeypatch) == pytest.approx(0.25)
