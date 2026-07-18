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
import pytest
import mindspore as ms
from mindspore import nn, ops
from hyper_parallel.platform.mindspore.activation_checkpoint import CheckpointWrapper, SwapWrapper
from hyper_parallel.core.activation_checkpoint import CheckpointPolicy

from mindformers.pynative.config.config import (
    RecomputeCommConfig,
    RecomputeConfig,
    SwapConfig,
)
import mindformers.pynative.distributed.activation_checkpoint as ac_mod
from mindformers.pynative.distributed.activation_checkpoint import (
    apply_ac,
    apply_recompute,
    apply_swap,
    _build_exclude_op_policy,
)


@pytest.fixture(autouse=True)
def _reset_config_list():
    """Isolate the module-global whitelist cache between tests."""
    ac_mod._config_list = {}
    yield
    ac_mod._config_list = {}


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
                 select_module=None, comm_enable=False, comm_select_module=None, exclude_op=None):
    """Build recompute and recompute_comm configs."""
    rc = RecomputeConfig(mode=mode, full_recompute_layer=full_recompute_layer,
                         select_module=select_module, exclude_op=exclude_op)
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


# ======================== Exclude op (no-recompute) ========================


class _FakeOp:
    """Stand-in for a dispatched op exposing a ``name`` attribute."""

    def __init__(self, name):
        self.name = name


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestExcludeOp:
    """Test exclude_op: configured ops are kept off the recompute path (case-insensitive)."""

    def test_policy_case_insensitive_substring(self):
        policy = _build_exclude_op_policy(["AllGather", "reducescatter"])
        # case-insensitive substring match against the real InnerComm* dispatch names -> saved
        assert policy(None, _FakeOp("InnerCommAllGather")) == CheckpointPolicy.MUST_SAVE
        assert policy(None, _FakeOp("innercommallgather")) == CheckpointPolicy.MUST_SAVE
        assert policy(None, _FakeOp("InnerCommReduceScatter")) == CheckpointPolicy.MUST_SAVE
        # non-matching ops are recomputed (MUST_RECOMPUTE; PREFER_RECOMPUTE is rejected by
        # the MindSpore selective-checkpoint dispatch)
        assert policy(None, _FakeOp("MatMulExt")) == CheckpointPolicy.MUST_RECOMPUTE
        assert policy(None, _FakeOp("InnerCommAllReduce")) == CheckpointPolicy.MUST_RECOMPUTE

    def test_policy_matches_compute_op(self):
        # exclude_op is not comm-specific: a compute op like matmul is kept too
        policy = _build_exclude_op_policy(["matmul"])
        assert policy(None, _FakeOp("MatMulExt")) == CheckpointPolicy.MUST_SAVE
        assert policy(None, _FakeOp("BatchMatMulExt")) == CheckpointPolicy.MUST_SAVE
        # other ops (including collectives) are still recomputed
        assert policy(None, _FakeOp("InnerCommReduceScatter")) == CheckpointPolicy.MUST_RECOMPUTE
        assert policy(None, _FakeOp("RmsNorm")) == CheckpointPolicy.MUST_RECOMPUTE

    def test_full_recompute_attaches_exclude_policy(self):
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(mode="full", full_recompute_layer=["0"],
                                             exclude_op=["AllGather"])
        apply_recompute(model, rc, rc_comm)
        wrapped = model.layers[0]
        assert isinstance(wrapped, CheckpointWrapper)
        assert callable(wrapped.checkpoint_kwargs.get("policy_fn"))

    def test_select_recompute_attaches_exclude_policy(self):
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(mode="select", select_module={"attention": ["0"]},
                                             exclude_op=["AllToAll"])
        apply_recompute(model, rc, rc_comm)
        wrapped = model.layers[0].attention
        assert isinstance(wrapped, CheckpointWrapper)
        assert callable(wrapped.checkpoint_kwargs.get("policy_fn"))

    def test_recompute_without_exclude_op_has_no_policy(self):
        model = MockModel(num_layers=2)
        rc, rc_comm = _make_recompute_config(mode="full", full_recompute_layer=["0"])
        apply_recompute(model, rc, rc_comm)
        wrapped = model.layers[0]
        assert isinstance(wrapped, CheckpointWrapper)
        assert "policy_fn" not in wrapped.checkpoint_kwargs
