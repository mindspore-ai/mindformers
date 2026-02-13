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
"""Test Callback and CallbackHandler"""

from unittest.mock import MagicMock, patch
import pytest

from mindformers.pynative.callback.callback import TrainerCallback, CallbackHandler


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestTrainerCallback:
    """Test TrainerCallback base class"""

    def test_methods(self):
        """Test that all methods can be called without error"""
        callback = TrainerCallback()
        args = MagicMock()
        state = MagicMock()

        # Ensure all methods can be called without error (default implementation is empty)
        callback.on_begin(args, state)
        callback.on_end(args, state)
        callback.on_train_begin(args, state)
        callback.on_train_end(args, state)
        callback.on_epoch_begin(args, state)
        callback.on_epoch_end(args, state)
        callback.on_step_begin(args, state)
        callback.on_step_end(args, state)


class TestCallbackHandler:
    """Test CallbackHandler class"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = MagicMock()
        self.optimizer = MagicMock()
        self.handler = CallbackHandler(model=self.model, optimizer=self.optimizer)

    def test_init(self):
        """Test initialization with callbacks"""
        cb1 = TrainerCallback()
        handler = CallbackHandler(callbacks=[cb1])
        assert cb1 in handler.callbacks

    def test_add_callback(self):
        """Test add_callback"""
        # Test adding instance
        cb1 = TrainerCallback()
        self.handler.add_callback(cb1)
        assert cb1 in self.handler.callbacks

        # Test adding class
        class MyCallback(TrainerCallback):
            pass

        self.handler.add_callback(MyCallback)
        assert any(isinstance(c, MyCallback) for c in self.handler.callbacks)

        # Test adding duplicate warning
        with patch("mindformers.pynative.callback.callback.logger") as mock_logger:
            self.handler.add_callback(MyCallback)  # Add again
            mock_logger.warning.assert_called()

    def test_remove_callback(self):
        """Test remove_callback"""
        cb1 = TrainerCallback()
        cb2 = TrainerCallback()
        self.handler.add_callback(cb1)
        self.handler.add_callback(cb2)

        # Test remove instance
        self.handler.remove_callback(cb1)
        assert cb1 not in self.handler.callbacks
        assert cb2 in self.handler.callbacks

        # Test remove class
        class MyCallback(TrainerCallback):
            pass

        cb3 = MyCallback()
        self.handler.add_callback(cb3)
        self.handler.remove_callback(MyCallback)
        assert cb3 not in self.handler.callbacks

        # Test remove non-existent
        self.handler.remove_callback(TrainerCallback)  # Should not raise error

    def test_pop_callback(self):
        """Test pop_callback"""
        cb1 = TrainerCallback()
        self.handler.add_callback(cb1)

        # Pop instance
        popped = self.handler.pop_callback(cb1)
        assert popped == cb1
        assert cb1 not in self.handler.callbacks

        # Pop class
        class MyCallback(TrainerCallback):
            pass

        cb2 = MyCallback()
        self.handler.add_callback(cb2)
        popped = self.handler.pop_callback(MyCallback)
        assert popped == cb2
        assert cb2 not in self.handler.callbacks

        # Pop non-existent
        assert self.handler.pop_callback(TrainerCallback) is None

    def test_event_calls(self):
        """Test all event calls"""
        cb1 = MagicMock(spec=TrainerCallback)
        cb2 = MagicMock(spec=TrainerCallback)
        self.handler.add_callback(cb1)
        self.handler.add_callback(cb2)

        args = MagicMock()
        state = MagicMock()

        # Define events to test
        events = [
            "on_begin",
            "on_end",
            "on_train_begin",
            "on_train_end",
            "on_epoch_begin",
            "on_epoch_end",
            "on_step_begin",
            "on_step_end",
        ]

        for event in events:
            # Call the event on handler
            getattr(self.handler, event)(args, state, extra_arg="test")

            # Verify callback method was called with correct args
            getattr(cb1, event).assert_called_with(
                args,
                state,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=None,
                train_dataset=None,
                eval_dataset=None,
                extra_arg="test",
            )
            getattr(cb2, event).assert_called()

    def test_callback_list(self):
        """Test callback_list property"""

        class CallbackA(TrainerCallback):
            pass

        class CallbackB(TrainerCallback):
            pass

        self.handler.add_callback(CallbackA())
        self.handler.add_callback(CallbackB())

        cb_list = self.handler.callback_list
        assert "CallbackA" in cb_list
        assert "CallbackB" in cb_list
        assert "\n" in cb_list
