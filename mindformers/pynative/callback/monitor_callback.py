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
"""Monitor callback for outputting monitor metrics."""

from mindformers.pynative.callback.callback import TrainerCallback


class MonitorCallback(TrainerCallback):
    """Callback for outputting monitor-collected metrics at each training step end."""

    def on_step_end(self, args, state, **kwargs):
        monitor = kwargs.get("monitor")
        if monitor and monitor.active:
            if kwargs.get("pp_metric_reduce_group") is not None:
                loss = kwargs.get("loss")
                if loss is not None and monitor.should_record("device_loss"):
                    monitor.record("device_loss", loss)
                if monitor.should_record("device_norm"):
                    monitor.record("device_norm")
            monitor.flush(step=state.global_step)
