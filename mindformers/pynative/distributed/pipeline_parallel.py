# Copyright (c) Meta Platforms, Inc. and affiliates
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
from typing import List, Tuple

import mindspore as ms
from mindspore import nn

from hyper_parallel import PipelineStage

class PpLayerSetting:
    """
    Layer setting for pipeline parallelism.
    """
    def __init__(self, num_layers: int, parallelism_config):
        self.num_layers = num_layers
        self.pp = parallelism_config.pipeline_parallel
        self.pp_interleave_num = parallelism_config.pipeline_parallel_interleave_num
        self.total_stages = self.pp * self.pp_interleave_num
        self.avg_layer = None

        self.layers_per_stage = parallelism_config.pipeline_parallel_layers_per_stage
        if self.layers_per_stage == "auto":
            self._compute_auto()
        else:
            self._init_explicit(self.layers_per_stage)

    def _init_explicit(self, layers_per_stage: List[str]):
        after_parse_layer = {}
        for pp_rank, stage_str in enumerate(layers_per_stage):
            ranges = stage_str.split(',')
            for chunk_id, r in enumerate(ranges):
                start, end = r.strip().split('-')
                after_parse_layer[str(chunk_id * self.pp + pp_rank)] = ((int(start), int(end)))

        self.layers_per_stage = after_parse_layer

    def _compute_auto(self):
        if self.num_layers % self.total_stages != 0:
            raise ValueError(
                f"Number of layers ({self.num_layers}) is not divisible by total stages ({self.total_stages})."
                f"Please configure the model distribution via 'pipeline_parallel_layers_per_stage'." 
            )
        self.avg_layer = self.num_layers // self.total_stages

    def get_layer_range(self, stage_id: int) -> Tuple[int, int]:
        if self.avg_layer:
            layer_start_id = stage_id * self.avg_layer
            layer_end_id = (stage_id + 1) * self.avg_layer - 1
            return (layer_start_id, layer_end_id + 1)
        return self.layers_per_stage[str(stage_id)]

    @property
    def num_virtual_stages(self) -> int:
        return self.pp * self.pp_interleave_num

class StageModelBuilder:
    """
    Builder for creating pipeline parallel stages.
    """
    def __init__(self, layer_setting: PpLayerSetting):
        self.layer_setting = layer_setting

    def build_stages(
        self,
        model_cls: type,
        model_config,
        pp_mesh,
    ) -> tuple[list[PipelineStage], list[nn.Cell]]:
        """
        Build pipeline parallel stages.
        """
        num_virtual_stages = self.layer_setting.num_virtual_stages
        local_stage_indices = self._get_local_stage_indices(pp_mesh)
        
        stages = []
        model_parts = []
        
        for stage_idx in local_stage_indices:
            layer_start, layer_end = self.layer_setting.get_layer_range(stage_idx)
            is_first = stage_idx == 0
            is_last = stage_idx == num_virtual_stages - 1
            
            with ms.DeviceCtx("meta"):
                stage_model = model_cls(
                    model_config,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    pre_process=is_first,
                    post_process=is_last,
                    stage_idx=stage_idx,
                    vp_size=num_virtual_stages,
                )
            model_parts.append(stage_model)
            
            stage = PipelineStage(
                submodule=stage_model,
                stage_index=stage_idx,
                stage_num=num_virtual_stages,
                device=pp_mesh.device_type,
                group=pp_mesh.get_group(),
                mesh=pp_mesh,
            )
            stages.append(stage)
        
        return stages, model_parts

    def _get_local_stage_indices(self, pp_mesh) -> list[int]:
        pp_rank = pp_mesh.get_local_rank()
        pp_size = pp_mesh.size()
        interleave_num = self.layer_setting.pp_interleave_num
        return [chunk_id * pp_size + pp_rank for chunk_id in range(interleave_num)]
