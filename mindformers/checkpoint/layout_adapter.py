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
"""Load/save checkpoint APIs for distributed parallel layout management."""
from typing import Dict, Tuple

import mindspore as ms
from mindspore import Parameter
from mindspore.nn.cell import Cell
from mindspore.parallel.strategy import get_strategy_metadata

from mindformers.core.context.build_context import get_context
from mindformers.tools.logger import logger

try:
    from hyper_parallel.core.distributed_checkpoint import get_global_layout
    from hyper_parallel.core.dtensor.dtensor import DTensor
except ImportError as e:
    get_global_layout = None
    DTensor = None
    logger.warning(f"Import get_global_layout, DTensor failed: {e}.")


from mindformers.tools.utils import get_real_rank


class LayoutAdapter:
    """
    Adapter class for extracting and managing distributed parallel layout information from MindSpore networks.

    This class provides unified interfaces to retrieve sharding strategy metadata across different
    parallel modes (PyNative and Graph mode), supporting both single-rank and multi-rank scenarios.
    """

    @staticmethod
    def is_pynative_mode() -> bool:
        """
        Check if the current execution mode is PyNative mode.

        Returns:
            bool: True if running in PyNative mode, False if in Graph mode.
        """
        return get_context("mode") == ms.context.PYNATIVE_MODE

    @staticmethod
    def get_all_layouts(network: Cell) -> Dict[int, Dict[str, list]]:
        """
        Retrieve distributed parallel layout information for all ranks in the network.

        This method automatically detects the execution mode and delegates to the appropriate
        backend implementation to collect layout metadata across all participating ranks.

        Args:
            network (Cell): The MindSpore network cell containing distributed parameters
                and their sharding strategies.

        Returns:
            Dict[int, Dict[str, list]]: A nested dictionary where:
                - Outer keys are rank IDs (int).
                - Inner dictionaries map parameter names (str) to layout information lists containing:
                    [ms.Layout object, parameter type, full shape].
                Returns empty dict if no layout information is available.
        """
        if LayoutAdapter.is_pynative_mode():
            return LayoutAdapter._get_layout_from_pynative(network)
        return LayoutAdapter._get_layout_from_graph(network)


    @staticmethod
    def get_current_layout(network: Cell) -> Dict[str, list]:
        """
        Retrieve distributed parallel layout information for the current rank only.

        This method extracts layout metadata specific to the executing rank, which is useful
        for rank-specific checkpoint operations and local tensor reconstruction.

        Args:
            network (Cell): The MindSpore network cell containing distributed parameters
                and their sharding strategies.

        Returns:
            Dict[str, list]: A dictionary mapping parameter names (str) to layout information
                lists containing [ms.Layout object, parameter type, full shape] for the
                current rank. Returns empty dict if no layout information is available.
        """
        rank_id = get_real_rank()
        if LayoutAdapter.is_pynative_mode():
            return LayoutAdapter._get_layout_from_pynative(network)[rank_id]
        return LayoutAdapter._get_layout_from_graph(network)[0]


    @staticmethod
    def _get_layout_from_pynative(network: Cell) -> Dict[int, Dict[str, list]]:
        """
        Extract distributed parallel layout information from PyNative mode execution.

        This method retrieves global layout metadata from the hyper_parallel framework and
        constructs ms.Layout objects for each parameter across all ranks. It handles the
        conversion of raw layout dictionaries into structured Layout instances with proper
        tensor map configuration.

        Args:
            network (Cell): The MindSpore network cell containing distributed parameters
                and their sharding strategies in PyNative mode.

        Returns:
            Dict[int, Dict[str, list]]: A nested dictionary where:
                - Outer keys are rank IDs (int).
                - Inner dictionaries map parameter names (str) to layout information lists:
                    [ms.Layout object with device_matrix/alias_name/rank_list/tensor_map,
                     parameter type, full shape].
                Returns empty dict if global layout is not available.
        """
        if get_global_layout is None:
            raise ImportError("hyper_parallel is required for PyNative mode. Please install it.")
        global_layout_dict = get_global_layout(network)

        if not global_layout_dict:
            return {}

        result = {}
        for rank_id, current_layout_dict in global_layout_dict.items():
            rank_id = int(rank_id)
            if rank_id not in result:
                result[rank_id] = {}
            for param_name, param_info in current_layout_dict.items():
                if 'device_matrix' not in param_info:
                    result[rank_id][param_name] = [None, param_info['type'], param_info['full_shape']]
                else :
                    layout_info = ms.Layout(
                        device_matrix=tuple(param_info['device_matrix']),
                        alias_name=tuple(param_info['alias_name']),
                        rank_list=list(param_info['rank_list'])
                    )
                    layout_info._tensor_map = tuple(param_info['tensor_map'])
                    result[rank_id][param_name] = [layout_info, param_info['type'], param_info['full_shape']]

        return result

    @staticmethod
    def _get_layout_from_graph(network: Cell) -> Dict[int, Dict[str, list]]:
        """
        Extract distributed parallel layout information from Graph mode execution.

        This method retrieves strategy metadata that was generated during graph compilation,
        which contains the optimized sharding decisions made by MindSpore's parallel compiler.

        Args:
            network (Cell): The MindSpore network cell containing distributed parameters
                and their compiled sharding strategies in Graph mode.

        Returns:
            Dict[int, Dict[str, list]]: A nested dictionary where:
                - Outer keys are rank IDs (int).
                - Inner dictionaries map parameter names (str) to layout information lists
                  obtained from the compiled strategy metadata.
        """
        return get_strategy_metadata(network)

    @staticmethod
    def preprocess_params(network: Cell):
        """
        Preprocess network parameters by converting DTensor types to Parameter in PyNative mode.

        This function is primarily used for parameter conversion in distributed training scenarios.
        In PyNative mode, if parameters are of DTensor type (distributed tensor), they are converted
        to Parameter form for subsequent saving or processing operations.

        Args:
            network (Cell): MindSpore neural network cell object containing parameters to be processed

        Returns:
            state_dict (Dict[str, Parameter]): Returns the original network object if not in PyNative mode;
                                If in PyNative mode, returns a dictionary where keys are parameter names
                                and values are processed parameter objects (DTensor converted to Parameter)
        """
        if not LayoutAdapter.is_pynative_mode():
            return network
        if DTensor is None:
            raise ImportError("DTensor is required for PyNative mode. Please install it.")
        state_dict = {}
        for param_name, param in network.parameters_dict().items():
            if isinstance(param, DTensor):
                param_value = Parameter([])
                param_value.data = param.to_local()
                param_value.name = param_name
                param_value.requires_grad = param.requires_grad
                state_dict[param.name] = param_value
            else:
                state_dict[param.name] = param

        return state_dict
