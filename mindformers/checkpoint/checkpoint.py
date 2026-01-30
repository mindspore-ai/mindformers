# Copyright 2025 Huawei Technologies Co., Ltd
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
"""load/save checkpoint apis."""
import os
import json
import tempfile
from time import time
from typing import Callable, Union, Dict, Optional, Tuple, List
from dataclasses import dataclass

import threading
from multiprocessing import active_children

from mindspore import Tensor, Parameter, load_param_into_net
from mindspore.common import dtype as mstype
from mindspore.nn import Cell
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.communication import comm_func
from mindspore import save_checkpoint as ms_save_checkpoint

from mindformers.tools.logger import logger
from mindformers.checkpoint.reshard import ReshardLoader
from mindformers.utils.file_utils import is_publicly_accessible_path
from mindformers.utils.parallel_utils import barrier_world
from mindformers.tools.utils import (
    get_output_subpath,
    get_real_rank,
    set_safe_mode_for_file_or_dir,
    get_real_group_size
)
from mindformers.checkpoint.utils import (
    get_checkpoint_iter_dir,
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    get_latest_iteration_from_tracker,
    get_common_filename,
    check_checkpoints_dir_max_num,
    get_metadata_filename,
    verify_ckpt_valid,
    FileType,
    get_core_network
)
from mindformers.checkpoint.fully_parallel import BalancedSaveStrategy, apply_balance_shard_strategy
from mindformers.checkpoint.metadata import (
    save_metadata,
    get_total_params_file_mapping_info,
    get_metadata_of_checkpoint
)
from mindformers.checkpoint.sharded_tensor import (
    get_sharded_tensor_from_cell,
    get_cur_sharded_tensor,
    get_cur_sharded_tensor_after_balanced,
    get_param_redundancy_after_balanced
)
from mindformers.checkpoint.broadcast import single_parameter_broadcast


@dataclass
class CommonInfo:
    """
    Save/load common info for checkpoint.
    """
    epoch_num: int = None
    """The number of training epochs."""

    step_num: int = None
    """Training step number in current epoch."""

    global_step: int = None
    """Training step number in global epochs."""

    loss_scale: float = None
    """Magnification factor of gradients."""

    global_batch_size: int = None
    """The total batch size during multi-NPU training."""

    def save_common(self, common_filename: str):
        """
        Save common info to 'common.json'.

        Args:
            common_filename (str): The file path of 'common.json' to save.
        """
        logger.info(f"Saving common info to '{common_filename}'.")

        common_info_str = json.dumps(self.__dict__, ensure_ascii=False, indent=4)
        with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(common_filename), delete=False) as tmp_file:
            tmp_file.write(common_info_str)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())  # Ensure data is written to disk
            temp_filename = tmp_file.name
        os.replace(temp_filename, common_filename)
        set_safe_mode_for_file_or_dir(common_filename)

        logger.info(f"'common.json' successfully saved at: '{common_filename}'.")

    @classmethod
    def load_common(cls, common_filename: str):
        """
        Load common info from 'common.json'.

        Args:
            common_filename(str): The file path of 'common.json' to load.
        """
        logger.info(f"Loading common info from '{common_filename}'.")

        try:
            with open(common_filename, 'r', encoding='utf-8') as f:
                common_data = json.load(f)
            logger.info(f"'common.json' successfully loaded as:\n{common_data}")

            return cls(**common_data)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Can not find 'common.json' file at: '{common_filename}'.") from e

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON format failed: {e}") from e


class AsyncSaveManager:
    """
    Manager async save checkpoint process.
        1. Sync save process in all ranks and execute finalize functions before next save action.
        2. Check save process and execute finalize functions at the beginning of each step.
    """

    def __init__(self, async_save: Union[bool, str]):
        """
        Args:
            async_save (Union[bool, str]): Can be False, True(default 'thread'), 'thread', 'process'
        """
        self.async_save = async_save
        self.idx = 0
        self.finalize_fns = []
        self.is_finalized = True
        self.start_time = 0

    def add_finalize_fn(self, finalize_fn: Callable) -> None:
        """
        Adds a new finalize function to the manager.
        Finalize functions will be executed once after current save action.
        Finalize functions are reset when prepare_before_save is called.

        Args:
            finalize_fn (Callable): Function to add to the manager.
                This function will be called once after current save action.
        """
        logger.info(f"(idx:{self.idx})add finalize function")
        self.finalize_fns.append(finalize_fn)

    def prepare_before_save(self) -> None:
        """
        Prepare before a new save checkpoint action.
            1. Wait save process in all ranks and execute finalize functions
            2. Reset flags and finalize functions
        """
        logger.info(f"(idx:{self.idx})prepare before save")
        if not self.is_finalized:
            logger.info(f"(idx:{self.idx})previous save action is not finalized, wait finish synchronized...")
            self.maybe_finalize(wait_finish=True)

        self.is_finalized = False
        self.idx = self.idx + 1
        self.finalize_fns = []
        self.start_time = time()
        logger.info(f"(idx:{self.idx})prepare before save done")

    def maybe_finalize(self, wait_finish: bool = False) -> None:
        """
        Execute finalize functions if all ranks finish async save.

        Args:
            wait_finish (bool): If True, wait all async save process finish.
        """
        logger.info(f"(idx:{self.idx})self.is_finalized: {self.is_finalized}")
        if not self.is_finalized:
            start_time = time()
            is_alive = self.check_async_save_alive(wait_finish)
            logger.info(f"(idx:{self.idx})async_save: {self.async_save}, is_alive: {is_alive}, "
                        f"check is_alive cost time: {time() - start_time:.3f}s")
            start_time = time()
            is_all_done = self.sync_all_async_save_status(is_alive)
            logger.info(f"(idx:{self.idx})after all_reduce, is_all_done:{is_all_done}, "
                        f"cost time: {time() - start_time:.3f}s")

            if is_all_done:
                logger.info(f"(idx:{self.idx})execute finalize functions!")
                start_time = time()
                # Execute finalize functions
                for finalize_fn in self.finalize_fns:
                    finalize_fn()
                self.is_finalized = True
                logger.info(f"(idx:{self.idx})finalize functions done, cost time: {time() - start_time:.3f}s")
                logger.info(f"(idx:{self.idx})async save total time: {time() - self.start_time:.3f}s")

    def check_async_save_alive(self, wait_finish: bool = False) -> bool:
        """
        Check if current async save action is still running.

        Args:
            wait_finish (bool): If True, wait all async save process finish.

        Returns:
            A bool flag. True if current async save action is still running, False if it is finished.
        """
        if self.async_save is False:
            return False

        # Async process
        if self.async_save == "process":
            for process in active_children():
                if process.name == "asyn_save_ckpt":
                    if wait_finish:
                        process.join()
                        return False
                    return True
            return False

        # Async thread
        for thread in threading.enumerate():
            if thread.name == "asyn_save_ckpt":
                if wait_finish:
                    thread.join()
                    return False
                return True
        return False

    def sync_all_async_save_status(self, is_alive: int) -> bool:
        """Check if all ranks have completed async save checkpoint

        Args:
            is_alive (bool): if True, the current async save action is not completed

        Returns:
            A bool flag. True if all ranks are done, False if at least one rank is not completed.
        """
        if self.async_save is False:
            return True
        if get_real_group_size() == 1:
            return not is_alive

        ten = Tensor([is_alive], dtype=mstype.int8)
        ten, _ = comm_func.all_reduce(ten)

        return ten[0] == 0


def save_checkpoint(iteration: int, network: Cell, optimizer: Optimizer = None,
                    async_save_manager: AsyncSaveManager = None, common_info: CommonInfo = None,
                    keep_max_num: int = 5, user_prefix: str = None, save_checkpoint_path: str = None,
                    sharded_tensor_metas: Dict = None, remove_redundancy: bool = False):
    """
    Saves the current state of the training process,
        including the model, optimizer, and learning rate scheduler, to a checkpoint file.

    Args:
        iteration (int): The current training iteration step.
        network (Cell): The MindSpore model object to be saved.
        optimizer (Optimizer, optional): The optimizer object associated with the model. Defaults to None.
        async_save_manager (AsyncSaveManager, optional): The manager of async save if save weight in async way.
        common_info (CommonInfo): The instance of common info to save step_num, epoch_num, global_step and so on.
        keep_max_num (int): The maximum number of weights that can be stored.
        user_prefix (str): The prefix of user assign to use for the checkpoint file name.
        save_checkpoint_path (str): The user can specify the path to save the weights.
            If None, the default path is 'output_dir/checkpoint'.
            And 'output_dir' is configured in yaml and defaults to './output' in the execution script path.
        sharded_tensor_metas (Dict): The ShardedTensor metas of this network.
        remove_redundancy (bool): Whether to remove redundancy of saving checkpoint.
    """
    logger.info('....... Start to save checkpoint as new format .......')

    # Get the root path of all checkpoints to save.
    if save_checkpoint_path:
        checkpoints_root_path = os.path.realpath(save_checkpoint_path)
    else:
        checkpoints_root_path = get_output_subpath("checkpoint", append_rank=False)

    if not is_publicly_accessible_path(checkpoints_root_path):
        raise RuntimeError("The 'save_checkpoint_megatron_format' feature is not currently supported "
                           "in 'non-shared storage environments' with multiple hosts.")
    logger.info(f"The root path of saved checkpoints is: '{checkpoints_root_path}'.")

    # Generate current iteration saving path.
    cur_iter_checkpoint_dir = get_checkpoint_iter_dir(checkpoints_root_path, iteration)
    logger.info(f"At current iteration '{iteration}', the weight will be saved in: '{cur_iter_checkpoint_dir}'.")

    # Whether to use async save.
    use_async_save = async_save_manager is not None

    if get_real_rank() == 0:
        os.makedirs(cur_iter_checkpoint_dir, exist_ok=True)
        set_safe_mode_for_file_or_dir(checkpoints_root_path)
        set_safe_mode_for_file_or_dir(cur_iter_checkpoint_dir)
    barrier_world(f"Rank_0 to ensure path '{cur_iter_checkpoint_dir}' is exists.")
    # Fix cache coherency issues with shared storage.
    # Force refresh the disk cache of the current node to ensure that the path can be accessed correctly.
    os.listdir(os.path.dirname(checkpoints_root_path))

    # Prepare async save manager before save.
    def iter_finalize_func():
        """Save checkpoint finalize function."""
        tracker_filename = get_checkpoint_tracker_filename(checkpoints_root_path)
        logger.info(f"save checkpoint tracker file to {tracker_filename}")
        with open(tracker_filename, "w", encoding='utf-8') as f:
            f.write(str(iteration))
        set_safe_mode_for_file_or_dir(tracker_filename)
        if use_async_save:
            logger.info(f"successfully async saved checkpoint from step '{iteration}' to '{checkpoints_root_path}'.")
        else:
            logger.info(f"successfully sync saved checkpoint from step '{iteration}' to '{checkpoints_root_path}'.")

    if use_async_save:
        async_save_manager.prepare_before_save()
        if get_real_rank() == 0:
            async_save_manager.add_finalize_fn(iter_finalize_func)

    # Check if the number of saved folders has exceeded, and delete the oldest one.
    if get_real_rank() == 0:
        # NOTE: Currently only supports shared storage scenarios.
        check_checkpoints_dir_max_num(keep_max_num, checkpoints_root_path)
        # If the current iteration checkpoint directory be removed, raise an error to remind user
        # to check whether the file path for saving checkpoints is configured correctly.
        if not os.path.exists(cur_iter_checkpoint_dir):
            raise FileNotFoundError(f"Can not find current iteration checkpoint directory: "
                                    f"'{cur_iter_checkpoint_dir}'. Please check your configuration item "
                                    f"'save_checkpoint_path' under the 'CheckpointMonitor' in yaml, "
                                    f"to ensure that there is no weight left by other tasks under the path.")
    barrier_world("Rank_0 checking saved weights iteration num...")

    # Save model weight.
    logger.info("....... Start to save model weight .......")
    model_keys = network.parameters_dict().keys()
    start_save_ckpt_time = time()

    if remove_redundancy and sharded_tensor_metas is not None:
        remove_model_redundancy = BalancedSaveStrategy(
            network,
            user_prefix=user_prefix,
            checkpoint_path=checkpoints_root_path,
            filter_func=lambda x: x in list(model_keys),
            file_type=FileType.MODEL
        )
        remove_model_redundancy.save(iteration)
    else:
        model_ckpt_filename = get_checkpoint_name(
            cur_iter_checkpoint_dir, user_prefix, get_real_rank(), get_real_group_size(), FileType.MODEL
        )
        ms_save_checkpoint(
            network,
            model_ckpt_filename,
            async_save=use_async_save,
            format="safetensors"
        )
        logger.info(f"Model checkpoint successfully saved at '{model_ckpt_filename}.safetensors'.")

    # Save optimizer weight.
    if optimizer is not None:
        if remove_redundancy and sharded_tensor_metas is not None:
            # Optimizer weight remove redundancy.
            remove_optimizer_redundancy = BalancedSaveStrategy(
                optimizer,
                user_prefix=user_prefix,
                checkpoint_path=checkpoints_root_path,
                filter_func=lambda x: x not in list(model_keys),
                file_type=FileType.OPTIMIZER
            )
            remove_optimizer_redundancy.save(iteration)
        else:
            # Optimizer weight has redundancy.
            logger.warning("....... Start to save optimizer weight .......")
            optimizer_ckpt_filename = get_checkpoint_name(
                cur_iter_checkpoint_dir, user_prefix, get_real_rank(), get_real_group_size(), FileType.OPTIMIZER
            )
            ms_save_checkpoint(
                optimizer,
                optimizer_ckpt_filename,
                async_save=use_async_save,
                format="safetensors",
                choice_func=lambda x: x not in list(model_keys)
            )
            logger.info(f"Optimizer checkpoint successfully saved at '{optimizer_ckpt_filename}.safetensors'.")
    else:
        logger.warning("Optimizer weight will not be save!")

    # Save 'common.json'.
    if get_real_rank() == 0:
        logger.info("...... Start saving common info ......")
        start_save_common_info_time = time()

        common_filename = get_common_filename(checkpoints_root_path, iteration)
        common_info.save_common(common_filename)

        logger.info(f"The 'common.json' is saved at '{common_filename}'.")
        logger.info(f"Save common info cost time: {time() - start_save_common_info_time:.3f}s.")

    # Save 'metadata.json'.
    if not remove_redundancy:
        metadata_file_path = get_metadata_filename(checkpoints_root_path, iteration)
        save_metadata_json(sharded_tensor_metas, model_keys, user_prefix, metadata_file_path)

    # Save tracker file in sync save process.
    if not use_async_save:
        barrier_world("All ranks for sync save checkpoint.")
        logger.info("Rank_0 execute finalize func.")
        if get_real_rank() == 0:
            iter_finalize_func()
        logger.info(f"Save checkpoint cost time: {time() - start_save_ckpt_time:.3f}s.")


def save_metadata_json(sharded_tensor_metas, model_keys, user_prefix, metadata_file_path):
    """Saving metadata.json used `get_strategy_metadata` API."""
    if sharded_tensor_metas is not None:
        logger.info("...... Start saving metadata ......")
        if get_real_rank() == 0:
            param_file_mappings = get_total_params_file_mapping_info(sharded_tensor_metas, user_prefix, model_keys)
            save_metadata(sharded_tensor_metas, param_file_mappings, metadata_file_path)

        # Barrier here to ensure 'metadata.json' saved, then continue training.
        barrier_world("Rank_0 is saving 'metadata.json' ...")
        logger.info(f"The 'metadata.json' saved successfully at '{metadata_file_path}'.")
    else:
        logger.info("No need to save metadata.json for single card.")


def load_checkpoint(
        checkpoint: str,
        network: Cell,
        optimizer: Optional[Optimizer] = None,
        global_step: Optional[int] = None,
        balanced_load: bool = False,
        reshard_worker_num: int = 1,
) -> None:
    """
    Load weights in MindSpore Transformers format (self-trained weights).

    Use ReshardLoader to handle distributed resharding.
    Self-trained weights do not require templates; the source parameter names can be used directly.

    Args:
        checkpoint: Path to the checkpoint file or directory containing checkpoint files.
        network: The target network (Cell) to load parameters into (cannot be None).
        optimizer: Optional optimizer (Cell) to load optimizer states into.
        global_step: Optional initial global step value if not found in checkpoint.
        balanced_load: Whether to adopt the balanced loading mode.
        reshard_worker_num: Max number of workers to process reshard params.

    Raises:
        ValueError: If the input `network` is None.
            (Other exceptions may be raised by dependent functions for checkpoint validation/loading)
    """
    # Validate mandatory network parameter
    check_the_param_for_load_ckpt(checkpoint, network)

    # Determine checkpoint directory path
    checkpoint_dir = get_checkpoint_path(checkpoint)

    logger.info("..........Start Load Checkpoint..........")
    start_load_time = time()

    # Retrieve metadata from checkpoint files
    logger.info("..........Get Metadata of Checkpoint..........")
    src_sharded_tensor_metas, param_file_mappings = get_metadata_of_checkpoint(checkpoint_dir)

    # Define parameter filtering function
    def filter_func(param_name: str) -> bool:
        if optimizer:
            return "accu_grads" not in param_name
        return param_name in list(network.parameters_dict().keys())

    param_redundancy = None
    logger.info("..........Get Metadata of Network..........")
    if balanced_load:
        rank_id_to_sharded_tensors = apply_balance_shard_strategy(network, filter_func)
        dst_sharded_tensor_metas = get_cur_sharded_tensor_after_balanced(rank_id_to_sharded_tensors)
        param_redundancy = get_param_redundancy_after_balanced(rank_id_to_sharded_tensors)
    else:
        dst_sharded_tensor_metas = get_cur_sharded_tensor(network, filter_func) \
            if get_real_group_size() > 1 else get_sharded_tensor_from_cell(network, optimizer)

    # Load using ReshardLoader
    # Self-trained weights do not require a template;
    #   the parameter names are directly the same as `dst_sharded_tensor_metas`.
    reshard_loader = ReshardLoader(
        checkpoint_dir=checkpoint_dir,
        # Format of `dst_sharded_tensor_metas` is: `{param_name: ShardedTensor}`.
        dst_sharded_tensor_metas=dst_sharded_tensor_metas,
        # Format of `src_sharded_tensor_metas` is: `{param_name: [ShardedTensor, ...]}`.
        src_sharded_tensor_metas=src_sharded_tensor_metas,
        # Format of `param_file_mappings` is: `{(param_name, global_offset): [...]}`.
        param_file_mappings=param_file_mappings,
        reshard_worker_num=reshard_worker_num,
        template=None
    )

    # The first level of resharding, yields `{param_name: Parameter}`.
    state_dict = reshard_loader.load()

    # Handle global_step for optimizer if needed
    if optimizer and "global_step" not in state_dict:
        # Initialize global_step with default or from common.json
        logger.info(".....Get Global Step for Optimizer.....")
        if not global_step:
            common_file = os.path.join(checkpoint_dir, 'common.json')
            global_step = 0 if not os.path.exists(common_file) else CommonInfo.load_common(common_file).global_step

        state_dict["global_step"] = Parameter(
            Tensor([global_step], mstype.int32), name="global_step", requires_grad=False
        )

    logger.info("..........Loading State Dict into Network..........")
    # Load `state_dict` into network and optimizer.
    load_parameters(
        network,
        state_dict,
        optimizer,
        balanced_load=balanced_load,
        param_redundancy=param_redundancy
    )

    logger.info("..........Loading Checkpoint Finished..........")
    logger.info(f"..........Loading Time Cost: {time() - start_load_time:.5f} s..........")


def load_hf_checkpoint(
        pretrained_model_dir: str,
        network: Cell,
        balanced_load: bool = False,
        reshard_worker_num: int = 1,
) -> None:
    """
    Load HuggingFace format weights.

    Uses a two-step processing approach:
        1. Reshard: Uses ReshardLoader to handle distributed slicing;
        2. Convert: Uses Template to handle QKV concatenation, Stack, and other conversions.

    Design notes:
        - ReshardLoader uses template.get_mf_name() to complete HF→MF parameter name mapping
        - WeightTemplate.get_mf_state_dict() completes the final weight conversion (e.g., QKV concatenation)
    """
    logger.info("..........Start Load Checkpoint..........")
    start_load_time = time()

    # 1. Get template
    core_network = get_core_network(network)
    template = core_network.template
    if template is None:
        raise ValueError(
            "The template of current model is None. Please check:"
            "\n  1. Whether the conversion template of model has been registered using `register_hf_weight_template`."
            "\n    You need to navigate to the entry point of the model's training code file "
            "(such as `TrainingQwen3ForCausalLM` in `modeling_qwen3_train.py`), "
            "and add the `@register_hf_weight_template` above its `__init__` function "
            "to register the conversion template."
            "\n  2. Whether the current model has added weight conversion rules in the model's utils file, such as:"
            "\n    weight_converters = ["
            "\n        RenameConvertOp("
            "\n            hf_names='model.embed_tokens.weight',"
            "\n            mf_names='embedding.word_embeddings.weight'"
            "\n        ),"
            "\n        ..."
            "\n    ]"
        )

    # 2. Get target metadata
    def filter_func(param_name: str) -> bool:
        return param_name in list(network.parameters_dict().keys())

    param_redundancy = None
    logger.info("..........Get Metadata of Network..........")
    if balanced_load:
        rank_id_to_sharded_tensors = apply_balance_shard_strategy(network, filter_func)
        dst_sharded_tensor_metas = get_cur_sharded_tensor_after_balanced(rank_id_to_sharded_tensors)
        param_redundancy = get_param_redundancy_after_balanced(rank_id_to_sharded_tensors)
    else:
        dst_sharded_tensor_metas = get_cur_sharded_tensor(network, filter_func) \
            if get_real_group_size() > 1 else get_sharded_tensor_from_cell(network)

    # 3. Get source metadata (built from HF weights)
    src_sharded_tensor_metas, param_file_mappings = get_metadata_of_checkpoint(pretrained_model_dir)

    # 4. Reshard:
    # Pass template for parameter name mapping, and ReshardLoader internally will:
    #   (1) Pre-build bidirectional mapping:
    #      - src_to_dst_mapping: {src_name: dst_name}, e.g., {q_proj: qkv, k_proj: qkv, v_proj: qkv}
    #      - dst_to_src_mapping: {dst_name: [src_names]}, e.g., {qkv: [q_proj, k_proj, v_proj]}
    #   (2) Iterate through dst_metas, for each target parameter, get all related source parameters via dst_to_src_mapping
    #   (3) For each source parameter (e.g., q, k, v), calculate offset and perform slicing separately
    reshard_loader = ReshardLoader(
        checkpoint_dir=pretrained_model_dir,
        dst_sharded_tensor_metas=dst_sharded_tensor_metas,  # {mf_param_name: ShardedTensor}
        src_sharded_tensor_metas=src_sharded_tensor_metas,  # {hf_param_name: [ShardedTensor]}
        param_file_mappings=param_file_mappings,  # {(hf_param_name, global_offset): [...]}
        reshard_worker_num=reshard_worker_num,
        template=template  # HF weights need template for parameter name mapping
    )

    # Get Reshard output: `{hf_param_name: tensor}`.
    # Returned keys are HF original parameter names (e.g., q_proj.weight, k_proj.weight, v_proj.weight)
    # Each source parameter has been sliced (only loading the part needed by current card)
    reshard_output = reshard_loader.load()

    # 5. Convert:
    # Use template.get_mf_state_dict() to convert `{hf_param_name: weight}` to `{mf_param_name: Parameter}.`
    # Internally iterates through parameters, calling add_hf_weight() for conversion:
    #   - Single-source weights (e.g., embed_tokens): directly rename
    #   - Multi-source weights (e.g., QKV): temporarily store q, k, v, then concatenate after all are ready
    # Since Reshard stage has completed slicing, no need to slice again during conversion
    state_dict = template.get_mf_state_dict(reshard_output)

    # 6. Load into network
    logger.info("..........Loading State Dict into Network..........")
    # Load state dictionary into network and optimizer
    load_parameters(
        network,
        state_dict,
        balanced_load=balanced_load,
        param_redundancy=param_redundancy
    )

    logger.info("..........Loading Hugging Face Checkpoint Finished..........")
    logger.info(f"..........Loading Time Cost: {time() - start_load_time:.5f} s..........")


def check_the_param_for_load_ckpt(checkpoint: str, network: Cell):
    """Check the params passing in `load_checkpoint` method is legal."""
    if network is None:
        raise ValueError("The 'network' cannot be None - a target network is required for loading.")

    if not os.path.exists(checkpoint):
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")


def load_parameters(
        network: Cell,
        state_dict: Dict[str, Parameter],
        optimizer: Optional[Cell] = None,
        state_dict_opt: Optional[Dict[str, Parameter]] = None,
        balanced_load: Optional[bool] = False,
        param_redundancy: Optional[Dict[Tuple, str]] = None
):
    """
    Loads parameters into a MindSpore network and optional optimizer, with support for redundant parameter handling.

    This function separates network-specific and optimizer-specific parameters from input state dictionaries,
    loads them into their respective components, and provides detailed logging of unloaded parameters. When
    `balanced_load` is enabled, it leverages shard balancing and parameter broadcasting to eliminate redundant
    parameter storage across ranks, improving memory efficiency in distributed training scenarios.

    Core workflow:
    1. Initialize optimizer state dictionary if not provided.
    2. (If balanced load enabled) Generate parameter redundancy map via shard balancing if not explicitly provided.
    3. Separate parameters from the main state dict into network-specific and optimizer-specific (state_dict_opt).
    4. Load network parameters, track unloaded parameters, and filter out cache-related entries from unloaded logs.
    5. (If balanced load enabled) Broadcast redundant parameters across ranks to ensure consistency.
    6. Load optimizer parameters (if optimizer and state_dict_opt are provided) and apply balanced load if enabled.
    7. Log detailed information about loaded/unloaded parameters for both network and optimizer.

    Args:
        network (Cell): Target MindSpore Network Cell to load parameters into. Must be a valid Cell instance.
        state_dict (Dict[str, Parameter]): Dictionary containing network parameters to load. Keys must match
            parameter names in the network (or optimizer, for parameters to be redirected).
        optimizer (Optional[Cell]): Optional MindSpore Optimizer Cell to load optimizer-specific parameters into.
            If provided, must be a valid Cell instance.
        state_dict_opt (Optional[Dict[str, Parameter]]): Optional dictionary containing optimizer parameters to load.
            Initialized as an empty dict if not provided.
        balanced_load (Optional[bool]): Whether to enable balanced loading with redundant parameter elimination.
            When True, uses `apply_balance_shard_strategy` to identify redundant parameters and
            `single_parameter_broadcast` to synchronize values across ranks. Defaults to False.
        param_redundancy (Optional[Dict[Tuple[int, ...], List[str]]]): Precomputed mapping of redundant rank groups
            (tuples of rank IDs) to lists of parameter keys. Only used if `balanced_load` is True; if not provided,
            generated dynamically via `apply_balance_shard_strategy`. Defaults to None.

    Raises:
        ValueError: If `network` is not a valid MindSpore Cell, `state_dict` is invalid (e.g., not a dict),
            `state_dict_opt` is provided but not a dict, or `optimizer` is provided but not a valid Cell.
        RuntimeError: If parameter loading fails due to mismatched keys or invalid parameter types (propagated from
            `load_param_into_net`).
    """

    def split_state_dict(network, state_dict, optimizer, state_dict_opt):
        """split state dict"""
        network_param_names = set(network.parameters_dict().keys())
        optimizer_param_names = set(optimizer.parameters_dict().keys()) if optimizer else set()
        for param_name in list(state_dict.keys()):
            if param_name not in network_param_names and param_name in optimizer_param_names and \
                    param_name not in state_dict_opt:
                state_dict_opt[param_name] = state_dict.pop(param_name)
        return network_param_names, optimizer_param_names, state_dict, state_dict_opt

    def print_not_load_info(param_list: List, param_info: str):
        if not param_list:
            logger.info(f"All {param_info} are loaded.")
            return

        logger.info(f"{param_info} not loaded:")
        for p in param_list:
            logger.info(f"  - {p}")

    state_dict_opt: Dict[str, Parameter] = {} if not state_dict_opt else state_dict_opt

    # Separate network and optimizer parameters
    if balanced_load and param_redundancy is None:
        rank_id_to_sharded_tensors = apply_balance_shard_strategy(network)
        param_redundancy = get_param_redundancy_after_balanced(rank_id_to_sharded_tensors)

    network_param_names, _, state_dict, state_dict_opt = \
        split_state_dict(network, state_dict, optimizer, state_dict_opt)

    # Load parameters into network
    logger.debug(f"Network state_dict keys: {list(state_dict.keys())}")
    param_not_load, ckpt_not_load = load_param_into_net(network, state_dict)
    if balanced_load:
        param_loaded = {param_name for param_name in state_dict if param_name not in ckpt_not_load}
        single_parameter_broadcast(network, param_redundancy, param_not_load, param_loaded)
    # Filter out cache and optimizer parameters from unloaded list
    param_not_load = [p for p in param_not_load if "key_cache" not in p and "value_cache" not in p]
    print_not_load_info(param_not_load, "Network parameters")
    print_not_load_info(ckpt_not_load, "Checkpoint weights")

    # Load parameters into optimizer if available
    if optimizer and state_dict_opt:
        logger.debug(f"Optimizer state_dict keys: {list(state_dict_opt.keys())}")
        param_not_load_opt, ckpt_not_load_opt = load_param_into_net(optimizer, state_dict_opt, strict_load=True)
        if balanced_load:
            param_loaded_opt = {param_name for param_name in state_dict_opt if param_name not in ckpt_not_load_opt}
            single_parameter_broadcast(optimizer, param_redundancy, param_not_load_opt, param_loaded_opt)

        param_not_load_opt = [p for p in param_not_load_opt if p not in network_param_names]
        print_not_load_info(param_not_load_opt, "Optimizer parameters")
        print_not_load_info(ckpt_not_load_opt, "Optimizer weights")


def get_checkpoint_path(checkpoint: str) -> str:
    """
    Retrieve a valid checkpoint directory.

    This function locates the latest checkpoint iteration from a training checkpoint
    directory, validates its existence and suitability, and returns the path to be used.

    Args:
        checkpoint: Base directory containing training checkpoints.

    Returns:
        The path to the valid checkpoint directory.

    Raises:
        ValueError: If the base checkpoint directory doesn't exist or the found checkpoint is invalid.
    """
    if not checkpoint:
        return ""

    if not os.path.exists(checkpoint):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")

    if not os.path.isdir(checkpoint):
        raise ValueError(f"Checkpoint path is not a directory: {checkpoint}")

    # Check all need checkpoint files if load Hugging Face checkpoint
    hf_index_json = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.exists(hf_index_json):
        with open(hf_index_json, 'r', encoding='utf-8') as f:
            index_json = json.load(f)
        if isinstance(index_json, dict):
            weight_map = index_json['weight_map'] if 'weight_map' in index_json else index_json
        else:
            raise ValueError(f"Format of '{hf_index_json}' is illegal!")

        sf_file_list = set(weight_map.values())
        not_exist_file = [
            f
            for f in sf_file_list
            if not os.path.isfile(os.path.join(checkpoint, f))
        ]
        not_exist_file.sort()
        if not_exist_file:
            raise ValueError(f"The files '{not_exist_file}' do not exist in `{checkpoint}`.")
        return checkpoint

    tracker_filename = get_checkpoint_tracker_filename(checkpoint)
    if os.path.exists(tracker_filename):
        iteration = get_latest_iteration_from_tracker(checkpoint)
        checkpoint = get_checkpoint_iter_dir(checkpoint, iteration)

    verify_ckpt_valid(checkpoint)
    logger.info(f"Get checkpoint: {checkpoint}")

    return checkpoint
