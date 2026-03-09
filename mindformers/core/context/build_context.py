# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Build context."""

import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Union

import mindspore
import mindspore.context as ms_context
from mindspore.device_context.ascend import op_precision
import psutil

import mindformers
from mindformers.core.config_args import (
    ContextConfig,
    MFContextConfig,
    ParallelContextConfig,
)
from mindformers.core.context.parallel import ParallelOperator
from mindformers.core.context.validators import RunMode, execute_validator
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerConfig
from mindformers.tools.utils import (
    MODE,
    get_output_subpath,
    check_in_dynamic_cluster,
)
from mindformers.version_control import (
    check_tft_valid,
    set_ms_deterministic,
    check_set_cpu_affintiy_bind_file_support,
)

_MS_CONTEXT_DEFAULTS = {
    "device_id": "0",
    "mode": "PYNATIVE_MODE",
    "max_device_memory": "1024GB",
}


class _MSContextApplier:
    """
    Applier for MS context configuration. Routes params to handlers,
    applies them to MindSpore APIs, and caches values.
    """

    _migrate = (
        ("device", ("device_id", "device_target"), "_apply_device"),
        (
            "memory",
            (
                "max_device_memory",
                "mempool_block_size",
                "memory_optimize_level",
            ),
            "_apply_memory",
        ),
        ("env", ("save_graphs", "save_graphs_path"), "_apply_env"),
        ("extra", ("max_call_depth",), "_apply_extra"),
        ("ascend", ("ascend_config",), "_apply_ascend"),
        (
            "set_context_remaining",
            ("mode", "jit_config", "ascend_config", "dataset_broadcast_opt_level"),
            "_apply_set_context_remaining",
        ),
    )

    @staticmethod
    def _get_param(params, key, instance=None):
        """
        Get value from params; if not in params, try instance attribute;
        otherwise None (or _MS_CONTEXT_DEFAULTS if essential).
        """
        if key in params:
            return params[key]
        if instance is not None and hasattr(instance, key):
            return getattr(instance, key)
        return _MS_CONTEXT_DEFAULTS.get(key)

    def __init__(self):
        # Build routing tables
        self._param_to_category = {}
        self._category_to_handler = {}
        for cat, keys, handler in self._migrate:
            for k in keys:
                self._param_to_category[k] = cat
            if handler is not None:
                self._category_to_handler[cat] = handler
        self.dataset_broadcast_opt_level = 0

    def get_category(self, param_key):
        """Resolve category for a context param_key (kwargs key)."""
        category = self._param_to_category.get(param_key)
        if category is not None:
            return category
        for cat, keys, _ in self._migrate:
            if param_key in keys:
                return cat
        return None

    def get_handler_name(self, category):
        """Return handler method name for category, or None."""
        return self._category_to_handler.get(category)

    def split_params(self, kwargs):
        """
        Split kwargs into groups by category;
        unknown params go to 'set_context_excluded'.
        """
        groups = defaultdict(dict)
        for param_key, value in kwargs.items():
            category = self.get_category(param_key)
            if category is not None:
                groups[category][param_key] = value
            else:
                groups["set_context_excluded"][param_key] = value
        return groups

    def dispatch_applies(self, groups):
        """
        Run each group's handler; skip 'set_context_excluded'
        (unknown params) and empty groups.
        """
        for category, params in groups.items():
            if category == "set_context_excluded" or not params:
                continue
            handler_name = self.get_handler_name(category)
            if handler_name:
                handler = getattr(self, handler_name)
                handler(params)

    def _apply_device(self, params):
        """
        Apply device_id, device_target via mindspore.set_device()
        and cache attributes.
        """
        device = {}
        target = self._get_param(params, "device_target", self) or os.getenv(
            "DEVICE_TARGET", None
        )
        dev_id = params.get("device_id")
        if dev_id:
            device['device_id'] = dev_id
        if target:
            device['device_target'] = target
            try:
                mindspore.set_device(**device)
            except RuntimeError as e:
                error_msg = str(e)
                if "The 'mindspore.set_device' can not be modified" in error_msg:
                    logger.warning(f"Fail to set device: {error_msg}.")
                elif "The runtime has been initialized" in error_msg:
                    logger.warning(f"Fail to set device: {error_msg}.")
                else:
                    raise e
            self.device_target = target
            if dev_id is not None:
                self.device_id = dev_id

    def _apply_memory(self, params):
        """
        Apply max_device_memory etc. via mindspore.runtime.set_memory()
        and cache attributes.
        """
        max_sz = self._get_param(params, "max_device_memory", self)
        if not max_sz:
            return
        mempool = params.get("mempool_block_size") or getattr(
            self, "mempool_block_size", None
        )
        opt_level = params.get("memory_optimize_level") or getattr(
            self, "memory_optimize_level", None
        )
        memory_kwargs = {"max_size": max_sz}
        if mempool is not None:
            memory_kwargs["increase_size"] = mempool
        if opt_level is not None:
            memory_kwargs["optimize_level"] = opt_level
        try:
            mindspore.runtime.set_memory(**memory_kwargs)
        except RuntimeError as err:
            if "can not be set repeatedly" in str(err):
                logger.warning("Memory config already set: %s", err)
            else:
                raise
        self.max_device_memory = max_sz
        if "mempool_block_size" in params:
            self.mempool_block_size = params["mempool_block_size"]
        if "memory_optimize_level" in params:
            self.memory_optimize_level = params["memory_optimize_level"]

    def _apply_env(self, params):
        """
        Apply save_graphs, save_graphs_path via environment variables
        and cache attributes.
        """
        env_map = {
            "save_graphs": (
                "MS_DEV_SAVE_GRAPHS",
                lambda val: str(int(bool(val))),
            ),
            "save_graphs_path": ("MS_DEV_SAVE_GRAPHS_PATH", str),
        }
        for param_key, value in params.items():
            if param_key in env_map:
                env_name, formatter = env_map[param_key]
                os.environ[env_name] = formatter(value)
                setattr(self, param_key, value)

    def _apply_extra(self, params):
        """
        Apply max_call_depth via mindspore.set_recursion_limit()
        and cache attributes.
        """
        value = params.get("max_call_depth")
        if value is not None:
            mindspore.set_recursion_limit(int(value))
            self.max_call_depth = int(value)

    def _apply_ascend(self, params):
        """
        Apply ascend_config: extract and process precision_mode via
        new API, then remove it from params. Remaining ascend_config
        (e.g., parallel_speed_up_json_path) will be handled by
        _apply_set_context_remaining.
        """
        ascend_config = params.get("ascend_config")
        if not ascend_config or not isinstance(ascend_config, dict):
            return

        precision_mode = ascend_config.get("precision_mode")
        if precision_mode is not None:
            op_precision.precision_mode(precision_mode)
            ascend_config.pop("precision_mode")

    @staticmethod
    def _read_speed_up_json(json_path):
        """Read parallel_speed_up JSON file; return dict or {} on error."""
        if not json_path or not os.path.isfile(json_path):
            return {}
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError, TypeError) as e:
            logger.debug("Could not read speed_up json from %s: %s", json_path, e)
            return {}

    @staticmethod
    def _write_speed_up_temp_file(dataset_broadcast_opt_level):
        """Create a temp JSON file with dataset_broadcast_opt_level; return its path."""
        data = {"dataset_broadcast_opt_level": dataset_broadcast_opt_level}
        fd, path = tempfile.mkstemp(suffix=".json", prefix="parallel_speed_up_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError:
            os.close(fd)
            raise
        return path

    def _apply_set_context_remaining(self, params):
        """
        Apply mode, jit_config, ascend_config via ms_context.set_context() and cache attributes.

        dataset_broadcast_opt_level / parallel_speed_up:
        - If params has dataset_broadcast_opt_level: create a temp file with that value and set
          ascend_config.parallel_speed_up_json_path to it for set_context.
        - Else if params has ascend_config with parallel_speed_up_json_path: read that file and
          cache dataset_broadcast_opt_level from it.

        dataset_broadcast_opt_level is never passed directly to set_context; MS reads it from
        the file pointed by ascend_config.parallel_speed_up_json_path.
        """
        if not params:
            return

        ascend_config = params.get("ascend_config")
        if ascend_config and not isinstance(ascend_config, dict):
            ascend_config = None
        has_ascend_config = ascend_config is not None

        json_path = None
        if "dataset_broadcast_opt_level" in params:
            self.dataset_broadcast_opt_level = int(params["dataset_broadcast_opt_level"])
            json_path = self._write_speed_up_temp_file(self.dataset_broadcast_opt_level)
        elif has_ascend_config:
            json_path = ascend_config.get("parallel_speed_up_json_path")
            if json_path:
                speed_up = self._read_speed_up_json(json_path)
                if speed_up.get("dataset_broadcast_opt_level") is not None:
                    self.dataset_broadcast_opt_level = int(speed_up["dataset_broadcast_opt_level"])

        if ascend_config:
            ascend_config_for_ms = dict(ascend_config)
        elif "dataset_broadcast_opt_level" in params:
            ascend_config_for_ms = {"parallel_speed_up_json_path": json_path}
        else:
            ascend_config_for_ms = {}

        params_for_ms = {k: params[k] for k in ("mode", "jit_config") if k in params}
        if ascend_config_for_ms:
            params_for_ms["ascend_config"] = ascend_config_for_ms
        if params_for_ms:
            ms_context.set_context(**params_for_ms)

        for key in ("mode", "jit_config", "ascend_config"):
            if key in params:
                setattr(self, key, params[key])


class Context:
    """The wrapper of mindformer context and mindspore context."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config=None):
        """Build context."""
        if not hasattr(self, "_initialized"):
            self.rank_id = 0
            self.device_num = 1
            self.config = config if config is not None else MindFormerConfig()
            self.mf_ctx_opr = MFContextOperator(self.config)
            self.ms_ctx_opr = MSContextOperator(self.config)
            self.parallel_opr = ParallelOperator(self.config)
            if check_tft_valid() and (
                "ARF:1" in os.getenv("MS_ENABLE_TFT", "")
            ):
                # pylint: disable=C0415
                from mindspore.utils import _tft_handler

                _tft_handler.init(config=self.config)
            if self.config.use_parallel:
                self.rank_id, self.device_num = (
                    self.parallel_opr.init_communication()
                )
            set_cpu_affinity(self.rank_id, self.device_num)
            _setup_affinity(self.config.get("context", {}))
            self._initialized = True

    @classmethod
    def is_exists(cls):
        """Check if singleton Context exists."""
        return cls._instance is not None

    @classmethod
    def reset_instance(cls):
        if cls._instance:
            cls._instance = None

    def set_mf_ctx_run_mode(self, run_mode):
        if run_mode is not None:
            members = [k.lower() for k, _ in RunMode.__members__.items()]
            if run_mode in members:
                self.mf_ctx_opr.run_mode = run_mode
            else:
                raise ValueError(
                    f'Fail to set run_mode, Invalid value. '
                    f'Expected one of {members}, got {run_mode}'
                )


@dataclass
class MSContextOperator:
    """The wrapper of mindspore context operation."""

    def __init__(self, config):
        self.config = config
        self.context_applier = _MSContextApplier()
        ms_kwargs = self._handle_data()
        self.set_context(**ms_kwargs)

    def _handle_data(self):
        """Get the valid ms config."""
        ctx = self.config.get('context', {})
        ms_ctx = {
            'mode': MODE.get(ctx.get('mode', _MS_CONTEXT_DEFAULTS["mode"])),
            "device_id": ctx.get("device_id", int(os.getenv("DEVICE_ID", _MS_CONTEXT_DEFAULTS["device_id"]))),
            "max_device_memory": ctx.get("max_device_memory", _MS_CONTEXT_DEFAULTS["max_device_memory"]),
        }
        self._set_device_id(ctx, ms_ctx)
        self._set_save_graphs_path(ctx, ms_ctx)
        self._set_predict_jit_config(ctx, ms_ctx)
        self._set_runtime_kernel_launch_group()
        return self._remove_mf_keys({**ctx, **ms_ctx})

    @staticmethod
    def _set_runtime_kernel_launch_group():
        """Set the parameters of kernel_launch_group"""
        kernel_launch_group = {}
        env_kernel_launch_group = os.getenv(
            "EXPERIMENTAL_KERNEL_LAUNCH_GROUP", None
        )
        if env_kernel_launch_group is not None:
            logger.info("........ Enable kernel_launch_group ........")
            pairs = env_kernel_launch_group.split(',')
            for pair in pairs:
                key, val = pair.split(':')
                kernel_launch_group[key] = val
            thread_num = int(kernel_launch_group.get('thread_num', 2))
            kernel_group_num = int(
                kernel_launch_group.get("kernel_group_num", 8)
            )
            mindspore.runtime.set_kernel_launch_group(
                thread_num=thread_num, kernel_group_num=kernel_group_num
            )

    def _set_device_id(self, ctx, ms_ctx):
        if self.config.use_parallel and check_in_dynamic_cluster():
            # for dynamic cluster, we should not set device id in context.
            ctx.pop('device_id', None)
            ms_ctx.pop('device_id', None)

    @staticmethod
    def _set_save_graphs_path(ctx, ms_ctx):
        if ctx.get('save_graphs'):
            ms_ctx['save_graphs_path'] = ctx.get(
                'save_graphs_path',
                get_output_subpath("debug/graphs_info", append_rank=False),
            )

    def _set_predict_jit_config(self, ctx, ms_ctx):
        """
        Parse jit_level and infer_boost from context (including
        jit_config), validate and pass via set_context(jit_config=...).
        """
        run_mode = self.config.get("run_mode")
        if run_mode is None or RunMode(run_mode) not in [
            RunMode.PREDICT,
            RunMode.EVAL,
        ]:
            return
        jit_level = ctx.get("jit_level", "O0")
        infer_boost = ctx.get("infer_boost", "on")
        jit_config = ctx.get("jit_config")
        if jit_config is not None:
            jit_level = jit_config.get("jit_level") or jit_level
            infer_boost = jit_config.get("infer_boost") or infer_boost
        logger.info(
            "Predict context config, jit_level: "
            f"{jit_level}, infer_boost: {infer_boost}"
        )
        if jit_level == "O1":
            raise ValueError("jit_level O1 is not supported in predict mode.")
        if jit_level == "O2" and infer_boost == "on":
            raise ValueError(
                "Only infer_boost must set off to "
                "set jit_level to O2 in predict mode."
            )
        ms_ctx['jit_config'] = {
            'jit_level': jit_level,
            'infer_boost': infer_boost,
        }

    @staticmethod
    def _remove_mf_keys(ctx_config):
        mf_keys = MFContextConfig.get_supported_kwargs()
        return {
            key: value
            for key, value in ctx_config.items()
            if key not in mf_keys
        }

    def set_context(self, **kwargs):
        """
        Set MS runtime context.
        Migrate deprecated set_context params to new APIs; the rest still go through set_context.
        See https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_context.html
        """
        if not kwargs:
            return
        groups = self.context_applier.split_params(kwargs)
        self.context_applier.dispatch_applies(groups)
        MSContextOperator._warn_set_context_excluded(
            groups.get("set_context_excluded", {})
        )

    @staticmethod
    def _warn_set_context_excluded(set_context_excluded):
        """Log info for context params not in whitelist; suggest calling mindspore APIs or MF config instead."""
        if not set_context_excluded:
            return
        if "deterministic" in set_context_excluded:
            logger.info(
                "context.deterministic is no longer set via set_context; "
                "use MF config train_precision_sync or infer_precision_sync for equivalent behavior."
            )
        others = [key for key in set_context_excluded if key != "deterministic"]
        if others:
            logger.info(
                "The following context options did not take effect via set_context; "
                "please refer to the official MindSpore documentation for corresponding APIs: %s",
                others,
            )

    def get_context(self, attr_key):
        """
        Get context attribute from applier cache.
        Returns default for essential params (device_target, device_id, mode, max_device_memory) if not set.
        """
        if attr_key == "device_target":
            return getattr(self.context_applier, attr_key, "Ascend")
        return getattr(self.context_applier, attr_key)


@dataclass
class MFContextOperator(MFContextConfig):
    """The wrapper of mindformers context operation."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config):
        if not hasattr(self, "_initialized"):
            self.config = config
            supported_kwargs = self._handle_data()
            logger.debug('MFContextConfig load configs: %s', supported_kwargs)
            super().__init__(**supported_kwargs)
            use_past = self.config.get_value(
                "model.model_config.use_past", False
            )
            if not hasattr(self, 'train_precision_sync'):
                self.train_precision_sync = None
            if not hasattr(self, 'infer_precision_sync'):
                self.infer_precision_sync = None
            self.set_env(use_past)
            self._initialized = True

    @classmethod
    def get_mf_ctx_instance(cls):
        """Check if singleton Context exists."""
        if cls._instance:
            return cls.__new__(cls)
        return None

    @classmethod
    def reset_instance(cls):
        if cls._instance:
            cls._instance = None

    def _handle_data(self):
        ctx_config = self.config.get('context', {})
        extra_config = self._get_first_level_config()
        return self.filter_kwargs(**{**extra_config, **ctx_config})

    def _get_first_level_config(self):
        return {
            k: v
            for k, v in self.config.items()
            if k in MFContextOperator.get_supported_kwargs()
        }

    def _call_ms_deterministic(self, deterministic):
        """Call mindspore set_deterministic function and handle result."""
        try:
            set_ms_deterministic(deterministic)
            ms_deterministic = self.config.get_value('context.deterministic')
            if ms_deterministic is not None:
                self.config.context.pop('deterministic')
                logger.warning(
                    'The deterministic in context has been unset when '
                    'train_precision_sync or infer_precision_sync was set.'
                )
        except RuntimeError as e:
            msg = "The 'mindspore.set_deterministic' can not be set repeatedly."
            if str(e) == msg:
                logger.warning(
                    "mindspore.set_deterministic has been set, "
                    "can not be set repeatedly. Key environment variables: "
                    f"HCCL_DETERMINISTIC: {os.getenv('HCCL_DETERMINISTIC')}, "
                    f"TE_PARALLEL_COMPILER: {os.getenv('TE_PARALLEL_COMPILER')}, "
                    f"CUSTOM_MATMUL_SHUFFLE: {os.getenv('CUSTOM_MATMUL_SHUFFLE')}, "
                    f"LCCL_DETERMINISTIC: {os.getenv('LCCL_DETERMINISTIC')}"
                )
                return '', ''
            raise e
        if deterministic:
            return 'off', '1'
        return 'on', '0'

    def _get_precision_env(self):
        """Set deterministic computing and get relative env variable."""
        custom_matmul_shuffle = os.getenv('CUSTOM_MATMUL_SHUFFLE')
        lccl_deterministic = os.getenv('LCCL_DETERMINISTIC')
        run_mode = (
            self.run_mode if hasattr(self, "run_mode") else None
        )
        if (
            run_mode
            in (
                RunMode.TRAIN.value,
                RunMode.FINETUNE.value,
                RunMode.PREDICT_WITH_TRAIN_MODEL.value,
            )
            and self.train_precision_sync is not None
        ):
            _, _ = self._call_ms_deterministic(self.train_precision_sync)

        if (
            run_mode == RunMode.PREDICT.value
            and self.infer_precision_sync is not None
        ):
            shuffle, lccl = self._call_ms_deterministic(
                self.infer_precision_sync
            )
            if shuffle and lccl:
                if custom_matmul_shuffle != shuffle:
                    logger.warning(
                        f"'CUSTOM_MATMUL_SHUFFLE' is set to '{shuffle}' "
                        f"because infer_precision_sync is {self.infer_precision_sync}."
                    )

                if lccl_deterministic != lccl:
                    logger.warning(
                        f"'LCCL_DETERMINISTIC' is set to '{lccl}' "
                        f"because infer_precision_sync is {self.infer_precision_sync}."
                    )
                custom_matmul_shuffle = shuffle
                lccl_deterministic = lccl

        return {
            'CUSTOM_MATMUL_SHUFFLE': custom_matmul_shuffle or 'on',
            'LCCL_DETERMINISTIC': lccl_deterministic or '0',
        }

    def set_env(self, use_past=False):
        """Update environment variables."""

        env = self._get_precision_env()

        env['MS_ENABLE_GRACEFUL_EXIT'] = '1' if self.use_graceful_exit else '0'

        run_mode = (
            self.run_mode if hasattr(self, "run_mode") else None
        )
        use_legacy = self.config.get_value('use_legacy', True)
        infer_flag = not use_legacy or use_past
        if (
            run_mode is not None
            and RunMode(run_mode) in [RunMode.PREDICT, RunMode.EVAL]
            and infer_flag
        ):
            ms_alloc_conf = os.environ.get('MS_ALLOC_CONF', 'enable_vmm:False')
            cpu_affinity = os.environ.get('CPU_AFFINITY', 'True')
            ms_internal_disable_custom_kernel_list = ""
            ms_internal_disable_custom_kernel_list = os.environ.get(
                'MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST', 'PagedAttention'
            )
            env['MS_ALLOC_CONF'] = ms_alloc_conf
            env['CPU_AFFINITY'] = cpu_affinity
            env['MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST'] = (
                ms_internal_disable_custom_kernel_list
            )
            env["RUN_MODE"] = run_mode

        if (
            run_mode is not None
            and RunMode(run_mode) in [RunMode.TRAIN, RunMode.FINETUNE]
            and not use_legacy
        ):
            env["MS_DEV_JIT_SYNTAX_LEVEL"] = "0"

        os.environ.update({k: v for k, v in env.items() if v is not None})
        logger.debug(
            "Environment variables to be set in mindformers context: %s", env
        )

    def set_context(self, **kwargs):
        """Set mf context value according to the input key words."""
        supported_kwargs = self.filter_kwargs(**kwargs)
        for k, v in supported_kwargs.items():
            setattr(self, k, v)

    def get_context(self, attr_key):
        """Get mf context value according to the input key."""
        return getattr(self, attr_key, None)


def _setup_affinity(context_config):
    """Set CPU affinity from context config."""
    has_cpu_list = "affinity_cpu_list" in context_config
    has_config = "affinity_config" in context_config
    if has_config:
        if has_cpu_list:
            logger.warning(
                "affinity_cpu_list will be removed in the near future, affinity_config is taking effect."
            )
        set_ms_affinity(context_config.get("affinity_config", {}), None)
    elif has_cpu_list:
        logger.warning(
            "affinity_cpu_list will be removed in the near future, please use affinity_config instead."
        )
        affinity_cpu_list = context_config.get("affinity_cpu_list", {})
        if not isinstance(affinity_cpu_list, dict):
            logger.warning(
                "custom bind policy affinity_cpu_list must be dict, but got %s.",
                affinity_cpu_list,
            )
        else:
            set_ms_affinity(None, affinity_cpu_list)
    else:
        set_ms_affinity(None, None)


def set_ms_affinity(affinity_config, affinity_cpu_list):
    """
    Set mindspore cpu affinity. Expecting one of the arguments is None.
    If both have values, affinity_cpu_list will be set to None
    """
    if affinity_config and affinity_cpu_list:
        affinity_cpu_list = None

    if isinstance(affinity_config, str) and affinity_config.endswith(".json"):
        if not check_set_cpu_affintiy_bind_file_support:
            raise RuntimeError(
                "affinity_config is not supported to set a string value in this version of MindSpore. " \
                "Please upgrade to 2.9.0 at least."
            )

        if not os.path.exists(affinity_config):
            raise FileNotFoundError(f"Affinity config file not found: {affinity_config}")

        if not os.access(affinity_config, os.R_OK):
            raise PermissionError(f"Permission denied: Cannot read affinity config file {affinity_config}")

        mindspore.runtime.set_cpu_affinity(True, None, None, affinity_config)
        return

    if affinity_config:
        # Check if any device_X in affinity_config has X >= device_num
        max_device_id = mindformers.tools.utils.get_real_group_size() - 1
        for key in affinity_config:
            try:
                x = int(key.split("_")[1])
            except Exception as exc:
                raise ValueError(
                    f"Invalid device config key {key} in affinity_config. "
                    f"The pattern should be `device_X`, where X refers to device id."
                ) from exc
            if x > max_device_id:
                raise ValueError(
                    f"Invalid device id {x} in affinity_config. "
                    f"Maximum allowed device id is {max_device_id}."
                )
        device_id = mindformers.tools.utils.get_real_local_rank()
        device_config = affinity_config.get(f'device_{device_id}', None)
        if device_config:
            affinity_cpu_list = device_config.get('affinity_cpu_list', None)
            module_to_cpu_dict = device_config.get('module_to_cpu_dict', None)
        else:
            affinity_cpu_list = None
            module_to_cpu_dict = None
    else:
        module_to_cpu_dict = None

    mindspore.runtime.set_cpu_affinity(
        True, affinity_cpu_list, module_to_cpu_dict
    )


def set_cpu_affinity(rank_id, rank_size):
    """cpu affinity"""
    use_cpu_affinity = os.environ.get('CPU_AFFINITY')
    if use_cpu_affinity and use_cpu_affinity.lower() in ('1', 'true'):
        mindspore.dataset.config.set_numa_enable(True)
        count = psutil.cpu_count()
        current_process = psutil.Process()
        used_cpus_num = count // rank_size
        used_cpus = list(
            range(rank_id * used_cpus_num, (rank_id + 1) * used_cpus_num)
        )
        cann_used_cpus = mindformers.utils.get_cann_workqueue_cores(rank_id)
        logger.info(f"cann workqueue cpus: {cann_used_cpus}")
        used_cpus = list(set(used_cpus) - set(cann_used_cpus))
        if not used_cpus:
            # cann setup all cpus, disable the binding cores
            logger.warning(
                f"CANN use cpus: {cann_used_cpus}, "
                "model get empty cpu list, disable binding cores"
            )
            used_cpus = list(
                range(rank_id * used_cpus_num, (rank_id + 1) * used_cpus_num)
            )
        current_process.cpu_affinity(used_cpus)
        logger.info(
            f"cpu_affinity, rank_id: {rank_id}, device_num: {rank_size}"
        )


def init_context(use_parallel=False, context_config=None, parallel_config=None):
    """
    Initialize the context.

    Args:
        use_parallel (bool, optional): Whether to use parallel. Default: ``False``.
        context_config (Union[dict, ContextConfig], optional): The context config. Default: ``None``.
        parallel_config (Union[dict, ParallelContextConfig], optional): The parallel context config. Default: ``None``.

    Returns:
        - Int, the local_rank number.
        - Int, the total available devices number.

    Examples:
        >>> from mindformers import init_context
        >>> init_context(use_parallel=False)
    """

    if isinstance(context_config, ContextConfig):
        context_config = context_config.__dict__

    if isinstance(parallel_config, ParallelContextConfig):
        parallel_config = parallel_config.__dict__

    config = {
        'use_parallel': use_parallel,
        'context': context_config,
        'parallel': parallel_config,
    }
    ctx = build_context(config)
    return ctx.rank_id, ctx.device_num


def build_context(config: Union[dict, MindFormerConfig]):
    """
    Build the context from config.

    Note:
        parameter config must contain keys: 'context', 'parallel',
        when config is dict.

    Args:
        config (Union[dict, MindFormerConfig]):
            The configuration to initialize the context.
            This can be a dictionary, a MindFormerConfig instance.

    Returns:
        Context instance, The instantiated context.

    Examples:
        >>> from mindformers import build_context
        >>> config = {'context': {'mode': 'GRAPH_MODE'}, 'parallel':{}}
        >>> build_context(config=config)
    """

    config['parallel_config'] = config.get('parallel_config', {})
    mf_config = MindFormerConfig(**config)

    execute_validator(mf_config)
    ctx = Context(mf_config)

    config['local_rank'] = ctx.rank_id
    config['device_num'] = ctx.device_num

    return ctx


def set_context(run_mode=None, **kwargs):
    """
    Set context for running environment.

    Context should be configured before running your program.
    If there is no configuration,
    it will be automatically set according to the device target by default.

    Note:
        Attribute name is required for setting attributes.
        Currently only run_mode belongs to MindFormers context.
        The kwargs will be passed to MindSpore set_context.
        The determination computation for training or inference can be controlled through keyword argument.
        The keyword arguments for enabling/disabling determination computation during training and inference are:
        ``train_precision_sync`` and ``infer_precision_sync``, respectively.
        The on/off states correspond to boolean values,
        where ``True`` indicates activation and ``False`` indicates deactivation.
        This operation is a one-time action.
        Repeated attempts will not succeed and will trigger warning logs.

    Args:
        run_mode (str, optional): The mode of the model behaviour.
            Can be one of ['train', 'finetune', 'eval', 'predict']. Default: ``None``.
        **kwargs: MindSpore context arguments.

    Examples:
        >>> from mindformers import build_context, set_context
        >>> config = {'context': {'mode': 'GRAPH_MODE'}, 'parallel':{}}
        >>> build_context(config=config)
        >>> set_context(max_device_memory='59GB')
        >>> set_context(run_mode='predict', infer_precision_sync=True)
        WARNING - 'CUSTOM_MATMUL_SHUFFLE' is set to 'off' because infer_precision_sync is True.
        WARNING - 'LCCL_DETERMINISTIC' is set to '1' because infer_precision_sync is True.
        >>> set_context(run_mode='predict', infer_precision_sync=True)
        WARNING - mindspore.set_deterministic has been set, can not be set repeatedly.
        Key environment variables: HCCL_DETERMINISTIC: true, TE_PARALLEL_COMPILER: 1,
        CUSTOM_MATMUL_SHUFFLE: off, LCCL_DETERMINISTIC: 1
    """
    ctx = Context()
    ctx.set_mf_ctx_run_mode(run_mode)

    train_precision_sync = kwargs.pop('train_precision_sync', None)
    infer_precision_sync = kwargs.pop('infer_precision_sync', None)
    if train_precision_sync is not None:
        if not isinstance(train_precision_sync, bool):
            raise ValueError(
                f"train_precision_sync should be bool, got {train_precision_sync}"
            )
        ctx.mf_ctx_opr.train_precision_sync = train_precision_sync
        ctx.mf_ctx_opr.set_env()

    if infer_precision_sync is not None:
        if not isinstance(infer_precision_sync, bool):
            raise ValueError(
                f'infer_precision_sync should be bool, got {infer_precision_sync}'
            )
        ctx.mf_ctx_opr.infer_precision_sync = infer_precision_sync
        ctx.mf_ctx_opr.set_env()

    mf_ctx_config = {}
    ms_ctx_config = {}
    for k, v in kwargs.items():
        if k in MFContextConfig.get_supported_kwargs():
            mf_ctx_config[k] = v
        else:
            ms_ctx_config[k] = v
    ctx.mf_ctx_opr.set_context(**mf_ctx_config)
    ctx.ms_ctx_opr.set_context(**ms_ctx_config)


def get_context(attr_key):
    """
    Get context attribute value according to the input key.
    If some attributes are not set, they will be automatically obtained.

    Args:
        attr_key (str): The key of the attribute.

    Returns:
        Object, The value of given attribute key.

    Raises:
        ValueError: If input key is not an attribute in context.

    Examples:
        >>> from mindformers import build_context, get_context
        >>> config = {'context': {'mode': 'GRAPH_MODE'}, 'parallel':{}}
        >>> build_context(config=config)
        >>> get_context('max_device_memory')
    """
    ctx = Context()

    if attr_key in MFContextConfig.get_supported_kwargs():
        return getattr(ctx.mf_ctx_opr, attr_key, None)
    return ctx.ms_ctx_opr.get_context(attr_key)


def build_mf_context(config: Union[dict, MindFormerConfig]):
    """
    Build the mindformer context from config.

    Note:
        parameter config must contain keys: 'context', 'parallel',
        when config is dict.

    Args:
        config (Union[dict, MindFormerConfig]):
            The configuration to initialize the context.
            This can be a dictionary, a MindFormerConfig instance.

    Returns:
        Mindformer context instance, The instantiated context.
    """

    config['parallel_config'] = config.get('parallel_config', {})
    mf_config = MindFormerConfig(**config)

    execute_validator(mf_config)
    return MFContextOperator(mf_config)


def build_parallel_context(config: Union[dict, MindFormerConfig]):
    """
    Build the parallel context from config.

    Note:
        parameter config must contain keys: 'context', 'parallel',
        when config is dict.

    Args:
        config (Union[dict, MindFormerConfig]):
            The configuration to initialize the context.
            This can be a dictionary, a MindFormerConfig instance.

    Returns:
        Mindformer context instance, The instantiated context.
    """

    mf_config = MindFormerConfig(**config)

    execute_validator(mf_config)
    return ParallelOperator(mf_config)


def is_legacy_model():
    """Determine whether it is use_legacy mode."""
    mf_ctx_instance = MFContextOperator.get_mf_ctx_instance()
    if mf_ctx_instance is not None:
        is_use_legacy = mf_ctx_instance.get_context("use_legacy")
        if is_use_legacy is not None:
            return is_use_legacy
    return True
