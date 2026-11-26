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
"""run_check module"""
import os
import platform
import re
import subprocess
import sys
import time
import json
from importlib import import_module
from io import StringIO
from pathlib import Path
from packaging.version import parse

import numpy as np
import mindspore as ms

from mindformers.core.context import build_context
from mindformers.tools.logger import logger


class BaseCheck:
    """The base check class, needs to be implemented."""

    def __init__(self, start, version_mapping=None):
        """
        Init function.

        Args:
            start (float): The start time of run_check.
            version_mapping (dict): The version compatibility dict. Default is None.
        """
        self.start = start
        self.version_mapping = version_mapping
        self.next_check = None

    def set_next(self, next_check):
        """
        Set next check.

        Args:
            next_check : The instance of next check.
        """
        self.next_check = next_check

    def check(self):
        """The actual checking flow control."""

    def _next(self):
        self.next_check.check()

    def _error(self, **kwargs):
        pass

    def _success(self, **kwargs):
        pass


class MSCheck(BaseCheck):
    """Test MindSpore's run_check API."""

    def check(self):
        """MindSpore check"""
        logger.info('------------------------------Starting MindSpore Run Check------------------------------')
        buffer = StringIO()
        sys.stdout = buffer

        ms.run_check()
        result = buffer.getvalue()

        sys.stdout = sys.__stdout__
        if re.search('failed', result):
            self._error(result=result)
            version_checker = VersionCheck(self.start, error_flag='MS')
            version_checker.check()
        else:
            self._success(result=result.split('\n')[1])
            self._next()

    def _error(self, **kwargs):
        logger.error('MindSpore failed!')
        logger.error(kwargs.get('result', None))
        logger.info('The MindSpore is not installed correctly, please refer to https://www.mindspore.cn/install/')

    def _success(self, **kwargs):
        logger.info(kwargs.get('result', None))


class MFCheck(BaseCheck):
    """MindFormers run test with a tiny pynative Qwen3 model forward and backward."""

    def __init__(self, start, version_mapping=None, batch_size=1, seq_length=8,
                 num_hidden_layers=2, vocab_size=128, hidden_size=64,
                 intermediate_size=128, num_attention_heads=4, num_key_value_heads=4,
                 head_dim=16, max_position_embeddings=128):
        """
        init function.

        Args:
            start (float): The start time of run_check.
            version_mapping (dict): The version compatibility dict. Default is None.
            batch_size (int): The batch size.
            seq_length (int): The sequence length.
            num_hidden_layers (int): The number of hidden layers.
            vocab_size (int): The vocab size.
            hidden_size (int): The hidden size.
            intermediate_size (int): The MLP intermediate size.
            num_attention_heads (int): The number of attention heads.
            num_key_value_heads (int): The number of key/value heads.
            head_dim (int): The attention head dimension.
            max_position_embeddings (int): The maximum position embeddings.
        """
        super().__init__(start, version_mapping)

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.ignore_token_id = -100
        self.model_config_kwargs = {
            "seq_length": seq_length,
            "vocab_size": vocab_size,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "max_position_embeddings": max_position_embeddings,
        }

    def _build_model(self):
        """Build a tiny pynative Qwen3 model for the self-check."""
        from mindformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from mindformers.models.qwen3.modeling_qwen3_train_pynative import PyNativeQwen3ForCausalLM

        model_config = Qwen3Config(**self.model_config_kwargs)
        model = PyNativeQwen3ForCausalLM(model_config)
        model.set_train(True)
        return model

    def _forward_backward(self, model):
        """Run one forward and backward pass on the tiny model."""
        vocab_size = self.model_config_kwargs['vocab_size']
        input_ids = np.random.randint(1, vocab_size, size=(self.batch_size, self.seq_length)).astype(np.int32)
        labels = np.concatenate(
            [input_ids[:, 1:], np.full((self.batch_size, 1), self.ignore_token_id, dtype=np.int32)], axis=1)
        input_ids = ms.Tensor(input_ids)
        labels = ms.Tensor(labels)

        def forward_fn(ids, lbl):
            output = model(input_ids=ids, labels=lbl)
            return output[0] if isinstance(output, (tuple, list)) else output

        grad_fn = ms.value_and_grad(forward_fn, grad_position=None,
                                    weights=ms.ParameterTuple(model.trainable_params()))
        return grad_fn(input_ids, labels)

    def check(self):
        """Run mindformers forward/backward test."""
        logger.info('------------------------------Starting MindFormers Forward/Backward Test'
                    '------------------------------')
        try:
            model = self._build_model()
            loss, grads = self._forward_backward(model)
            if not np.isfinite(loss.asnumpy()).all():
                raise RuntimeError(f"loss is not finite, got {loss}.")
            if not grads or all(grad is None for grad in grads):
                raise RuntimeError("no gradient is generated in the backward pass.")
        except Exception as e:
            self._error(error_flag='Forward/Backward')
            version_checker = VersionCheck(self.start, error_flag='Forward/Backward')
            version_checker.check()
            raise RuntimeError(f"Run check failed: {e}") from e

        self._success(test='Forward/Backward')
        self._next()

    def _error(self, **kwargs):
        sys.stdout = sys.__stdout__
        error_flag = kwargs.get('error_flag', None)
        logger.error(f'{error_flag} test failed!', exc_info=True)
        logger.info('If you need any help, please open an issue in the MindFormers repository: '
                    'https://atomgit.com/mindspore/mindformers/issues')

    def _success(self, **kwargs):
        test = kwargs.get('test', None)
        logger.info(f'{test} test passed!')


class VersionCheck(BaseCheck):
    """Version check"""

    def __init__(self, start, version_mapping=None, error_flag=None):
        super().__init__(start, version_mapping)
        self.error_flag = error_flag

    @staticmethod
    def get_recommend_versions(mf_version=None, version_mapping=None):
        """
        Check whether the given mf_version is in record.

        Args:
            mf_version (str): The mf_version needs to be searched. Default is None.
            version_mapping (dict): The version mapping dict. Default is None.

        Returns:
            dict:
                - is_mf_version_in_record (bool): Whether the MindFormers version is in record.
                - mf_version (str): The latest MindFormers version or the input if the version is in record.
                - ms_version (str): The corresponding MindSpore version.
                - cann_version (str): The corresponding cann version.
                - driver_version (str): The corresponding driver version.
        """
        matched_vs = version_mapping['mf'].get(mf_version, None)
        if matched_vs:
            is_mf_version_in_record = True
        else:
            is_mf_version_in_record = False
            mf_version = max(version_mapping['mf'].keys(), key=lambda x: tuple(map(int, x.split('.'))))
        ms_version = version_mapping['mf'][mf_version]['prefer']
        cann_version = version_mapping['ms'][ms_version]['prefer']
        driver_version = version_mapping['cann'][cann_version]['prefer']
        return {'is_mf_version_in_record': is_mf_version_in_record,
                'mf_version': mf_version,
                'ms_version': ms_version,
                'cann_version': cann_version,
                'driver_version': driver_version}

    def _print_matched_logs(self, end):
        """
        The installed versions are matched.

        Args:
            end (float): The end time of run_check.
        """
        if self.error_flag:
            logger.error(f'The run check failed in {end - self.start:.2f} seconds, '
                         f'please check the above info for more details')
        else:
            logger.info(
                f'All checks passed, used {end - self.start:.2f} seconds, the environment is correctly set up!')

    def _print_unmatched_logs(self, end, mf_version, ms_version, cann_version, driver_version):
        """
        The installed versions are unmatched.

        Args:
            end (float): The end time of run_check.
            mf_version (str): The recommended mindformers version.
            ms_version (str): The recommended mindspore version.
            cann_version (str): The recommended cann-toolkit version.
            driver_version (str): The recommended driver version.
        """
        if self.error_flag:
            logger.error(f"The run check failed in {end - self.start:.2f} "
                         f"seconds, It's recommended to install mindformers=={mf_version} mindspore=={ms_version} "
                         f"cann-toolkit=={cann_version} driver=={driver_version}")
        else:
            logger.warning(f'The installed software are unmatched '
                           f'but all checks passed in {end - self.start:.2f} seconds')
            logger.info(f"It's recommended to install mindformers=={mf_version} mindspore=={ms_version} "
                        f"cann-toolkit=={cann_version} driver=={driver_version}")

    def check(self):
        """Check whether the software versions are matched."""
        logger.info('------------------------------Searching Environment Info------------------------------')
        try:
            mf_version = import_module(__package__).__version__
            ms_version = ms.__version__
            logger.info(f'MindFormers version: {mf_version}')
            logger.info(f'MindSpore version: {ms_version}')
            ms_version = parse(ms_version).base_version  # e.g. 2.6.0rc1 will be transformed to 2.6.0

            # one of ['aarch64', 'x86_64']
            arch = platform.machine()

            cann_pth = os.getenv('ASCEND_HOME_PATH')
            if not cann_pth:
                cann_pth = Path('/usr/local/Ascend/ascend-toolkit/latest')

            cann_info_file = f'{str(cann_pth)}/{arch}-linux/ascend_toolkit_install.info'
            driver_info_file = '/usr/local/Ascend/driver/version.info'
            is_cann_info_file_exist = Path(cann_info_file).exists()
            is_driver_info_file_exist = Path(driver_info_file).exists()

            if is_cann_info_file_exist:
                cann_info = subprocess.run(["cat", cann_info_file], shell=False, capture_output=True,
                                           check=True).stdout.decode('utf-8')
                cann_version = re.search(r'=(\d[\d.\w]+)', cann_info).group(1)
                logger.info(f'Ascend-cann-toolkit version: {cann_version}')
            else:
                logger.warning('The environment variable "ASCEND_HOME_PATH" is not set, please check whether the CANN '
                               'Toolkit is installed. Try to execute "source ${HOME}/Ascend/ascend-toolkit/set_env.sh".'
                               ' For installation manual, please refer to '
                               'https://www.hiascend.com/document/detail/zh/canncommercial/')

            if is_driver_info_file_exist:
                driver_info = subprocess.run(["cat", driver_info_file], shell=False, capture_output=True,
                                             check=True).stdout.decode('utf-8')
                driver_version = re.search(r'=(\d[\d.\w]+)', driver_info).group(1)
                logger.info(f'Ascend driver version: {driver_version}')
            else:
                logger.warning('Cannot find driver info file under default path, please check whether the driver is '
                               'installed properly. Please refer to https://www.hiascend.com/document/detail/zh/canncommercial/')

            if not is_cann_info_file_exist or not is_driver_info_file_exist:
                raise RuntimeError

        # version unmatched
        except RuntimeError:
            end = time.perf_counter()
            recommend_version_dict = self.get_recommend_versions(mf_version, self.version_mapping)
            self._print_unmatched_logs(
                end,
                recommend_version_dict['mf_version'],
                recommend_version_dict['ms_version'],
                recommend_version_dict['cann_version'],
                recommend_version_dict['driver_version']
            )

        else:
            end = time.perf_counter()
            recommend_version_dict = self.get_recommend_versions(mf_version, self.version_mapping)

            # version matched
            if recommend_version_dict['is_mf_version_in_record'] \
                    and self.version_match_check(ms_version, cann_version, driver_version, recommend_version_dict):
                self._print_matched_logs(end)

            # version unmatched
            else:
                self._print_unmatched_logs(
                    end,
                    recommend_version_dict['mf_version'],
                    recommend_version_dict['ms_version'],
                    recommend_version_dict['cann_version'],
                    recommend_version_dict['driver_version']
                )

    def version_match_check(self, ms_version, cann_version, driver_version, recommend_version_dict):
        """Check whether the given versions are valid."""
        if not (ms_version == recommend_version_dict['ms_version'] or ms_version in
                self.version_mapping['mf'][import_module(__package__).__version__]['support']):
            return False
        if not cann_version == self.version_mapping['ms'][ms_version]['prefer']:
            return False
        if not driver_version == self.version_mapping['cann'][cann_version]['prefer']:
            return False
        return True


def run_check():
    """
    Check whether the installed CANN, driver, MindSpore and MindFormers versions are matched.

    The structure of the VERSION_MAP.json is shown as below:

    .. code-block::

        {
            'mf': {
                'version1': {
                    'prefer': 'prefered ms version',
                    'support': [competible ms version list]
                },
            },
            'ms': {
                'version1': {
                    'prefer' : 'prefered cann version',
                    'support': [competible cann version list]
                },
            },
            'cann': {
                'version1': {
                    'prefer' : 'prefered driver version',
                    'support': [competible driver version list]
                },
            }
        }

    Examples:
        >>> from mindformers import run_check
        >>> run_check()
    """

    version_file = Path(f'{__file__}').parent / 'VERSION_MAP.json'
    if not version_file.is_file():
        raise RuntimeError('Cannot find VERSION_MAP.json or the found one is not a file')

    with open(version_file, encoding='utf-8') as f:
        version_mapping = json.load(f)

    os.environ['MS_ALLOC_CONF'] = "enable_vmm:False"

    start = time.perf_counter()
    build_context({"context": {"mode": 1}})

    ms_checker = MSCheck(start, version_mapping=version_mapping)
    mf_checker = MFCheck(start, version_mapping=version_mapping)
    version_checker = VersionCheck(start, version_mapping=version_mapping)

    ms_checker.set_next(mf_checker)
    mf_checker.set_next(version_checker)

    ms_checker.check()
