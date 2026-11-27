# Copyright 2022 Huawei Technologies Co., Ltd
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

"""Build Tokenizer API."""
import os.path

from mindformers.core.context.build_context import get_context


def build_tokenizer(config=None, pretrained_model_dir=None, trust_remote_code=False):
    r"""Build tokenizer For MindFormer.
    The tokenizer is always instantiated with ``transformers.AutoTokenizer`` from a local model directory.

    Args:
        config (dict, str): The task tokenizer's config or the local directory of the tokenizer.
            When it is a dict, the ``pretrained_model_dir`` field in it is used. Default: None.
        pretrained_model_dir (str): The local model directory that holds the tokenizer files.
            When it is not given, the value in ``config`` or the runtime context is used. Default: None.
        trust_remote_code (bool): Whether to trust remote code when loading the tokenizer. Default: False.

    Return:
        The tokenizer instance.

    Examples:
        >>> from mindformers import build_tokenizer
        >>> tokenizer = build_tokenizer(pretrained_model_dir='/path/to/tokenizer_dir')
    """
    if pretrained_model_dir is None and config is not None:
        if isinstance(config, str):
            pretrained_model_dir = config
        else:
            pretrained_model_dir = config.get("pretrained_model_dir", None)
    if not pretrained_model_dir:
        pretrained_model_dir = get_context('pretrained_model_dir')
    if not pretrained_model_dir:
        raise ValueError("The current interface only supports building a tokenizer from a local model "
                         "directory, but the provided path is empty or None.")
    pretrained_model_dir = os.path.realpath(pretrained_model_dir)
    if not os.path.isdir(pretrained_model_dir):
        raise ValueError(f"The current interface only supports building a tokenizer from a local model "
                         f"directory, but the provided path '{pretrained_model_dir}' is not a valid directory.")

    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_dir,
                                         trust_remote_code=trust_remote_code)
