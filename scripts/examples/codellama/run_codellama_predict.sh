#!/bin/bash
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
CONFIG_PATH=$1
CKPT_PATH=$2
TOKENIZER_PATH=$3
DEVICE_NUM=$4

script_path="$(realpath "$(dirname "$0")")"

bash "$script_path"/../../msrun_launcher.sh \
 "$script_path/run_codellama_generate.py \
 --config_path $CONFIG_PATH \
 --load_checkpoint $CKPT_PATH \
 --model_file $TOKENIZER_PATH \
 --use_parallel" "$DEVICE_NUM"
