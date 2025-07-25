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
"""Cogvlm2_Image model."""
import numpy as np

from vlmeval.vlm.cogvlm import CogVlm
from vlmeval.dataset.image_mcq import MMMUDataset
from vlmeval.dataset import DATASET_TYPE

from toolkit.benchmarks.vlmevalkit_models.multimodal_models import init_model


class CogVlmImage(CogVlm):
    """
    Initialize the model of CogVlmImage and implement inference.

    Args:
        model_path (str): Directory of containing model config.
    """
    def __init__(self, model_path):
        self.model_output = init_model(model_path)
        self.config = self.model_output.config
        self.model_config = self.model_output.model_config
        self.generation_config = self.model_output.generation_config
        self.processor = self.model_output.processor
        self.tokenizer = self.model_output.tokenizer
        self.model = self.model_output.model
        self.batch_size = self.model_output.batch_size

    def update_config(self):
        self.generation_config.max_new_tokens = 2048

    def generate_inner(self, message, dataset=None):
        """Perform model predict and return predict results.

           Args:
               message (List[Dict]): Chat text.
               dataset (str): Dataset name.

           Returns:
               Predict result.
        """
        self.update_config()
        if dataset is not None and dataset.startswith('MMMU_'):
            message = MMMUDataset.split_MMMU(message)
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        if dataset is not None and DATASET_TYPE(dataset) in ['MCQ', 'Y/N']:
            prompt += '\nShort Answer.'
        tmp_inputs = [
            [{"image": image_path}, {"text": prompt}]
        ]
        inputs = tmp_inputs * self.batch_size

        # Perform model predict and process the data for predict.
        data = self.processor(inputs)
        res = self.model.generate(
            **data,
            generation_config=self.generation_config,
        )
        input_id_length = np.max(np.argwhere(data.get("input_ids")[0] != self.tokenizer.pad_token_id)) + 1
        result = self.tokenizer.decode(res[0][input_id_length:], skip_special_tokens=True)
        return result
