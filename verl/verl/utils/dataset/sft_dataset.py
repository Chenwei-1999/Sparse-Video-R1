# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union, Optional
import os
import copy
import pandas as pd
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, ProcessorMixin
from omegaconf import ListConfig

from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.agents.frames_sampler import sample_video_frames
from verl.utils.agents.construct_prompt import generate_prompt
from verl.utils import hf_tokenizer

import json


def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    # Handle empty or None image
    if image is None:
        raise ValueError("Received None image")
    
    # Handle empty dictionary
    if isinstance(image, dict) and not image:
        raise ValueError("Received empty image dictionary")

    if isinstance(image, dict):
        if 'bytes' not in image:
            raise ValueError("Image dictionary missing 'bytes' key")
        if not image['bytes']:
            raise ValueError("Image dictionary has empty bytes")
        image = Image.open(BytesIO(image['bytes']))

    # Handle invalid image
    if not image:
        raise ValueError("Failed to load image")

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class SFTDataset(Dataset):
    """
    Multi-modal SFT Dataset supporting both text and video/image inputs
    """
    def __init__(self,
                 dataset_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 response_key='response',
                 mm_key='videos',  # multi-modal key
                 max_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/sft',
                 max_frames=5,
                 resolution=1.0,
                 sampling_strategy=None,
                 truncation='error'):
        
        if not isinstance(dataset_files, (List, ListConfig)):
            dataset_files = [dataset_files]

        self.dataset_files = copy.deepcopy(dataset_files)
        self.cache_dir = os.path.expanduser(cache_dir)
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.response_key = response_key
        self.mm_key = mm_key
        self.max_length = max_length
        self.filter_prompts = filter_prompts
        self.resolution = resolution
        self.truncation = truncation
        self.max_frames = max_frames
        self.sampling_strategy = sampling_strategy
        
        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, dataset_file in enumerate(self.dataset_files):
            self.dataset_files[i] = copy_to_local(src=dataset_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for dataset_file in self.dataset_files:
            if dataset_file.endswith('.parquet'):
                dataframe = pd.read_parquet(dataset_file)
                dataframes.append(dataframe)
            elif dataset_file.endswith('.json'):
                with open(dataset_file, 'r') as f:
                    data = json.load(f)
                dataframes.extend(data)

        self.dataframe = pd.DataFrame(dataframes)
        print(f'dataset len: {len(self.dataframe)}')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        
        prompt = row_dict[self.prompt_key]
        response = row_dict[self.response_key]
        
        is_multi_modal = self.mm_key in row_dict
        num_frames = self.max_frames

        # Handle video/image inputs if present
        if is_multi_modal and self.mm_key == 'video' and isinstance(prompt, str):
            height = row_dict.get('height', None)
            width = row_dict.get('width', None)
            video_path = row_dict[self.mm_key]

            sampled_frames, sampled_times, total_frames = sample_video_frames(
                video_path, height=height, width=width, 
                num_frames=num_frames, strategy=self.sampling_strategy, 
                ratio=self.resolution
            )
            
            row_dict["extra_info"] = row_dict.get("extra_info", {})
            row_dict["extra_info"]["total_frames"] = total_frames
            row_dict["frames"] = sampled_frames
            row_dict["times"] = sampled_times
            
            # Generate prompt with video frame information
            instruction = "Please only generate the answer. \n"
            prompt = '<image>'*len(sampled_frames) + instruction + prompt

            # Process images
            row_dict['multi_modal_data'] = {'image': [process_image(image) for image in sampled_frames]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

        # Create chat format
        prompt_chat = [{'role': 'user', 'content': prompt}]
        prompt_chat_str = self.tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + self.tokenizer.eos_token

        if is_multi_modal:
            # Handle image tokens in prompt
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_chat_str:
                    prompt_chat_str = prompt_chat_str.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1
                prompt_chat_str = prompt_chat_str.replace('<|placeholder|>', self.processor.image_token)

        # Tokenize
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_chat_str + response_chat_str,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation
        )

        # Get position IDs
        if is_multi_modal:
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )
            position_ids = position_ids[0]
        else:
            position_ids = compute_position_id_with_mask(attention_mask[0])

        # Create loss mask
        loss_mask = attention_mask[0].clone()
        prompt_ids = self.tokenizer(prompt_chat_str, add_special_tokens=False)['input_ids']
        prompt_length = len(prompt_ids)
        
        if prompt_length > 1:
            # Mask out prompt tokens
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0
            
        # Mask out the last token (usually EOS)
        if loss_mask.size(0) > 0:
            loss_mask[-1] = 0

        return {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids,
            'loss_mask': loss_mask,
            'is_multi_modal': is_multi_modal,
            'video_path': video_path if is_multi_modal else None
        }
