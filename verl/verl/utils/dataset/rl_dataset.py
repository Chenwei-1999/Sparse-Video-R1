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

from omegaconf import ListConfig
import os
from typing import List, Union, Optional
import copy
import pandas as pd
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

from verl.utils.agents.frames_sampler import sample_video_frames
from verl.utils.agents.construct_prompt import generate_prompt

import random
import json

def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


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


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 dataset_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 mm_key='videos', # multi-modal key
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 max_frames=5,
                 resolution=1.0,
                 max_rounds=5,
                 sampling_strategy=None,
                 return_raw_chat=False,
                 truncation='error',
                 filter_overlong_prompts=False):
        if not isinstance(dataset_files, (List, ListConfig)):
            dataset_files = [dataset_files]

        self.dataset_files = copy.deepcopy(dataset_files)
        self.original_dataset_files = copy.deepcopy(dataset_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.mm_key = mm_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.resolution = resolution

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts
        self.max_frames = max_frames
        self.max_rounds = max_rounds
        self.sampling_strategy = sampling_strategy
        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        dataset_files = self.dataset_files if not use_origin_parquet else self.original_dataset_files
        for i, parquet_file in enumerate(dataset_files):
            self.dataset_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        
        dataframes = []

        # for parquet_file in self.dataset_files:
        #     # read parquet files and cache
        #     dataframe = pd.read_parquet(parquet_file)
        #     dataframes.append(dataframe)
        # self.dataframe = pd.concat(dataframes)
        for dataset_file in self.dataset_files:
            # the dataset_file is json
            if dataset_file.endswith('.parquet'):
                dataframe = pd.read_parquet(dataset_file)
                dataframes.append(dataframe)
            elif dataset_file.endswith('.json'):
                with open(dataset_file, 'r') as f:
                    data = json.load(f)
                dataframes.extend(data)

        self.dataframe = pd.DataFrame(dataframes)

        print(f'dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        # Note, for multi-modal, we NO need to filter out too long image tokens
        if self.filter_overlong_prompts and self.max_prompt_length>0:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
                tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                                                                 axis=1)]

            print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_dataset_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)
        row_dict['question'] = chat
      
        is_multi_modal = self.mm_key in row_dict
        num_frames = self.max_frames

        # This project settings is, question is the raw question from each datasets, and video is the video path
        # We need sample frames from video, construct real prompt with sampled frames and questions
        if is_multi_modal and self.mm_key == 'video' and type(chat) == str: 
            height = row_dict.get('height', None)
            width = row_dict.get('width', None)
            # sample video frames
            video_path = row_dict[self.mm_key]

            sampled_frames, sampled_times, total_frames = sample_video_frames(video_path, height=height, width=width, num_frames=num_frames, strategy=self.sampling_strategy, ratio=self.resolution)
            row_dict["extra_info"] = row_dict.get("extra_info", {})
            row_dict["extra_info"]["total_frames"] = total_frames
            row_dict["frames"] = sampled_frames
            row_dict["times"] = sampled_times
            row_dict["round"] = 1

            row_dict[self.mm_key] = [frame for frame in sampled_frames]
            prompt = generate_prompt(question=chat, 
                                     timestamps=sampled_times, 
                                     total_frames=total_frames,
                                     max_rounds=self.max_rounds, 
                                     max_frames=self.max_frames)
            chat = [
                {
                    "role": "user",
                    "content": '<image>'*len(sampled_frames) + prompt
                }
            ]
        
        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        
        if is_multi_modal:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.mm_key)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if is_multi_modal:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        row_dict['is_multi_modal'] = is_multi_modal
        row_dict['video_path'] = video_path
        

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt

        index = row_dict["extra_info"].get("index", None)
        row_dict["index"] = index
        
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
