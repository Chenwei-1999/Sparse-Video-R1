# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings
import wandb


import argparse

import os

if __name__== "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
  parser.add_argument('--video_path', type=str, default='/shares/hlw3876/chenwei/NExT-QA')
  parser.add_argument('--dataset', type=str, default='NEXT-QA')
  parser.add_argument('--max_iter', type=int, default=5)

  args = parser.parse_args()

  if args.dataset not in args.video_path:
    raise ValueError(f"Please make sure the video path matches dataset name in it.")
  
  if args.dataset == 'NEXT-QA':
    from TinyZero.test_video.helper import get_answer
  

  




