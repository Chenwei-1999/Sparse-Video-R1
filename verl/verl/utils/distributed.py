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
"""Utilities for distributed training."""

import os


def initialize_global_process_group(timeout_second=36000):
    from datetime import timedelta

    import torch.distributed

    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=timeout_second))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Set CUDA device before initializing process group
    torch.cuda.set_device(local_rank)
    
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=timeout_second)
    )
    
    return local_rank, rank, world_size
