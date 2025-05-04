import copy
import logging
import os
import random
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import torch
from omegaconf import DictConfig, ListConfig
from dataclasses import dataclass
from verl.utils.tracking import Tracking

from copy import deepcopy
from verl.utils.model import compute_position_id_with_mask
from verl.utils.agents.frames_sampler import sample_video_frames, sample_frames_from_next_obs
from verl.utils.agents.construct_prompt import generate_prompt
from verl.utils.agents.reward_function import extract_solution
from verl.utils.agents.tensor_helper import TensorHelper, TensorConfig
from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.dataset.vision_utils import process_image, process_video
import verl.utils.torch_functional as verl_F
@dataclass
class GenerationConfig:
    max_rounds: int
    max_frames: int
    max_prompt_length: int 
    max_response_length: int
    num_gpus: int
    no_think_rl: bool=False

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        processor,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        logger: Tracking = None,
        ratio: float = 1.0
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_rollout_wg = actor_rollout_wg
        self.is_validation = is_validation
        self.max_rounds = config.max_rounds
        self.max_frames = config.max_frames
        self.no_think_rl = config.no_think_rl
        self.num_gpus = config.num_gpus
        self.ratio = ratio
        self.logger = logger
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
        ))
        self.max_prompt_length = config.max_prompt_length
        self.max_response_length = config.max_response_length
        self.truncation = 'error'
        # Initialize completion round for each batch item
        self.completion_rounds = {}
        # Initialize conversation history and responses
        self.conversation_history = None
        # Initialize final outputs

    def run_llm_loop(self, gen_batch, global_steps=None) -> Tuple[Dict, Dict]:
        """
        Run the LLM generation loop for multiple rounds.
        """
        rollings = deepcopy(gen_batch)
        batch_size = rollings.batch['input_ids'].shape[0]
        dones = [False] * batch_size

        final_output_batch = [None] * batch_size
        # Initialize conversation history and responses for this batch
        self.conversation_history = [[] for _ in range(batch_size)]
        
        # Initialize completion tracking for this batch
        self.completion_rounds = {i: self.max_rounds for i in range(batch_size)}
        
        # Seed conversation history with initial prompt and images for each sample
        # Get any initial multi-modal data
        initial_mm = rollings.non_tensor_batch.get('multi_modal_data', [None] * batch_size)
        for i in range(batch_size):
            # Append initial images if available
            mm_item = initial_mm[i] if i < len(initial_mm) else None

            if mm_item is not None and isinstance(mm_item, dict) and 'image' in mm_item:                
              
                for img in mm_item['image']:
                    self.conversation_history[i].append({'type': 'image', 'image': img})
                # Append initial prompt text
                prompt_text = rollings.non_tensor_batch['extra_info'][i]['prompt']
                self.conversation_history[i].append({'type': 'text', 'text': prompt_text})

        for step in range(1, self.max_rounds + 1):

            # 1) generate responses only for non-done samples
            active_indices = [i for i, done in enumerate(dones) if not done]

            # Create sub-batch for active samples
            active_batch = self._create_sub_batch(rollings, active_indices)
            gen_output = self._generate_with_gpu_padding(active_batch)
            
            # Generate for active samples
            responses_str = self.tokenizer.batch_decode(
                gen_output.batch['responses'], 
                skip_special_tokens=True
            )
                        
            if all(dones) or step == self.max_rounds:
                for idx, orig_idx in enumerate(active_indices):
                # put the 1-element DataProto slice into its original slot
                    final_output_batch[orig_idx] = gen_output[idx:idx+1]
                break 
            # 2) error/correction checking for active samples
            current_times = [rollings.non_tensor_batch['extra_info'][i]['times'] for i in active_indices]

            total_times_list = [rollings.non_tensor_batch['extra_info'][i]['total_times'] for i in active_indices]

            correction_info = self.correction(responses_str, current_times, total_times_list)

            # 3) execute predictions and get next observations
            next_obs, step_dones = self.execute_predictions(
                current_times,
                correction_info
            )

            # Update done status and store completion rounds
            for idx, orig_idx in enumerate(active_indices):
                self.conversation_history[orig_idx].append({"type": "text", "text": responses_str[idx]})
                if step_dones[idx] or step == self.max_rounds:        
                    dones[orig_idx] = True
                    self.completion_rounds[orig_idx] = step
                    final_output_batch[orig_idx] = gen_output[idx:idx+1]

            
            for idx, orig_idx in enumerate(active_indices):
                # Update current timestamps for next round
                rollings.non_tensor_batch['extra_info'][orig_idx]['times'] = next_obs[idx]
                # Update past_times for the current step
                rollings.non_tensor_batch['extra_info'][orig_idx]['past_times'][step-1] = next_obs[idx]
                rollings.non_tensor_batch['extra_info'][orig_idx]['current_turn'] = step

            # 5) Sample frames for next round for non-done samples, do not sample frames for done samples and correction samples
            sampled_frames_batch = []
            for idx, orig_idx in enumerate(active_indices):
                if step_dones[idx] or correction_info[idx]["needs_correction"]:
                    sampled_frames_batch.append([])
                else:
                    sampled_frames_batch.append(
                        sample_frames_from_next_obs(
                            rollings.non_tensor_batch['extra_info'][orig_idx]['video_path'],
                            next_obs[idx],
                            rollings.non_tensor_batch['extra_info'][orig_idx]['height'],
                            rollings.non_tensor_batch['extra_info'][orig_idx]['width'],
                            ratio=self.ratio
                        )
                    )

            # 6) Update rolling state with new frames and observations
            rollings = self.update_rollings_state(
                rollings,
                active_indices,  # Pass full dones list
                sampled_frames_batch=sampled_frames_batch,
                new_times=next_obs,
                new_round=step + 1,
                responses_str=responses_str,
                correction_info=correction_info
            )

        return DataProto.concat(final_output_batch)

    def _create_sub_batch(self, full_batch: DataProto, indices: List[int]) -> DataProto:
        """Create a sub-batch containing only the specified indices, preserving batch, non_tensor_batch, and meta_info."""
        # Index each tensor in the TensorDict directly (PyTorch auto-converts Python list to LongTensor)
        new_tensors = {k: v[indices] for k, v in full_batch.batch.items()}
        # Index each numpy array in non_tensor_batch
        new_non_tensors = {k: full_batch.non_tensor_batch[k][indices] for k in full_batch.non_tensor_batch}
        return DataProto.from_dict(
            tensors=new_tensors,
            non_tensors=new_non_tensors,
        )

    def _expand_to_full_batch(self, sub_batch_data: List, active_indices: List[int], full_size: int) -> List:
        """Expand sub-batch data to full batch size, filling with None or empty lists for inactive samples."""
        full_batch_data = [[] if isinstance(sub_batch_data[0], list) else None] * full_size
        for idx, orig_idx in enumerate(active_indices):
            full_batch_data[orig_idx] = sub_batch_data[idx]
        return full_batch_data


    def update_rollings_state(
        self,
        rollings: DataProto,
        active_indices: List[int],
        sampled_frames_batch: List[List[Dict]],
        new_times: List[List[int]],
        new_round: int,
        responses_str: Optional[List[str]] = None,
        correction_info: Optional[List[Dict]] = None
    ) -> DataProto:
        """
        Update the rolling state with new frames and observations.
        Handles both text and image inputs properly through the processor.
        """

        for idx, orig_idx in enumerate(active_indices):
            # Update frame history
            if sampled_frames_batch[idx]:
                rollings.non_tensor_batch['extra_info'][orig_idx]['frames'] = sampled_frames_batch[idx]
            
            # Update current timestamps for next round
            if new_times[idx]:
                rollings.non_tensor_batch['extra_info'][orig_idx]['times'] = new_times[idx]
            
            # Update round info
            rollings.non_tensor_batch['extra_info'][orig_idx]['current_round'] = new_round
            
            # Get correction message if needed
            if correction_info[idx]["needs_correction"]:
                prompt = correction_info[idx]["message"]
            else:
                prompt = generate_prompt(
                question=rollings.non_tensor_batch['extra_info'][orig_idx]['question'],
                timestamps=new_times[idx] if new_times[idx] else [],
                total_times=rollings.non_tensor_batch['extra_info'][orig_idx].get('total_times'),
                n_round=new_round,
                max_rounds=self.max_rounds,
                max_frames=self.max_frames,
                first_round=False
            )
            
            # Process current round's images
            current_frames = sampled_frames_batch[idx] if sampled_frames_batch[idx] else []
            current_images = [process_image({"image": frame['image']}) for frame in current_frames]
            # add response to conversation history
            # Append current round's images, prompt, and response to history
            if current_images:
                for image in current_images:
                    self.conversation_history[orig_idx].append({"type": "image", "image": image})
            self.conversation_history[orig_idx].append({"type": "text", "text": prompt})

            
            # Construct messages with full history and generate prompt
            messages = [{"role": "user", "content": self.conversation_history[orig_idx]}]
            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            # Debug: check that number of vision placeholders matches number of images
            all_images = [item["image"] for item in self.conversation_history[orig_idx] if item.get("type") == "image"]
            num_images = len(all_images)
            # Count placeholders in raw_prompt (vision_start token)
            if random.randint(0, 100) == 64:
                print('============================================')
                print("[Chat History]")
                for i in range(len(self.conversation_history[orig_idx])):
                    print(f'{self.conversation_history[orig_idx][i]}')
                print('============================================')
            # Collect images from history
            # Process through model with full history
            model_inputs = self.processor(
                text=[raw_prompt], images=all_images, return_tensors="pt"
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
            
            rollings.batch['input_ids'][orig_idx] = input_ids[0]
            rollings.batch['attention_mask'][orig_idx] = attention_mask[0]

            rollings.non_tensor_batch['multi_modal_data'][orig_idx] = {"image": all_images}
            rollings.non_tensor_batch['multi_modal_inputs'][orig_idx] = dict(model_inputs)
            rollings.non_tensor_batch['raw_prompt_ids'][orig_idx] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

            
            # Handle position_ids if using Qwen2VL
            print(self.processor)
            print(self.processor.image_processor.__class__.__name__)
            if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
                from verl.models.transformers.qwen2_vl import get_rope_index
                
                position_ids = [
                    get_rope_index(
                        self.processor,
                        input_ids=input_ids[0],
                        image_grid_thw=model_inputs.get("image_grid_thw"),
                        video_grid_thw=model_inputs.get("video_grid_thw"),
                        second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                        attention_mask=attention_mask[0],
                    )
                ] 

            else:
                # Use standard position IDs
                position_ids = compute_position_id_with_mask(attention_mask)
     
            rollings.batch['position_ids'][orig_idx] = position_ids[0]
        
        return rollings
    
    def sample_frames(self,
                      video_paths: List[str],
                      next_obs: List[int],
                      heights: List[int],
                      widths: List[int]) -> List[List[Dict]]:
        """
        For each sample, if not done, parse the next_obs string to extract frame timestamps,
        then sample frames from the video using sample_frames_from_next_obs.

        Args:
            video_paths (List[str]): A list of video paths (one per sample).
            next_obs (List[List(int])]): A list of observation strings (e.g., [2, 3, 4]).
            dones (List[bool]): A list of booleans indicating whether each sample is done.
            heights (List[int]): A list of desired heights (one per sample).
            widths (List[int]): A list of desired widths (one per sample).
        
        Returns:
            List[List[Dict]]: A list (per sample) of lists of dictionaries for each sampled frame.
                              Each dictionary has keys like 'image' and 'timestamp'.
        """
        sampled_frames_batch = []
        assert len(video_paths) == len(next_obs) == len(heights) == len(widths), "Mismatched lengths in video_paths, next_obs, heights, and widths." 

        # Iterate over each sample using zip.
        for idx, obs in enumerate(next_obs):
            sampled_frames = sample_frames_from_next_obs(
                video_paths[idx],
                obs,
                heights[idx],
                widths[idx], 
                ratio=self.ratio
            )
            sampled_frames_batch.append(sampled_frames)

        return sampled_frames_batch

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt'
        )['input_ids']
        
        if next_obs_ids.shape[1] > self.max_obs_length:
            print("[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG")
            next_obs_ids = next_obs_ids[:, :self.max_obs_length]
        return next_obs_ids
    

    def _generate_with_gpu_padding(self, gen_batch: DataProto) -> DataProto:
        """
        Generate sequences with proper GPU padding handling using built-in padding functions.
        
        Args:
            gen_batch: Input batch data
            
        Returns:
            DataProto with generated sequences
        """
        # Get batch size from input_ids tensor

        # Pad batch to be divisible by world_size
        padded_batch, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
        
        # Generate with padded batch
        output_batch = self.actor_rollout_wg.generate_sequences(padded_batch)
        
        # Remove padding from output
        unpadded_batch = unpad_dataproto(output_batch, pad_size=pad_size)
        
        return unpadded_batch


    
    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']
    
    def correction(self, responses_str: List[str], current_frames: List[List[int]],  total_times: List[int]) -> List[Dict]:
        """
        Check responses for errors and generate correction info.
        
        Args:
            responses_str: List of response strings
            current_frames: List of current frame timestamps
            total_times: set of total frame timestamps

        Returns:
            List of correction dictionaries with validation results
        """
        corrections = []
        for response, frames, total_time in zip(responses_str, current_frames, total_times):
            correction_info = {
                "needs_correction": False,
                "message": "",
                "error_type": None,
            }
            
            error_type, error_message = extract_solution(response, frames, self.max_frames, total_time)
            if error_type == 'valid_frame_ops':
                correction_info['needs_correction'] = False
                correction_info['message'] = error_message
                correction_info['error_type'] = error_type
            elif error_type == 'valid_answer':
                correction_info['needs_correction'] = False
                correction_info['message'] = error_message
                correction_info['error_type'] = error_type
            elif error_type == 'format_error':
                correction_info['needs_correction'] = True
                correction_info['message'] = "Invalid response format: " + error_message + " Please try again."
                correction_info['error_type'] = error_type
            elif error_type == 'frame_error':
                correction_info['needs_correction'] = True
                correction_info['message'] = "Invalid frame selection: " + error_message + " Please try again."
                correction_info['error_type'] = error_type
            else:
                raise ValueError(f"Invalid error type: {error_type}")

            corrections.append(correction_info)
        
        return corrections


    def execute_predictions(
        self,
        current_frames: List[List[float]],
        correction_info: List[Dict],
        ) -> Tuple[List[List[float]], List[bool]]:
        """
        Execute frame selection predictions and determine if samples are done.
        
        Args:
            current_frames: List of current frame timestamps for active samples
            correction_info: List of correction info for active samples
            
        Returns:
            Tuple of (next observations, done flags) for active samples
        """
        next_obs = []
        dones = []

        for frames, corr in zip(current_frames, correction_info):
            error_type = corr['error_type']
            message = corr['message']
            if error_type == 'valid_answer':
                next_obs.append(frames)
                dones.append(True)
            elif error_type == 'valid_frame_ops':
                add_frames = message['add']
                remove_frames = message['remove']
                next_frames = update_frames(frames, add_frames, remove_frames, self.max_frames)
                next_obs.append(next_frames)
                dones.append(False)
            else:
                next_obs.append(frames)
                dones.append(False)
        
        return next_obs, dones

def update_frames(current_frames: List[int], add_frames: List[int], remove_frames: List[int]) -> Tuple[List[int], bool]:
    """
    Update the current frame selection given add and remove instructions.
    
    Returns a tuple (new_frames, is_done) where:
      - new_frames is the updated list of frames.
    """
    
    # Remove frames that are specified
    new_frames = [frame for frame in current_frames if frame not in remove_frames]
    
    # Add frames that are specified (if not already present)
    for a in add_frames:
        if a not in new_frames:
            new_frames.append(a)

    
    return new_frames