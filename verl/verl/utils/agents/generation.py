import copy
import logging
import os
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import torch
from omegaconf import DictConfig, ListConfig
from dataclasses import dataclass

from copy import deepcopy
from verl.utils.model import compute_position_id_with_mask
from verl.utils.agents.frames_sampler import sample_video_frames, sample_frames_from_next_obs
from verl.utils.agents.construct_prompt import generate_prompt
from verl.utils.agents.reward_function import compute_score, extract_solution, parse_frame_list, extract_answer
from verl.utils.tracking import Tracking
from verl.utils.agents.tensor_helper import TensorHelper, TensorConfig
from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.dataset.vision_utils import process_image, process_video
logger = logging.getLogger(__name__)

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
        logger: Tracking,
        is_validation: bool = False,
        ratio: float = 1.0
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_rollout_wg = actor_rollout_wg
        self.logger = logger 
        self.is_validation = is_validation
        self.max_rounds = config.max_rounds
        self.max_frames = config.max_frames
        self.no_think_rl = config.no_think_rl
        self.num_gpus = config.num_gpus
        self.ratio = ratio
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
        self.responses_history = None

    def _compose_final_output(self, rollings: DataProto, responses: torch.Tensor, meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        if responses is None:
            raise ValueError("Missing 'responses' in final_output")

        final_output = {}
        final_output['prompts'] = rollings.batch['input_ids']
        final_output['responses'] = responses
        final_output['input_ids'] = rollings.batch['input_ids']  # Already contains full context
        final_output['attention_mask'] = rollings.batch['attention_mask']
        final_output['position_ids'] = rollings.batch['position_ids']
        final_output['reward_model'] = rollings.batch.get('reward_model', None)
        final_output['data_source'] = rollings.non_tensor_batch.get('data_source', '')
        final_output['extra_info'] = rollings.non_tensor_batch.get('extra_info', {})
        
        final_output = DataProto.from_single_dict(final_output)
        final_output.meta_info.update(meta_info)

        return final_output

    def run_llm_loop(self, gen_batch, global_steps=None) -> Tuple[Dict, Dict]:
        """
        Run the LLM generation loop for multiple rounds.
        """
        rollings = deepcopy(gen_batch)
        batch_size = rollings.batch['input_ids'].shape[0]
        final_response_ids_list = [None] * batch_size
        dones = [False] * batch_size
        meta_info = {}
        
        # Initialize conversation history and responses for this batch
        if self.conversation_history is None:
            self.conversation_history = [[] for _ in range(batch_size)]
        self.responses_history = [[] for _ in range(batch_size)]
        
        # Initialize completion tracking for this batch
        self.completion_rounds = {i: self.max_rounds for i in range(batch_size)}

        for step in range(1, self.max_rounds + 1):
            # Skip if all samples are done
            if all(dones):
                break
            # 1) generate responses only for non-done samples
            active_indices = [i for i, done in enumerate(dones) if not done]
            if not active_indices:
                break

            # Create sub-batch for active samples
            active_batch = self._create_sub_batch(rollings, active_indices)
            gen_output = self._generate_with_gpu_padding(active_batch)
            
            # Generate for active samples
            responses_str = self.tokenizer.batch_decode(
                gen_output.batch['responses'], 
                skip_special_tokens=True
            )

            # Store responses in history
            for idx, orig_idx in enumerate(active_indices):
                self.responses_history[orig_idx].append(responses_str[idx])

            # 2) error/correction checking for active samples
            current_times = [rollings.non_tensor_batch['extra_info'][i]['times'] for i in active_indices]
            correction_info = self.correction(responses_str, current_times)

            # 3) execute predictions and get next observations
            next_obs, step_dones = self.execute_predictions(
                responses_str,
                current_times,
                correction_info
            )
            responses_ids = gen_output.batch['responses']

            # Update done status and store completion rounds
            for idx, orig_idx in enumerate(active_indices):
                if step_dones[idx]:
                    dones[orig_idx] = True
                    self.completion_rounds[orig_idx] = step
                    final_response_ids_list[orig_idx] = responses_ids[idx]
                    meta_info[f'sample_{orig_idx}'] = {
                        'final_round': step,
                        'timestamps': rollings.non_tensor_batch['extra_info'][orig_idx]['past_times'][:step],
                        'max_frames': self.max_frames,
                        'max_rounds': self.max_rounds,
                    }

            # 4) Calculate rewards for active samples
            for idx, orig_idx in enumerate(active_indices):
                # Update current timestamps for next round
                rollings.non_tensor_batch['extra_info'][orig_idx]['times'] = next_obs[idx]
                # Update past_times for the current step
                rollings.non_tensor_batch['extra_info'][orig_idx]['past_times'][step-1] = next_obs[idx]
                rollings.non_tensor_batch['extra_info'][orig_idx]['current_turn'] = step
                rollings.non_tensor_batch['extra_info'][orig_idx]['max_turns'] = self.max_rounds
                
                # Calculate reward
                reward = compute_score(
                    data_source=rollings.non_tensor_batch.get('data_source', ''),
                    solution_str=responses_str[idx],
                    ground_truth=None,
                    extra_info=rollings.non_tensor_batch['extra_info'][orig_idx],
                    status='running'
                )
                
                # Store reward directly in rollings
                rewards = rollings.non_tensor_batch['extra_info'][orig_idx].get('rewards', [])
                if len(rewards) <= step - 1:
                    rewards.extend([None] * (step - len(rewards)))
                rewards[step-1] = reward
                rollings.non_tensor_batch['extra_info'][orig_idx]['rewards'] = rewards

            # 5) Sample frames for next round for non-done samples
            sampled_frames_batch = []
            for idx, orig_idx in enumerate(active_indices):
                if step_dones[idx]:
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
                dones,  # Pass full dones list
                sampled_frames_batch=self._expand_to_full_batch(sampled_frames_batch, active_indices, batch_size),
                new_times=self._expand_to_full_batch(next_obs, active_indices, batch_size),
                new_round=step + 1,
                correction_info=self._expand_to_full_batch(correction_info, active_indices, batch_size)
            )
        
        # Handle any samples that didn't finish within max_rounds
        for i in range(batch_size):
            if final_response_ids_list[i] is None:
                try:
                    active_idx = active_indices.index(i)
                    final_response_ids_list[i] = responses_ids[active_idx]
                except ValueError:
                    final_response_ids_list[i] = torch.full_like(responses_ids[0], self.tokenizer.pad_token_id)
                
                meta_info[f'sample_{i}'] = {
                    'final_round': self.max_rounds,
                    'timestamps': rollings.non_tensor_batch['extra_info'][i]['past_times'],
                    'max_frames': self.max_frames,
                    'max_rounds': self.max_rounds,
                }

        final_response_ids_tensor = torch.stack(final_response_ids_list, dim=0)
        return self._compose_final_output(
            rollings, final_response_ids_tensor, meta_info
        )

    def _create_sub_batch(self, full_batch: DataProto, indices: List[int]) -> DataProto:
        """Create a sub-batch containing only the specified indices, preserving batch, non_tensor_batch, and meta_info."""
        # Index each tensor in the TensorDict directly (PyTorch auto-converts Python list to LongTensor)
        new_tensors = {k: v[indices] for k, v in full_batch.batch.items()}
        # Index each numpy array in non_tensor_batch
        new_non_tensors = {k: full_batch.non_tensor_batch[k][indices] for k in full_batch.non_tensor_batch}
        # Maintain the same meta_info dictionary
        return DataProto.from_dict(
            tensors=new_tensors,
            non_tensors=new_non_tensors,
            meta_info=full_batch.meta_info,
        )

    def _expand_to_full_batch(self, sub_batch_data: List, active_indices: List[int], full_size: int) -> List:
        """Expand sub-batch data to full batch size, filling with None or empty lists for inactive samples."""
        full_batch_data = [[] if isinstance(sub_batch_data[0], list) else None] * full_size
        for idx, orig_idx in enumerate(active_indices):
            full_batch_data[orig_idx] = sub_batch_data[idx]
        return full_batch_data

    def construct_prompt_tokens(self, 
                              question: str,
                              current_frames: List[Dict],
                              new_times: List[float],
                              prev_times: List[List[float]],
                              correction: str,
                              round_num: int,
                              responses_history: List[str],
                              total_frames: Optional[int] = None) -> Dict:
        """
        Construct prompt tokens for the current round using proper processor pipeline.
        Returns dict with input_ids, attention_mask, and other model inputs.
        """
        # Construct the current prompt
        prompt = correction if correction else generate_prompt(
            question=question,
            timestamps=new_times,
            total_frames=total_frames,
            n_round=round_num,
            max_rounds=self.max_rounds,
            max_frames=self.max_frames,
        )

        # Process images if present
        images = [process_image({"image": frame['image']}) for frame in current_frames]
        multi_modal_data = {}
        multi_modal_data["image"] = images


        # Construct messages with proper format
        messages = [
            {
                "role": "user",
                "content": [],
            }
        ]
        
        # Add images to messages
        for image in images:
            messages[0]["content"].append({"type": "image", "image": image})
        
        # Add text to messages
        messages[0]["content"].append({"type": "text", "text": prompt})
        raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")

        # Apply chat template
        raw_prompt = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )

        # Process through the processor to get all necessary inputs
        model_inputs = self.processor(
            text=[raw_prompt], 
            images=images if images else None,
            return_tensors="pt"
        )

        # Remove second_per_grid_ts if present
        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        return model_inputs

    def update_rollings_state(
        self,
        rollings: DataProto,
        dones: List[bool],
        sampled_frames_batch: List[List[Dict]],
        new_times: List[List[float]],
        new_round: int,
        correction_info: Optional[List[Dict]] = None
    ) -> DataProto:
        """
        Update the rolling state with new frames and observations.
        Handles both text and image inputs properly through the processor.
        """
        batch_size = len(dones)
        updated_rollings = deepcopy(rollings)
        
        # Generate new prompts for active samples
        active_indices = [i for i, done in enumerate(dones) if not done]
        if active_indices:
            for i in active_indices:
                # Update frame history
                if sampled_frames_batch[i]:
                    updated_rollings.non_tensor_batch['extra_info'][i]['frames'] = sampled_frames_batch[i]
                
                # Update current timestamps for next round
                if new_times[i]:
                    updated_rollings.non_tensor_batch['extra_info'][i]['times'] = new_times[i]
                
                # Update round info
                updated_rollings.non_tensor_batch['extra_info'][i]['current_round'] = new_round
                
                # Get correction message if needed
                correction = ""
                if correction_info and i < len(correction_info):
                    if correction_info[i].get("needs_correction", False):
                        correction = correction_info[i].get("message", "")
                
                # Process current round's images
                current_frames = sampled_frames_batch[i] if sampled_frames_batch[i] else []
                current_images = [process_image({"image": frame['image']}) for frame in current_frames]
                
                # Generate current round's prompt
                prompt = correction if correction else generate_prompt(
                    question=updated_rollings.non_tensor_batch['extra_info'][i]['question'],
                    timestamps=new_times[i] if new_times[i] else [],
                    total_frames=updated_rollings.non_tensor_batch['extra_info'][i].get('total_frames'),
                    n_round=new_round,
                    max_rounds=self.max_rounds,
                    max_frames=self.max_frames,
                )
                
                # Initialize conversation history if needed (only do this once, not every round)
                if i >= len(self.conversation_history) or self.conversation_history[i] is None:
                    self.conversation_history[i] = []
                    
                # Add current round's content to history in sequence
                # Add images for this round
                if current_images:
                    for image in current_images:
                        self.conversation_history[i].append(
                            {"type": "image", "image": image}
                        )
                # Add prompt for this round
                self.conversation_history[i].append(
                    {"type": "text", "text": prompt}
                )
                
                # Add previous round's response if it exists
                current_round = new_round - 1  # new_round is the next round, so current_round is new_round - 1
                if current_round > 0 and len(self.responses_history[i]) >= current_round:
                    current_round_response = self.responses_history[i][current_round - 1]  # -1 because list is 0-indexed
                    if current_round_response:  # Only add if response exists and is not empty
                        self.conversation_history[i].append(
                            {"type": "text", "text": current_round_response}
                        )
                
                # Construct messages with full history
                messages = [{"role": "user", "content": self.conversation_history[i]}]
                
                # Apply chat template and get model inputs
                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                print("--------------------------------")
                print(f"raw_prompt: {raw_prompt}")
                print("--------------------------------")
                # Get all images from history
                all_images = [item["image"] for item in self.conversation_history[i] 
                            if item["type"] == "image"]
                
                # Process through model
                model_inputs = self.processor(text=[raw_prompt], images=all_images, return_tensors="pt")
                
                # Use TensorHelper to pad the tensors properly
                padded_inputs = {}
                for key, value in model_inputs.items():
                    # Only pad sequence-like tensors
                    if len(value[0].shape) > 0:
                        # Use TensorHelper to pad the tensor
                        padded_tensor = self.tensor_fn.pad_tensor(
                            value[0].unsqueeze(0),  # Add batch dimension for TensorHelper
                            pad_id=0 if key != 'input_ids' else self.tokenizer.pad_token_id
                        )
                        padded_inputs[key] = padded_tensor
                    else:
                        # For non-sequence tensors, keep as is
                        padded_inputs[key] = value
                
                # Update all model inputs
                for key, value in padded_inputs.items():
                    if key not in updated_rollings.batch:
                        updated_rollings.batch[key] = torch.zeros(
                            (batch_size,) + value.shape[1:],
                            dtype=value.dtype,
                            device=value.device
                        )
                    updated_rollings.batch[key][i] = value[0]  # Take first item since padding adds batch dimension
                
                # Update multi_modal_data in non_tensor_batch
                if 'multi_modal_data' not in updated_rollings.non_tensor_batch:
                    updated_rollings.non_tensor_batch['multi_modal_data'] = [None] * batch_size
                updated_rollings.non_tensor_batch['multi_modal_data'][i] = {"image": all_images}
                
                # Handle position_ids if using Qwen2VL
                if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
                    from verl.models.transformers.qwen2_vl import get_rope_index
                    
                    position_ids = get_rope_index(
                        self.processor,
                        input_ids=updated_rollings.batch['input_ids'][i],
                        image_grid_thw=model_inputs.get('image_grid_thw'),
                        video_grid_thw=model_inputs.get('video_grid_thw'),
                        second_per_grid_ts=None,  # We removed this earlier
                        attention_mask=updated_rollings.batch['attention_mask'][i],
                    )
                    
                    if 'position_ids' not in updated_rollings.batch:
                        updated_rollings.batch['position_ids'] = torch.zeros(
                            (batch_size,) + position_ids.shape[1:],
                            dtype=position_ids.dtype,
                            device=position_ids.device
                        )
                    updated_rollings.batch['position_ids'][i] = position_ids
                else:
                    # Use standard position IDs
                    position_ids = compute_position_id_with_mask(updated_rollings.batch['attention_mask'][i].unsqueeze(0))[0]
                    if 'position_ids' not in updated_rollings.batch:
                        updated_rollings.batch['position_ids'] = torch.zeros(
                            (batch_size,) + position_ids.shape,
                            dtype=position_ids.dtype,
                            device=position_ids.device
                        )
                    updated_rollings.batch['position_ids'][i] = position_ids
        
        return updated_rollings
    
    def sample_frames(self,
                      video_paths: List[str],
                      next_obs: List[str],
                      heights: List[int],
                      widths: List[int]) -> List[List[Dict]]:
        """
        For each sample, if not done, parse the next_obs string to extract frame timestamps,
        then sample frames from the video using sample_frames_from_next_obs.

        Args:
            video_paths (List[str]): A list of video paths (one per sample).
            next_obs (List[List(float])]): A list of observation strings (e.g., [2, 3, 4]).
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
        batch_size = gen_batch.batch['input_ids'].shape[0]
        
        # If single GPU or batch size is divisible by world_size, no padding needed
        if self.num_gpus <= 1 or batch_size % self.actor_rollout_wg.world_size == 0:
            return self.actor_rollout_wg.generate_sequences(gen_batch)
            
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
    
    def correction(self, responses_str: List[str], current_frames: List[List[float]]) -> List[Dict]:
        """
        Check responses for errors and generate correction info.
        
        Args:
            responses_str: List of response strings
            current_frames: List of current frame timestamps
            
        Returns:
            List of correction dictionaries with validation results
        """
        corrections = []
        for response, frames in zip(responses_str, current_frames):
            correction_info = {
                "needs_correction": False,
                "error_type": None,
                "message": "",
                "invalid_frames": []
            }
            
            try:
                # First check for required think tag
                think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
                if not think_match:
                    correction_info.update({
                        "needs_correction": True,
                        "error_type": "think_error",
                        "message": "Missing <think> reasoning section. Please provide your reasoning."
                    })
                    corrections.append(correction_info)
                    continue

                # Check for frame or answer tags
                frame_match = re.search(r'<frame>(.*?)</frame>', response, re.DOTALL)
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)

                if not frame_match and not answer_match:
                    correction_info.update({
                        "needs_correction": True,
                        "error_type": "format_error",
                        "message": "Missing both <frame> and <answer> tags. Please provide either frame operations or final answer."
                    })
                    corrections.append(correction_info)
                    continue

                # If we have an answer tag, validate it
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    if not answer_content:
                        correction_info.update({
                            "needs_correction": True,
                            "error_type": "answer_error",
                            "message": "Empty answer tag. Please provide a valid answer."
                        })
                    elif not re.match(r'^\d+$', answer_content):
                        correction_info.update({
                            "needs_correction": True,
                            "error_type": "answer_error",
                            "message": "Answer must be a single number."
                        })
                    corrections.append(correction_info)
                    continue

                # If we have a frame tag, validate frame operations
                if frame_match:
                    frame_content = frame_match.group(1).strip()
                    add_match = re.search(r'\+\[(.*?)\]', frame_content)
                    remove_match = re.search(r'-\[(.*?)\]', frame_content)

                    if not (add_match or remove_match):
                        correction_info.update({
                            "needs_correction": True,
                            "error_type": "frame_error",
                            "message": "Invalid frame operation format. Use +[...] to add frames and/or -[...] to remove frames."
                        })
                        corrections.append(correction_info)
                        continue

                    # Parse and validate frame lists
                    added = parse_frame_list(add_match.group(1)) if add_match else []
                    removed = parse_frame_list(remove_match.group(1)) if remove_match else []

                    # Validate frame operations
                    invalid_frames = []
                    for frame in removed:
                        if frame not in frames:
                            invalid_frames.append(frame)

                    if invalid_frames:
                        correction_info.update({
                            "needs_correction": True,
                            "error_type": "frame_error",
                            "message": f"Cannot remove frames that don't exist: {invalid_frames}",
                            "invalid_frames": invalid_frames
                        })
                        corrections.append(correction_info)
                        continue

                    # Check for duplicate frames in additions
                    duplicate_frames = [f for f in added if f in frames]
                    if duplicate_frames:
                        correction_info.update({
                            "needs_correction": True,
                            "error_type": "frame_error",
                            "message": f"Cannot add frames that already exist: {duplicate_frames}",
                            "invalid_frames": duplicate_frames
                        })
                        corrections.append(correction_info)
                        continue

                    # Check frame count limit
                    new_frame_count = len(frames) - len(removed) + len(added)
                    if new_frame_count > self.max_frames:
                        correction_info.update({
                            "needs_correction": True,
                            "error_type": "frame_error",
                            "message": f"Operation would exceed maximum allowed frames ({self.max_frames})"
                        })
                        corrections.append(correction_info)
                        continue

            except Exception as e:
                correction_info.update({
                    "needs_correction": True,
                    "error_type": "parsing_error",
                    "message": f"Error processing response: {str(e)}"
                })
            
            corrections.append(correction_info)
            
        return corrections


    def execute_predictions(
        self,
        responses_str: List[str],
        current_frames: List[List[float]],
        correction_info: List[Dict]
    ) -> Tuple[List[List[float]], List[bool]]:
        """
        Execute frame selection predictions and determine if samples are done.
        
        Args:
            responses_str: List of response strings for active samples
            current_frames: List of current frame timestamps for active samples
            correction_info: List of correction info for active samples
            
        Returns:
            Tuple of (next observations, done flags) for active samples
        """
        next_obs = []
        dones = []

        # Ensure all input lists have the same length
        if not (len(responses_str) == len(current_frames) == len(correction_info)):
            raise ValueError(
                f"Mismatched lengths: responses={len(responses_str)}, "
                f"frames={len(current_frames)}, corrections={len(correction_info)}"
            )
        
        for resp, curr_frames, corr in zip(responses_str, current_frames, correction_info):
            # If needs correction, continue with current frames
            if corr.get("needs_correction", False):
                next_obs.append(curr_frames)
                dones.append(False)
                continue

            # Extract either frame operations or final answer
            content = extract_answer(resp) if resp else None
            if content is None:
                # Invalid response format - needs correction
                next_obs.append(curr_frames)
                dones.append(False)
                continue

            try:
                # Check if it's a frame operation or final answer
                if '<frame>' in resp:
                    # Handle frame operations
                    add_match = re.search(r'\+\[(.*?)\]', content)
                    remove_match = re.search(r'-\[(.*?)\]', content)
                    add_frames = parse_frame_list(add_match.group(1)) if add_match else []
                    remove_frames = parse_frame_list(remove_match.group(1)) if remove_match else []

                    new_frames, is_done = update_frames(
                        curr_frames, add_frames, remove_frames, self.max_frames
                    )
                    next_obs.append(new_frames)
                    dones.append(is_done)
                else:
                    # It's a final answer - we're done
                    # Only mark as done if it's a valid answer (single number)
                    if re.match(r'^\d+$', content.strip()):
                        next_obs.append(curr_frames)
                        dones.append(True)
                    else:
                        # Invalid answer format - needs correction
                        next_obs.append(curr_frames)
                        dones.append(False)
            except Exception as e:
                logger.warning(f"Error processing response '{content}': {str(e)}")
                # Error in processing - needs correction
                next_obs.append(curr_frames)
                dones.append(False)
        
        return next_obs, dones

def update_frames(current_frames: List[int], add_frames: List[int], remove_frames: List[int], max_frames: int) -> Tuple[List[int], bool]:
    """
    Update the current frame selection given add and remove instructions.
    
    Returns a tuple (new_frames, is_done) where:
      - new_frames is the updated list of frames.
      - is_done is True if:
           * any remove frame is not in current_frames,
           * or the new selection size exceeds max_frames.
    """
    # Check: if any removal instruction refers to a frame not in current_frames,
    # we cannot update further → mark as done.
    for r in remove_frames:
        if r not in current_frames:
            return current_frames, True
    
    # Remove frames that are specified
    new_frames = [frame for frame in current_frames if frame not in remove_frames]
    
    # Add frames that are specified (if not already present)
    for a in add_frames:
        if a not in new_frames:
            new_frames.append(a)
    
    # Check if the new selection exceeds max_frames
    if len(new_frames) > max_frames:
        new_frames = new_frames[:max_frames]
        is_done = True
    else:
        is_done = False
    
    
    return new_frames, is_done
