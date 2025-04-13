import torch
import numpy as np
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from .frames_sampler import sample_frames_from_next_obs
from .construct_prompt import generate_prompt
from verl import DataProto
from verl.utils.tracking import Tracking
import verl.utils.torch_functional as verl_F
from copy import deepcopy
from tensordict import TensorDict
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))
    if isinstance(image, bytes):
        # base64 encoded image
        image = Image.open(BytesIO(image))

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
        self.truncation = 'error'
        
    def run_llm_loop(self, gen_batch, global_steps=None) -> Tuple[Dict, Dict]:
        # Make a deep copy of the input batch and save the original inputs.
        rollings = deepcopy(gen_batch)
        # Filter tensors in rollings using global_indices
        batch_size = rollings.batch['input_ids'].shape[0]
        rounds_per_sample = [0] * batch_size
        final_response_ids_list = [None] * batch_size
        original_inputs = {
            'input_ids': gen_batch.batch['input_ids'],
            'position_ids': gen_batch.batch['position_ids'],
            'attention_mask': gen_batch.batch['attention_mask'], 
            'reward_model': gen_batch.non_tensor_batch['reward_model'],
            'data_source': gen_batch.non_tensor_batch['data_source'], 
            'extra_info': gen_batch.non_tensor_batch.get('extra_info', {}),
        }
        rollings.non_tensor_batch['batch_indices'] = list(range(batch_size))
        rollings.non_tensor_batch["previous_times"] = [[gen_batch.non_tensor_batch['times'][i]] for i in range(batch_size)]
        # dones is local for the current step, initialized to False
        dones = [False] * batch_size 
          # extra_info (dict, optional): Dictionary containing additional info like:
          #             'timestamps' (list): Current frame timestamps. Required for modification checks.
          #             'max_frames' (int): Maximum allowed frames. Required for modification checks.
          #             'current_turn' (int): The current turn number. Required for max_turns check.
          #             'max_turns' (int): The maximum number of turns allowed. Required for max_turns check.
          #             'iter_decay' (float): Decay factor for iterations (optional).
          #             'n_frames' (int): Number of frames used (optional, for penalty, distinct from len(timestamps)).
          #             'frame_decay' (float): Decay factor for frames used (optional).
          #             'type' (str): Type of the data source, val for validation data.
                  # Initialize global_indices to map active samples back to the original batch.
        meta_info = {}
        
        for step in range(1, self.max_rounds+1):
            # rollings already contains only the active samples.
            # Generate responses for the active samples.

            gen_output = self._generate_with_gpu_padding(rollings)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            # If this is the final round, record responses for all active samples.
            if step == self.max_rounds:
                # mappings
                # i is the index in the current rollings batch
                # idx is the original index in the global batch
                for i, idx in enumerate(rollings.non_tensor_batch['batch_indices']):
                    if final_response_ids_list[idx] is None:
                        final_response_ids_list[idx] = gen_output.batch['responses'][i]
                        extra_info = {}
                        extra_info['timestamps'] = rollings.non_tensor_batch['previous_times'][i][-1]  # last round timestamps
                        extra_info['max_frames'] = self.max_frames
                        extra_info['current_turn'] = step
                        extra_info['max_turns'] = self.max_rounds
<<<<<<< Updated upstream
                        original_inputs['extra_info'][idx].update(extra_info)
=======
                    
                        original_inputs['extra_info'][i].update(extra_info)
>>>>>>> Stashed changes
                break
            
            # Get current times for active samples; if missing, default to empty lists.
            current_times = rollings.non_tensor_batch['times']

            # Execute predictions to get next observations and done flags.
            next_obs, dones = self.execute_predictions(responses_str, current_times)
<<<<<<< Updated upstream
=======
            for i, idx in enumerate(rollings.non_tensor_batch['batch_indices']):
                rounds_per_sample[idx] += 1
            print(f'step {step}/{self.max_rounds}: Generated responses for {len(responses_ids)} active samples.')
            print(f"There are {sum(dones)} done samples out of {len(dones)} total samples in this round.")
>>>>>>> Stashed changes
            # Record responses for samples that are done.
            for i, idx in enumerate(rollings.non_tensor_batch['batch_indices']):
                if dones[i]:
                    final_response_ids_list[idx] = gen_output.batch['responses'][i]
                    extra_info = {}
                    extra_info['timestamps'] = rollings.non_tensor_batch['previous_times'][i][-1]  # last round timestamps
                    extra_info['max_frames'] = self.max_frames
                    extra_info['current_turn'] = step
                    extra_info['max_turns'] = self.max_rounds
                    original_inputs['extra_info'][idx].update(extra_info)
                    # extra_info['n_frames'] = len(rollings.non_tensor_batch['previous_times'][i][-1])
            # Filter out done samples: update global_indices.
            if sum(dones) == len(dones):
                # All samples are done, break the loop.
                break
            # only keep things not done
            video_path = [rollings.non_tensor_batch['video_path'][i] for i, done in enumerate(dones) if not done]
            height = [rollings.non_tensor_batch['height'][i] for i, done in enumerate(dones) if not done]
            width = [rollings.non_tensor_batch['width'][i] for i, done in enumerate(dones) if not done]

            sampled_frames_batch = self.sample_frames(video_path, next_obs, height, width)
            # Update the rollings state for the remaining active samples.
            # Since rollings already represents the active subset, we update it directly.
            rollings = self.update_rollings_state(
                rollings,
                dones,
                sampled_frames_batch=sampled_frames_batch,
                new_times=next_obs,
                new_round=step + 1
            )

        final_response_ids_tensor = torch.stack(final_response_ids_list, dim=0)
        prefix = "val" if self.is_validation else "train"
        meta_info['rounds_per_sample'] = rounds_per_sample
        self.logger.log_rounds_per_sample(rounds_per_sample, prefix, global_steps)

        return self._compose_final_output(original_inputs, final_response_ids_tensor, meta_info)

        
    def _compose_final_output(self, original_inputs: Dict,
                              responses: torch.Tensor,
                              meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = {}
        # Preserve original 'responses' from final_output input
        if responses is None:
            raise ValueError("Missing 'responses' in final_output")
        # Add everything explicitly
        final_output['prompts'] = original_inputs['input_ids']
        final_output['responses'] = responses
    
        final_output['input_ids'] = torch.cat([
            original_inputs['input_ids'],
            final_output['responses']
        ], dim=1)

        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(original_inputs['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)

        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        final_output['reward_model'] = original_inputs['reward_model']
        final_output['data_source'] = original_inputs['data_source']
        final_output['extra_info'] = original_inputs.get('extra_info', {})
        final_output = DataProto.from_single_dict(final_output)
        final_output.meta_info.update(meta_info)

        return final_output

    def pad_to_length(self, tensor: torch.Tensor, target_length: int, pad_token_id: int) -> torch.Tensor:
        current_length = tensor.shape[0]
        if current_length < target_length:
            pad_size = target_length - current_length
            pad_tensor = torch.full((pad_size,), pad_token_id, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, pad_tensor], dim=0)
        else:
            return tensor[:target_length]    
        

    def update_rollings_state(self,
                              rollings,
                              dones, 
                              sampled_frames_batch: List[List[Dict]],
                              new_times: List[List[float]],
                              new_round: int) -> "DataProto":
        """
        Only keep the not-done samples in rollings and update their states.
        (This version avoids converting non-tensor fields to NumPy arrays.)
        """
        # Check that the lengths match.
        assert len(dones) == len(rollings.non_tensor_batch["times"]), \
            "Mismatch in dones and rollings.non_tensor_batch['times'] length."
        assert len(sampled_frames_batch) == len(new_times), \
            "Mismatch in sampled_frames_batch and new_times lengths."

        # Determine which samples to keep, only keep not done samples
        to_keep = [not done for done in dones]

        # Filter tensor fields.
        tensor_batch = {}
        for k, v in rollings.batch.items():
            mask = torch.tensor(to_keep, dtype=torch.bool, device=v.device)
            tensor_batch[k] = v[mask]

        # Filter non-tensor fields without converting to numpy.
        non_tensor_batch = {}
        for k, v in rollings.non_tensor_batch.items():
            # Here we assume that v is a list.
            non_tensor_batch[k] = [item for item, keep in zip(v, to_keep) if keep]

        # --- Instead of using DataProto.from_dict (which converts non_tensor_batch),
        # we build a new DataProto manually.
        #
        # We use __new__ to bypass __post_init__ (which would enforce NumPy conversion)
        new_rollings = DataProto.__new__(DataProto)
        # Rebuild the tensor dict manually.
        # (Assume that at least one key exists; adjust if needed.)
        new_batch_size = tensor_batch[next(iter(tensor_batch))].shape[0] if tensor_batch else 0
        new_rollings.batch = TensorDict(source=tensor_batch, batch_size=(new_batch_size,))
        new_rollings.non_tensor_batch = non_tensor_batch
        new_rollings.meta_info = rollings.meta_info

        # --- Now update non-tensor fields for each remaining sample.
        #
        # Append the current times to previous_times.
        new_rollings.non_tensor_batch["previous_times"].append(new_rollings.non_tensor_batch["times"])
        
        batch_size = len(new_times)
        for i in range(batch_size):
            sample_question = new_rollings.non_tensor_batch['question'][i]
            sample_times = new_times[i]
            # Append the new times to the per-sample history.
            new_rollings.non_tensor_batch["previous_times"][i].append(sample_times)
            sample_prev_times = new_rollings.non_tensor_batch["previous_times"][i]
            prompt = generate_prompt(
                question=sample_question,
                timestamps=sample_times,
                n_round=new_round,
                max_rounds=self.max_rounds,
                max_frames=self.max_frames,
                previous_frames=sample_prev_times,
            )
            sample_frames = sampled_frames_batch[i]
            num_frames = len(sample_frames) if sample_frames is not None else 0
            chat = [{
                "role": "user",
                "content": ("<image>" * num_frames) + prompt
            }]
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )
            
            # If multi-modal input is needed, update the prompt and process images.
            if new_rollings.non_tensor_batch['is_multi_modal'][0]:
                raw_prompt = prompt_with_chat_template.replace(
                    '<image>',
                    '<|vision_start|><|image_pad|><|vision_end|>'
                )
                processed_images = [process_image(frame_dict) for frame_dict in sample_frames]
                new_rollings.non_tensor_batch["multi_modal_data"][i] = {"image": processed_images}
                image_inputs = self.processor.image_processor(
                    new_rollings.non_tensor_batch["multi_modal_data"][i]["image"],
                    return_tensors='pt'
                )
                new_rollings.non_tensor_batch["multi_modal_inputs"][i] = {k: v for k, v in image_inputs.items()}
                image_grid_thw = image_inputs.get("image_grid_thw", None)
                if image_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size ** 2
                    index = 0
                    while '<image>' in prompt_with_chat_template:
                        prompt_with_chat_template = prompt_with_chat_template.replace(
                            '<image>',
                            '<|vision_start|>' +
                            '<|placeholder|>' * (int(image_grid_thw[index].prod() // merge_length)) +
                            '<|vision_end|>',
                            1,
                        )
                        index += 1
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<|placeholder|>', self.processor.image_token
                    )
            else:
                raw_prompt = prompt_with_chat_template

            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=self.tokenizer,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation
            )
            if new_rollings.non_tensor_batch['is_multi_modal'][0]:
                from verl.models.transformers.qwen2_vl import get_rope_index
                pos_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask[0]
                )
            else:
                pos_ids = compute_position_id_with_mask(attention_mask)

            # Update the tensor batch for sample i.
            new_rollings.batch["input_ids"][i] = input_ids[0]
            new_rollings.batch["attention_mask"][i] = attention_mask[0]
            new_rollings.batch["position_ids"][i] = pos_ids[0]
            new_rollings.non_tensor_batch["raw_prompt_ids"][i] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        return new_rollings

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
    

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
        Wrapper for generation that handles multi-GPU padding requirements.
        If num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
        If active_batch size is not divisible by num_gpus, pad with first sequence
        then remove padding from output.
        """
        # Deep copy the active batch and pop the keys needed for generation.
        gen_batch = deepcopy(active_batch)
        gen_batch = gen_batch.pop(
            batch_keys=['input_ids', 'attention_mask', 'position_ids'],
            non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
        )
        num_gpus = self.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(gen_batch)
        
        batch_size = gen_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(gen_batch)
        
        # Compute padding size.
        padding_size = num_gpus - remainder

        # Pad tensor fields.
        padded_batch = {}
        for k, v in gen_batch.batch.items():
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        # Pad non-tensor fields.
        padded_non_tensor = {}
        for k, v in gen_batch.non_tensor_batch.items():
            # Ensure v is a list.
            if isinstance(v, np.ndarray):
                v = list(v)
            if isinstance(v, list):
                padded_non_tensor[k] = v + [v[0]] * padding_size
            else:
                raise TypeError(f"Non-tensor field {k} must be a list or numpy array.")
        # Convert padded non-tensor fields to np.ndarray with dtype=object.
        padded_non_tensor = {k: np.array(v, dtype=object) for k, v in padded_non_tensor.items()}

        # Rebuild DataProto manually to avoid unwanted conversion.
        padded_gen_batch = DataProto.__new__(DataProto)
        padded_gen_batch.batch = TensorDict(
            source=padded_batch,
            batch_size=(padded_batch["input_ids"].shape[0],)
        )
        padded_gen_batch.non_tensor_batch = padded_non_tensor
        padded_gen_batch.meta_info = gen_batch.meta_info
        # Generate sequences with the padded batch.
        padded_output = self.actor_rollout_wg.generate_sequences(padded_gen_batch)
        print(f"{padded_gen_batch.batch['input_ids'].shape[0]} sequences generated with padding, removing {padding_size} padding sequences.")
        print(f"Generated sequences with padding: {batch_size} -> {padded_output.batch['responses'].shape[0]} sequences.")
        # Remove padding from tensor fields for all keys.
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        print(f"Trimmed batch size: {len(trimmed_batch['responses'])} sequences after removing padding.")
        padded_output.batch = trimmed_batch
        
        # Remove padding from non-tensor fields.
        trimmed_non_tensor = {}
        for k, v in padded_output.non_tensor_batch.items():
            print("non_tensor_keys", k)
            if isinstance(v, np.ndarray):
                trimmed_non_tensor[k] = v[:-padding_size]
            else:
                trimmed_non_tensor[k] = v  # Fallback if not an array.
        padded_output.non_tensor_batch = trimmed_non_tensor
        
        # Trim meta_info if needed.
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta

        return padded_output

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to remove 1. multiple answers or 2. reward hacking attempts."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )
        # responses_str = [resp.split('</answer>')[0] + '</answer>' 
        #             if '</answer>' in resp else resp 
        #             for resp in responses_str]
        responses_str = self._process_answer_tag(responses_str)
        
        if self.no_think_rl:
            # Extract only the content inside the first <answer>...</answer> tag
            processed = []
            for resp in responses_str:
                match = re.search(r"<answer>(.*?)</answer>", resp)
                if match:
                    # Only keep the answer tag with its content
                    processed.append(f"<answer>{match.group(1)}</answer>")
                else:
                    # If no answer tag is found, leave the response unchanged
                    processed.append(resp)
            responses_str = processed
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']
    

    def _process_answer_tag(self, responses_str: List[str]):
        """
        Process a list of response strings to keep only the first <answer></answer> tag pair
        while preserving the rest of the string content.
        
        Args:
            responses_str (List[str]): List of response strings potentially containing answer tags
            
        Returns:
            List[str]: Processed responses with only first answer tag pair preserved
        """
        def process_single_response(resp):
            # If no answer tags present, return original string
            if '<answer>' not in resp or '</answer>' not in resp:
                return resp
                
            # Find the first complete <answer> tag pair
            pattern = r'<answer>.*?</answer>'
            match = re.search(pattern, resp, re.DOTALL)
            
            if not match:
                return resp
                
            # Get the matched answer tag content
            answer_content = match.group(0)
            
            # Replace all subsequent answer tag pairs with their content
            rest_of_string = resp[match.end():]
            cleaned_rest = re.sub(r'<answer>(.*?)</answer>', r'\1', rest_of_string, flags=re.DOTALL)
            
            return resp[:match.start()] + answer_content + cleaned_rest
        
        # Process each response string
        return [process_single_response(resp) for resp in responses_str]


    
    def execute_predictions(self, responses_str: List[str], current_frames_list: List[List[int]]) -> Tuple[List[str], List[bool]]:
        """
        Custom execute_predictions for a frame-selection task.
        
        For each sample:
          - It extracts the content of the first <answer> tag.
          - It then looks for frame update instructions:
              * If the answer contains add instructions (e.g. "+[3,4]") and/or remove instructions (e.g. "-[2,5]"),
                those lists are parsed.
          - The round is considered done (and no update is applied) if:
              1. There are no add frames (i.e. only remove instructions or nothing).
              2. Neither add nor remove instructions are present.
              3. The number of add frames exceeds the allowed max_frames.
          - Otherwise, update the current frame selection using update_frames.
        
        The function returns:
          - next_obs: a list of strings representing the new selected frames, len(next_obs) <= len(dones).
          - dones: a list of booleans indicating if the round is done, len(dones) = len(responses_str).
        """
        next_obs = []
        dones = []

        for curr_frames, response in zip(current_frames_list, responses_str):
            answer = extract_answer(response)
            if answer is None:
                # No answer tag found; treat as final answer.
                dones.append(True)
                continue
            
            # Parse add and remove instructions using regex.
            add_frames = []
            remove_frames = []
            
            # if no frames are selected, treat it as done
            add_match = re.search(r'\+\[(.*?)\]', answer)
            if add_match:
                add_frames = [int(x.strip()) for x in add_match.group(1).split(',') if x.strip().isdigit()]
            else:
                dones.append(True)
                continue
            
            remove_match = re.search(r'-\[(.*?)\]', answer)
            if remove_match:
                remove_frames = [int(x.strip()) for x in remove_match.group(1).split(',') if x.strip().isdigit()]
            # If the number of frames exceeds the maximum allowed, mark as done.
            if len(add_frames) + len(curr_frames) -len(remove_frames)> self.max_frames:
                dones.append(True)
                continue
            
            # Update the current frames using the helper.
            new_frames, is_done = update_frames(curr_frames, add_frames, remove_frames, self.max_frames)
            if is_done:
                dones.append(True)
                continue
            next_obs.append(new_frames)
            dones.append(False)
            
        
        return next_obs, dones  # Return the add/remove frames for logging if needed

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
    
    # If the updated selection exceeds max_frames, we cannot proceed further.
    if len(new_frames) > max_frames:
        return new_frames, True
    
    return new_frames, False

def extract_answer(text: str) -> str:
    """
    Extract the content of the first <answer>...</answer> tag from the text.
    Returns None if no answer tag is found.
    """
    match = re.search(r"<answer>(.*?)</answer>", text)
    if match:
        return match.group(1).strip()
    return None
