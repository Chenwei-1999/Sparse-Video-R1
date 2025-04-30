import torch
import random
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
from verl.utils.agents.reward_function import extract_solution, parse_frame_list
from verl.models.transformers.qwen2_vl import get_rope_index
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
        rollings = deepcopy(gen_batch)
        batch_size = rollings.batch['input_ids'].shape[0]
        final_response_ids_list = [None] * batch_size

        original_inputs = {
            'input_ids': gen_batch.batch['input_ids'],
            'position_ids': gen_batch.batch['position_ids'],
            'attention_mask': gen_batch.batch['attention_mask'],
            'reward_model': gen_batch.non_tensor_batch['extra_info'][0].get('reward_model'),
            'data_source': gen_batch.non_tensor_batch['extra_info'][0].get('data_source'),
            'extra_info': gen_batch.non_tensor_batch.get('extra_info', {}),
        }

        dones = [False] * batch_size
        meta_info = {}

        for step in range(1, self.max_rounds + 1):
            # 1) generate
            gen_output = self._generate_with_gpu_padding(rollings)
            responses_ids, responses_str = self._postprocess_responses(
                gen_output.batch['responses']
            )

            # 2) error/correction
            current_times = [rollings.non_tensor_batch['extra_info'][i]['times'] for i in range(batch_size)]
            correction_info = self.correction(
                responses_str, current_times
            )
            correction_prompts = self.correction_prompt(correction_info)
            
            # Store correction prompts in extra_info
            for i in range(batch_size):
                rollings.non_tensor_batch['extra_info'][i]['correction_prompt'] = correction_prompts[i]

            # 3) mark done & collect final_response_ids_list
            if step == self.max_rounds:
                dones = [True] * batch_size
            else:
                next_obs, dones = self.execute_predictions(
                    responses_str,
                    current_times,
                    correction_info
                )

            for i in range(batch_size):
                if dones[i]:
                    final_response_ids_list[i] = responses_ids[i]
                    extra = {
                        'timestamps': rollings.non_tensor_batch['extra_info'][i]['previous_times'][-1],
                        'max_frames': self.max_frames,
                        'current_turn': step,
                        'max_turns': self.max_rounds,
                    }
                    original_inputs['extra_info'][i].update(extra)

            if all(dones):
                break

            # 5) build a full‐length sampled_frames list
            sampled_frames_batch: List[List[Dict]] = []
            for i in range(batch_size):
                if dones[i]:
                    sampled_frames_batch.append([])   # placeholder for done
                else:
                    sampled_frames_batch.append(
                        sample_frames_from_next_obs(
                            rollings.non_tensor_batch['extra_info'][i]['video_path'],
                            next_obs[i],
                            rollings.non_tensor_batch['extra_info'][i]['height'],
                            rollings.non_tensor_batch['extra_info'][i]['width'],
                            ratio=self.ratio
                        )
                    )

            # 6) update _all_ samples in place (no filtering)
            rollings = self.update_rollings_state(
                rollings,
                dones,
                sampled_frames_batch=sampled_frames_batch,
                new_times=next_obs,
                new_round=step + 1
            )

        final_response_ids_tensor = torch.stack(final_response_ids_list, dim=0)
        return self._compose_final_output(
            original_inputs, final_response_ids_tensor, meta_info
        )

        
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
        

    def update_rollings_state(
        self,
        rollings: DataProto,
        dones: List[bool],
        sampled_frames_batch: List[List[Dict]],
        new_times: List[List[float]],
        new_round: int
    ) -> DataProto:
        """
        Keep every sample in the DataProto.  For done samples, we simply
        carry forward their last state; for active samples, we update in place.
        """
        # Carry over everything by deep copy
        new_rollings = deepcopy(rollings)

        # Update times in extra_info for each sample
        for i in range(len(dones)):
            new_rollings.non_tensor_batch['extra_info'][i]['times'] = new_times[i]

        for i in range(len(dones)):
            # if done, leave new_rollings.batch[*][i] and non_tensor fields unchanged
            if dones[i]:
                continue

            # 1) extend histories
            new_rollings.non_tensor_batch['extra_info'][i]['previous_times'].append(new_times[i])

            # 2) rebuild prompt
            question = new_rollings.non_tensor_batch['extra_info'][i]['question']
            prev_times = new_rollings.non_tensor_batch['extra_info'][i]['previous_times']
            corr = new_rollings.non_tensor_batch['extra_info'][i].get('correction_prompt', '')

            base = generate_prompt(
                question=question,
                timestamps=new_times[i],
                n_round=new_round,
                max_rounds=self.max_rounds,
                max_frames=self.max_frames,
                previous_frames=prev_times,
            )
            full = f"{corr}\n\n{base}" if corr else base
            if random.randint(1, 20) == 10:
                print(f'full: {full}')
            # 3) build chat & template
            num_frames = len(sampled_frames_batch[i])
            chat = [{"role":"user","content":("<image>"*num_frames)+full}]
            templ = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )

            # 4) multimodal if needed
            if new_rollings.non_tensor_batch['extra_info'][i].get('is_multi_modal', False):
                raw = templ.replace(
                    '<image>','<|vision_start|><|image_pad|><|vision_end|>'
                )
                imgs = [process_image(f) for f in sampled_frames_batch[i]]
                new_rollings.non_tensor_batch['extra_info'][i]['multi_modal_data'] = {"image": imgs}
                img_inputs = self.processor.image_processor(imgs, return_tensors='pt')
                new_rollings.non_tensor_batch['extra_info'][i]['multi_modal_inputs'] = img_inputs

                input_ids, attn_mask = verl_F.tokenize_and_postprocess_data(
                    prompt=templ,
                    tokenizer=self.tokenizer,
                    max_length=self.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.truncation
                )
                pos_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=img_inputs['image_grid_thw'],
                    attention_mask=attn_mask[0]
                )
            else:
                raw = templ
                input_ids, attn_mask = verl_F.tokenize_and_postprocess_data(
                    prompt=templ,
                    tokenizer=self.tokenizer,
                    max_length=self.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.truncation
                )
                pos_ids = compute_position_id_with_mask(attn_mask)

            # 5) write tokens back into the i-th slot
            new_rollings.batch['input_ids'][i]      = input_ids[0]
            new_rollings.batch['attention_mask'][i] = attn_mask[0]
            new_rollings.batch['position_ids'][i]   = pos_ids[0]
            new_rollings.non_tensor_batch['raw_prompt_ids'][i] = (
                self.tokenizer.encode(raw, add_special_tokens=False)
            )

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

        # Remove padding from tensor fields for all keys.
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        padded_output.batch = trimmed_batch
        
        # Remove padding from non-tensor fields.
        trimmed_non_tensor = {}
        for k, v in padded_output.non_tensor_batch.items():
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
    
    def correction(self, responses_str: List[str], current_frames_list: List[List[int]]) -> List[Dict]:
        """Classify errors in responses"""
        correction_info = []
        for i, resp in enumerate(responses_str):
            # Get current frame state from rollings
            current_frames = current_frames_list[i]
            total_frames = self.max_frames
            
            error_type, error_msg, _ = extract_solution(
                resp, 
                total_frames=total_frames,
                current_frames=current_frames,
                simplified=False
            )
            
            # Map error types to correction messages
            error_map = {
                'format_error': "Response format incorrect",
                'think_error': "Missing reasoning section",
                'answer_error': "Missing answer tag",
                'invalid_answer': "Invalid multiple-choice selection",
                'frame_error': "Invalid frame operation"
            }
            
            if error_type in error_map:
                correction_info.append({
                    'error_type': error_type,
                    'message': f"{error_map[error_type]} - {error_msg}",
                    'original': resp
                })
            else:
                correction_info.append(None)
            
        return correction_info

    def correction_prompt(self, correction_info: List[Dict]) -> List[str]:
        """Generate corrective feedback prompts"""
        prompts = []
        template = (
            "This is your previous response: {original}\n"
            "It contains the following error: {message}\n"
            "Please follow the instructions to avoid the error."
        )
        
        for info in correction_info:
            if not info:
                prompts.append("")
                continue
            
            prompts.append(template.format(
                original=info['original'],
                message=info['message']
            ))
        
        return prompts

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


    def execute_predictions(
        self,
        responses_str: List[str],
        current_frames_list: List[List[int]],
        correction_info: List[Dict]
    ) -> Tuple[List[List[int]], List[bool]]:
        next_obs = []
        dones = []
        
        for i, (resp, curr_frames) in enumerate(zip(responses_str, current_frames_list)):
            if correction_info[i] is not None:
                next_obs.append(curr_frames)
                dones.append(False)
                continue

            answer = extract_answer(resp)
            if answer is None:
                next_obs.append(curr_frames)
                dones.append(True)
                continue

            add_match = re.search(r'\+\[(.*?)\]', answer)
            remove_match = re.search(r'-\[(.*?)\]', answer)
            add_frames = parse_frame_list(add_match.group(1)) if add_match else []
            remove_frames = parse_frame_list(remove_match.group(1)) if remove_match else []

            new_frames, is_done = update_frames(
                curr_frames, add_frames, remove_frames, self.max_frames
            )
            next_obs.append(new_frames)
            dones.append(is_done)
        
        # **Return the list you built, not an undefined name**
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

def extract_answer(text: str) -> str:
    """
    Extract the content of the first <answer>...</answer> tag from the text.
    Returns None if no answer tag is found.
    """
    match = re.search(r"<answer>(.*?)</answer>", text)
    if match:
        return match.group(1).strip()
    return None
