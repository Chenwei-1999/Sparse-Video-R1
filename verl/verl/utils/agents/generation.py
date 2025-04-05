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
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
        ))
        self.max_prompt_length = config.max_prompt_length
        self.truncation = 'error'
    
    def run_llm_loop(self, gen_batch,
                    ) -> Tuple[Dict, Dict]:
        rollings = deepcopy(gen_batch)
        questions = [rollings.non_tensor_batch.get("question")[i] for i in range(rollings.batch['input_ids'].shape[0])]

        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        original_inputs= {'input_ids': gen_batch.batch['input_ids'], 
                          'position_ids': gen_batch.batch['attention_mask'],  # position_ids are same as attention_mask in this context
                          'attention_mask': gen_batch.batch['attention_mask'],
                          'multi_modal_inputs': gen_batch.non_tensor_batch.get("multi_modal_inputs", None), 
                          'raw_prompt_ids': gen_batch.non_tensor_batch.get("raw_prompt_ids", None),
                          'multi_modal_data': gen_batch.non_tensor_batch.get("multi_modal_data", None),}
        # Main generation loop
        for step in range(self.max_rounds):
            if not active_mask.sum():
                break
            rollings_active = DataProto.from_single_dict({
                **{k: v[active_mask] for k, v in rollings.batch.items()},
                **{k: v[active_mask] for k, v in rollings.non_tensor_batch.items()},
            })

            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info 
            if step == self.max_rounds - 1:
                break
            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            next_obs, dones = self.execute_predictions(
                responses_str, 
                rollings.non_tensor_batch.get("times")  # current_frames_list
            )
            active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_num_list.append(active_mask.sum().item())
            # print(f"current frames: {rollings.non_tensor_batch.get('times', [[]])}")
            # print(f'actions taken: {responses_str}')
            # print(f"Next obs: {next_obs}, dones: {dones}, active count: {active_mask.sum().item()} at step {step+1}/{self.max_rounds}")

            video_path = rollings.non_tensor_batch.get("video_path")
            height = rollings.non_tensor_batch.get("height")  
            width = rollings.non_tensor_batch.get("width")
            sampled_frames_batch = self.sample_frames(video_path, next_obs, dones, height, width)
            current_times = rollings.non_tensor_batch.get("times", [[]] * len(dones))
            new_times = []
            for i in range(len(dones)):
                if not dones[i]:
                    # Parse from obs
                    match = re.search(r'\[([^\]]+)\]', next_obs[i])
                    if match:
                        frames_str = match.group(1)
                        ts = [float(x.strip()) for x in re.split(r'[,\s]+', frames_str) if x.strip()]
                    else:
                        ts = current_times[i]  # fallback
                    new_times.append(ts)
                else:
                    # Done samples keep their current times
                    new_times.append(current_times[i])

            rollings = self.update_rollings_state(
                rollings,
                questions=questions,
                sampled_frames_batch=sampled_frames_batch,
                new_times=new_times,
                new_round=step + 1
            )
        final_output  = {"responses": gen_output.batch['responses'],}
        return self._compose_final_output(original_inputs, final_output, meta_info)
        
    def _compose_final_output(self, original_inputs: Dict,
                              final_output: Dict,
                              meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""

        # Preserve original 'responses' from final_output input
        responses = final_output.get("responses", None)
        if responses is None:
            raise ValueError("Missing 'responses' in final_output")

        # Add everything explicitly
        final_output['responses'] = responses
        final_output['input_ids'] = original_inputs['input_ids']
        final_output['multi_modal_inputs'] = original_inputs.get('multi_modal_inputs', None)
        final_output['multi_modal_data'] = original_inputs.get('multi_modal_data', None)
        final_output['raw_prompt_ids'] = original_inputs.get('raw_prompt_ids', None)
        final_output['attention_mask'] = original_inputs['attention_mask']
        final_output['position_ids'] = original_inputs['position_ids']

        final_output = DataProto.from_single_dict(final_output)
        final_output.meta_info.update(meta_info)

        return final_output

    
    def update_rollings_state(self,
                              rollings,
                              questions: List[str],
                              sampled_frames_batch: List[List[Dict]],
                              new_times: List[List[float]],
                              new_round: int) -> "DataProto":
        # --- 1. Update non-tensor fields: times and round.
        rollings.non_tensor_batch["times"] = new_times
        rollings.non_tensor_batch["round"] = new_round

        batch_size = rollings.batch["input_ids"].shape[0]
        # Initialize per-sample history if needed.
        prev_frames = rollings.non_tensor_batch.get("previous_frames", None)
        if prev_frames is None or len(prev_frames) != batch_size:
            prev_frames = [[] for _ in range(batch_size)]
        prev_rounds = rollings.non_tensor_batch.get("previous_rounds", None)
        if prev_rounds is None or len(prev_rounds) != batch_size:
            prev_rounds = [[] for _ in range(batch_size)]
        # Update per-sample history.
        for i in range(batch_size):
            prev_frames[i].append(new_times[i])
            # Optionally, update prev_rounds[i] with current actions if available.
        rollings.non_tensor_batch["previous_frames"] = prev_frames
        rollings.non_tensor_batch["previous_rounds"] = prev_rounds


        is_multi_modal = rollings.non_tensor_batch.get("is_multi_modal", False)

        for i in range(batch_size):
            # If the sample is done, skip updating (or copy previous state)
            if sampled_frames_batch[i] is None:
                # updated_input_ids.append(rollings.batch["input_ids"][i])
                # updated_attention_mask.append(rollings.batch["attention_mask"][i])
                # updated_position_ids.append(rollings.batch["position_ids"][i])
                # updated_raw_prompt_ids.append(rollings.non_tensor_batch["raw_prompt_ids"][i])
                continue

            sample_question = questions[i]
            sample_times = new_times[i]
            sample_prev_frames = prev_frames[i]  # per-sample history
            sample_prev_rounds = prev_rounds[i]  # per-sample history
            prompt = generate_prompt(
                question=sample_question,
                timestamps=sample_times,
                n_round=new_round,
                max_rounds=self.max_rounds,
                max_frames=self.max_frames,
                previous_frames=sample_prev_frames,
                previous_rounds=sample_prev_rounds
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
            if is_multi_modal:
                if sample_frames is not None:
                    raw_prompt = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|><|image_pad|><|vision_end|>'
                    )
                    processed_images = [process_image(frame_dict['image'])
                                        for frame_dict in sample_frames]
                    rollings.non_tensor_batch["multi_modal_data"] = {"image": processed_images}
                    image_inputs = self.processor.image_processor(
                        rollings.non_tensor_batch["multi_modal_data"]["image"],
                        return_tensors='pt'
                    )
                    rollings.non_tensor_batch["multi_modal_inputs"] = {k: v for k, v in image_inputs.items()}
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
                    raw_prompt_final = raw_prompt
                else:
                    raw_prompt_final = prompt_with_chat_template
            else:
                raw_prompt_final = prompt_with_chat_template

            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=self.tokenizer,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation
            )
            if is_multi_modal:
                from verl.models.transformers.qwen2_vl import get_rope_index
                pos_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask[0]
                )
            else:
                pos_ids = compute_position_id_with_mask(attention_mask)

            # updated_input_ids.append(input_ids[0])
            # updated_attention_mask.append(attention_mask[0])
            # updated_position_ids.append(pos_ids[0])
            # updated_raw_prompt_ids.append(self.tokenizer.encode(raw_prompt_final, add_special_tokens=False))
            # updated_raw_prompts.append(raw_prompt_final)
            rollings.batch["input_ids"][i] = input_ids[0]
            rollings.batch["attention_mask"][i] = attention_mask[0]
            rollings.batch["position_ids"][i] = pos_ids[0]
            rollings.non_tensor_batch["raw_prompt_ids"][i] = self.tokenizer.encode(raw_prompt_final, add_special_tokens=False)
            rollings.non_tensor_batch["raw_prompt"][i] = raw_prompt_final
            # rollings.batch["attention_mask"] = torch.stack(updated_attention_mask, dim=0)
            # rollings.batch["position_ids"] = torch.stack(updated_position_ids, dim=0)
            # rollings.non_tensor_batch["raw_prompt_ids"] = updated_raw_prompt_ids
            # rollings.non_tensor_batch["raw_prompt"] = updated_raw_prompts
        return rollings



    
    def sample_frames(self,
                      video_paths: List[str],
                      next_obs: List[str],
                      dones: List[bool],
                      heights: List[int],
                      widths: List[int]) -> List[List[Dict]]:
        """
        For each sample, if not done, parse the next_obs string to extract frame timestamps,
        then sample frames from the video using sample_frames_from_next_obs.

        Args:
            video_paths (List[str]): A list of video paths (one per sample).
            next_obs (List[str]): A list of observation strings (e.g., "Selected frames: [2, 3, 4]").
            dones (List[bool]): A list of booleans indicating whether each sample is done.
            heights (List[int]): A list of desired heights (one per sample).
            widths (List[int]): A list of desired widths (one per sample).
        
        Returns:
            List[List[Dict]]: A list (per sample) of lists of dictionaries for each sampled frame.
                              Each dictionary has keys like 'image' and 'timestamp'.
        """
        pattern = re.compile(r'\[([^\]]+)\]')
        new_times = []
        sampled_frames_batch = []
        
        # Iterate over each sample using zip.
        for idx, (obs, done) in enumerate(zip(next_obs, dones)):
            # Parse the frame timestamps from the observation string.
            match = pattern.search(obs)
            if match:
                frames_str = match.group(1)
                ts = [float(x.strip()) for x in re.split(r'[,\s]+', frames_str) if x.strip()]
            else:
                ts = []  # Fall back to an empty list if parsing fails.
            new_times.append(ts)
            
            # For samples not done, sample frames from the corresponding video.
            if not done:
                sampled_frames = sample_frames_from_next_obs(
                    video_paths[idx],
                    obs,
                    heights[idx],
                    widths[idx]
                )
            else:
                sampled_frames = None
            sampled_frames_batch.append(sampled_frames)
        
        # Optionally, if you need to update any tracking structure with new_times,
        # you can do so here.
        
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
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        print(active_batch.batch.keys())
        print(active_batch.non_tensor_batch.keys())
        gen_batch = active_batch.pop(
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
            
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in gen_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
            
        padded_gen_batch = DataProto.from_dict(padded_batch)
        
        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_gen_batch)
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch['attention_mask', 'input_ids', 'position_ids']
        return padded_output
     
    # def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
    #                         next_obs_ids: torch.Tensor) -> Dict:
    #     """Update rolling state with new responses and observations."""
    #     # Concatenate and handle padding
    #     new_input_ids = self.tensor_fn.concatenate_with_padding([
    #         rollings.batch['input_ids'],
    #         cur_responses,
    #         next_obs_ids
    #     ])
        
    #     # Create attention mask and position ids
    #     new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
    #     new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

    #     # Cut to appropriate length
    #     effective_len = new_attention_mask.sum(dim=1).max()
    #     max_len = min(self.max_prompt_length, effective_len)
        
    #     return DataProto.from_dict({
    #         'input_ids': new_input_ids[:, -max_len:],
    #         'position_ids': new_position_ids[:, -max_len:],
    #         'attention_mask': new_attention_mask[:, -max_len:]
    #     })

    
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
            print("Extracted answer tags:", responses_str)
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
          - next_obs: a list of strings representing the new selected frames.
          - dones: a list of booleans indicating if the round is done.
        
        (In your run_llm_loop, you can pass rollings.batch["times"] as current_frames_list.)
        """
        next_obs = []
        dones = []
        
        for curr_frames, response in zip(current_frames_list, responses_str):
            answer = extract_answer(response)
            if answer is None:
                # No answer tag found; treat as final answer.
                next_obs.append(f"Selected frames: {curr_frames}")
                dones.append(True)
                continue
            
            # Parse add and remove instructions using regex.
            add_frames = []
            remove_frames = []
            
            add_match = re.search(r'\+\[(.*?)\]', answer)
            if add_match:
                add_frames = [int(x.strip()) for x in add_match.group(1).split(',') if x.strip().isdigit()]
                
            remove_match = re.search(r'-\[(.*?)\]', answer)
            if remove_match:
                remove_frames = [int(x.strip()) for x in remove_match.group(1).split(',') if x.strip().isdigit()]
            
            # If there are no add instructions, we consider this a final answer.
            if not add_frames:
                next_obs.append(f"Selected frames: {curr_frames}")
                dones.append(True)
                continue
            
            # If the number of add frames exceeds the maximum allowed, mark as done.
            if len(add_frames) > self.max_frames:
                next_obs.append(f"Selected frames: {curr_frames}")
                dones.append(True)
                continue
            
            # Update the current frames using the helper.
            new_frames, is_done = update_frames(curr_frames, add_frames, remove_frames, self.max_frames)
            next_obs.append(f"Selected frames: {new_frames}")
            dones.append(is_done)
        
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


def reward_function(trajectory: List[Dict[str, Any]], ground_truth: str, final_round: bool) -> float:
    """
    Compute a reward for the given trajectory based on the following rules:
    
      - If no answer is found (i.e. no <answer> tag or an empty answer), return -1.
      - If the answer contains only removal instructions (i.e. has a "-[...]" but no "+[...]"),
        return -1.
      - If the answer contains add instructions (i.e. "+[...]"), meaning the agent is still exploring,
        return 0.1.
      - Otherwise, treat the answer as final: if it matches the ground truth (after normalization),
        return 1; otherwise, return 0.
    
    Once a sample’s answer is final (or we are at the final round), the computed reward is stored
    in the last round dictionary and immediately returned in subsequent rounds.
    
    Args:
        trajectory: List of round dictionaries (each with a "response" key containing the LLM output).
        ground_truth: The expected final answer as a string.
        final_round: Boolean flag indicating if the current round is the final round.
    
    Returns:
        A float reward computed based on the above rules.
    """
    if not trajectory:
        return -1  # No rounds means no answer.
    
    # Use the last round's response as the final output.
    last_round = trajectory[-1]
    # If reward was already computed and fixed, return it.
    if "reward" in last_round:
        return last_round["reward"]
    
    final_response = last_round.get('response', '')
    answer = extract_answer(final_response)
    
    if not answer:
        computed_reward = -1
    else:
        # Check for exploration instructions.
        has_add = bool(re.search(r'\+\[.*?\]', answer))
        has_remove = bool(re.search(r'-\[(.*?)\]', answer))
        
        # If the answer contains only removal instructions (no add), consider it unformulated.
        if has_remove and not has_add:
            computed_reward = -1
        # If the answer shows exploration, return a small reward.
        elif has_add:
            computed_reward = 0.1
        else:
            # Otherwise, treat the answer as final.
            normalized_answer = answer.lower().strip()
            normalized_truth = ground_truth.lower().strip()
            computed_reward = 1 if normalized_answer == normalized_truth else 0
    
    # If we are in the final round, fix the reward for this sample.
    if final_round:
        last_round["reward"] = computed_reward
        
    return computed_reward
