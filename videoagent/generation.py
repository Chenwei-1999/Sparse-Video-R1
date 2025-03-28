import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from dataset.NExT_QA import generate_prompt, sample_frames

from verl import DataProto
from verl.utils.tracking import Tracking
import verl.utils.torch_functional as verl_F

import shutil
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

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        processor,
        actor_rollout_wg,
        logger: Tracking,
        is_validation: bool = False,
        max_prompt_length: int = 10240,
        max_turns: int = 5,
        max_frames: int = 5,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_rollout_wg = actor_rollout_wg
        self.logger = logger 
        self.is_validation = is_validation
        self.max_turns = max_turns
        self.max_frames = max_frames
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=max_prompt_length,
        ))
        self.max_prompt_length = max_prompt_length
        self.truncation = 'error'
    
    def run_llm_loop(self, gen_batch,
                    ) -> Tuple[Dict, Dict]:

        
        rollings = gen_batch

        # Main generation loop
        for step in range(self.max_turns):
            gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            if step == self.max_turns - 1:
                # Final generation
                break
            # responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_str = self.tokenizer.batch_decode(
                gen_output.batch['responses'], 
                skip_special_tokens=True
            )
            responses_str = self._process_answer_tag(responses_str)
            for idx in range(len(responses_str)):
                response = responses_str[idx]
                if isinstance(response, dict):
                    n_frames = rollings.non_tensor_batch['num_frames'][idx] 
                    add_frames = response.get('add', [])
                    remove_frames = response.get('remove', [])
                    if n_frames + len(add_frames) - len(remove_frames) > self.max_frames:
                        continue
                    if len(add_frames) == 0:
                        continue
                    # non valid remove frames
                    valid_remove = True
                    for frame in remove_frames:
                        if frame not in rollings.non_tensor_batch['times'][idx]:
                            valid_remove = False
                    if not valid_remove:
                        continue
                    frames = rollings.non_tensor_batch['times'][idx].tolist() 
                    # get the non -1 frames from 'frames'
                    frames = [frame for frame in frames if frame != -1]
                    # add frames to original frames
                    for frame in add_frames:
                        if frame not in frames:
                            frames.append(frame)
                    # remove frames from original frames
                    for frame in remove_frames:
                        if frame in frames:
                            frames.remove(frame)
                    frames = sorted(frames)

                    while len(frames) < self.max_frames:
                        frames.append(-1)         
                    # sample frames
                    sampled_frames_base64, times = sample_frames(
                        video_path = rollings.non_tensor_batch['video_path'][idx],
                        frames=frames,
                    )
                    # generate prompt
                    q_prompt = rollings.non_tensor_batch['q_prompt'][idx]
                    full_prompt = generate_prompt(q_prompt, times, max_frames=self.max_frames)
                    prompt_with_chat_template = (
                    "<|im_start|>system\n"
                    "You are a helpful assistant.\n"
                    "<|im_end|>\n"
                    "<|im_start|>user\n"
                    f"{full_prompt}<|im_end|>\n"
                    "<|im_start|>assistant"
                   )
                    raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
                    multi_modal_data = {'image': [process_image(image) for image in sampled_frames_base64]}
                    image_inputs = self.processor.image_processor(multi_modal_data['image'], return_tensors='pt')
                    image_grid_thw = image_inputs['image_grid_thw'] # t: time, h: height, w: width
                    multi_modal_inputs = {key: val for key, val in image_inputs.items()}
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
                    from verl.models.transformers.qwen2_vl import get_rope_index

                    position_ids = get_rope_index(
                        self.processor,
                        input_ids=input_ids[0],
                        image_grid_thw=image_grid_thw,
                        attention_mask=attention_mask[0],
                    )  # (3, seq_len)
                    input_ids = input_ids[0]
                    attention_mask = attention_mask[0]
                    position_ids = position_ids[0]
                    raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
                    rollings.non_tensor_batch['raw_prompt_ids'][idx] = raw_prompt_ids
                    rollings.non_tensor_batch['multi_modal_data'][idx] = multi_modal_data
                    rollings.non_tensor_batch['multi_modal_inputs'][idx] = multi_modal_inputs
                    rollings.batch['input_ids'][idx] = input_ids
                    rollings.batch['attention_mask'][idx] = attention_mask
                    rollings.batch['position_ids'][idx] = position_ids
                    rollings.non_tensor_batch['times'][idx] = times
                    rollings.non_tensor_batch['num_frames'][idx] = len(times)
                    
            # responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
        return gen_output
    
    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.max_prompt_length, effective_len)
        
        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })

    
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
    @staticmethod
    def _process_answer_tag(responses_str):
        """
        Process a list of response strings to keep only the first <answer></answer> tag pair
        while preserving the rest of the string content.
        
        Args:
            responses_str (List[str]): List of response strings potentially containing answer tags
            
        Returns:
            List[str]: Processed responses with only first answer tag pair preserved
        """
        def process_single_response(text):
            match = re.search(r'<answer>(.*?)</answer>', text)
            
            if not match:
                return -1  # No <answer> tag found

            answer_content = match.group(1).strip()

            # Case 1: Direct numeric answer in the range 0-4
            if answer_content in ["0", "1", "2", "3", "4"]:
                return int(answer_content)
            
            # Case 2: Extracting frame modifications in the format +[x,y,z]-[a,b]
            add_match = re.search(r'\+\[(.*?)\]', answer_content)
            remove_match = re.search(r'\-\[(.*?)\]', answer_content)

            def extract_numbers(text):
                """Extract valid integers from a comma-separated string, ignoring non-numeric values."""
                if not text:
                    return []
                return [int(num) for num in re.findall(r'\b\d+\b', text)]  # Extracts only valid numbers

            add_frames = extract_numbers(add_match.group(1)) if add_match else []
            remove_frames = extract_numbers(remove_match.group(1)) if remove_match else []

            # If no valid frames were found, return -1
            if not add_frames and not remove_frames:
                return -1

            return {"add": add_frames, "remove": remove_frames}
        
        # Process each response string
        return [process_single_response(resp) for resp in responses_str]
    

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output