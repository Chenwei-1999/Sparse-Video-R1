import random
import re
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from copy import deepcopy

# Suppose you already have these imports somewhere else:
# from .tensor_helper import TensorHelper, TensorConfig
# from verl import DataProto
# from ragen.utils.plot import parse_llm_output
# from your_code import generate_prompt, sample_video_frames, process_image, ...

@dataclass
class VideoSamplingConfig:
    max_turns: int             # max number of "refine" turns
    max_frames: int            # maximum frames allowed
    max_prompt_length: int
    num_gpus: int
    # You can add more fields as needed (logging dict, if you want)

def parse_frame_instructions(answer_text: str) -> Tuple[List[int], List[int]]:
    """
    Parse strings like:
      <answer>+[3,4,10]-[2]</answer>
    to find which frames to add or remove.

    Returns:
        add_list (List[int]): frames to add
        remove_list (List[int]): frames to remove
    """
    # Example pattern for +[...] or -[...]
    pattern_add = r'\+\[(.*?)\]'
    pattern_remove = r'\-\[(.*?)\]'

    add_matches = re.findall(pattern_add, answer_text)
    remove_matches = re.findall(pattern_remove, answer_text)

    # Convert CSV-like strings to integer lists
    def to_int_list(s):
        s = s.strip()
        if not s:
            return []
        return [int(x) for x in s.split(',') if x.strip().isdigit()]

    add_list, remove_list = [], []
    for grp in add_matches:
        add_list.extend(to_int_list(grp))
    for grp in remove_matches:
        remove_list.extend(to_int_list(grp))

    return add_list, remove_list

class VideoSamplingManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,  
        config: VideoSamplingConfig,
        logger=None
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.logger = logger
        # Optional: a TensorHelper if you want to handle complex padding/position_ids
        # self.tensor_helper = TensorHelper(TensorConfig(
        #     pad_token_id=tokenizer.pad_token_id,
        #     max_prompt_length=config.max_prompt_length
        # ))

    def _postprocess_response(self, response_str: str) -> str:
        """
        Extract text inside the first <answer>...</answer> block.
        If no such block, return the entire response_str.
        """
        match = re.search(r'<answer>(.*?)</answer>', response_str, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return response_str.strip()

    def refine_frames_with_model(
        self,
        initial_gen_batch,  # DataProto with input_ids, attention_mask, etc. for the *initial* frames
        batch,              # DataProto or a container with the row_dict data
        max_refine_turns: int = 3
    ):
        """
        Multi-turn loop:
          1) Generate with the current frames.
          2) Parse <answer> for +[...] or -[...].
          3) Add/Remove frames. Rebuild prompt & re-tokenize.
          4) Repeat until no change or max_refine_turns.
          5) Return final DataProto with updated frames/prompt.
        """
        # We'll track the frames in a local list:
        # Your code in __getitem__ might store frames in something like:
        #    batch.non_tensor_batch['multi_modal_data']['image']   -> the actual frame pixel arrays
        #    batch.non_tensor_batch['timestamps'] -> the second indices for those frames
        # We'll assume that we want to add or remove *indices* from a larger set.
        # If you only have these frames, you might need to store the full array of potential frames.

        # For demonstration, let's do the simpler approach:
        # We only manage the currently selected frames, stored in 'multi_modal_data'.
        # If we get +[...] for frames we don't have, we'll skip them (or you can keep a bigger pool).
        # Similarly for removal.

        current_frames = batch.non_tensor_batch['multi_modal_data'].get('image', [])
        current_timestamps = batch.non_tensor_batch.get('timestamps', [])

        for turn_i in range(max_refine_turns):
            # 1) Generate
            refine_output = self.actor_rollout_wg.generate_sequences(initial_gen_batch)
            responses_str = self.tokenizer.batch_decode(
                refine_output.batch['responses'], skip_special_tokens=True
            )
            if not responses_str:
                # No model output => break
                break

            # Let's assume single-batch for simplicity
            answer_text = self._postprocess_response(responses_str[0])

            # 2) Parse instructions
            add_list, remove_list = parse_frame_instructions(answer_text)
            # If no frames to add/remove, we assume the model is satisfied (or gave final answer).
            if not add_list and not remove_list:
                break

            # 3) Add/Remove frames
            # For demonstration, let's interpret add_list or remove_list as timestamps indexes
            # E.g., if the model says +[3], that means "please add the frame near second=3"
            # But you might interpret them as indices into a global pool of frames.
            # We'll show a simple approach: if there's a mismatch, we skip.

            # Convert current_timestamps to a set for easier handling
            ts_set = set(current_timestamps)
            
            # Remove frames if found
            for t in remove_list:
                if t in ts_set:
                    # Find index in current_timestamps
                    idx_to_remove = current_timestamps.index(t)
                    current_timestamps.pop(idx_to_remove)
                    current_frames.pop(idx_to_remove)
                    ts_set.remove(t)


            # Add frames if within some range (0 <= t <= ???)
            for t in add_list:
                # Suppose you can re-sample that frame from disk or from a stored pool
                # We'll do a dummy example: just say we add a placeholder if not present
                if len(current_frames) < self.config.max_frames:
                    if t not in ts_set:
                        # TODO: Actually load the frame from disk or a stored cache
                        new_frame = f"<frame@{t}s>"  # placeholder string or actual image
                        current_frames.append(new_frame)
                        current_timestamps.append(t)
                        ts_set.add(t)


            # Enforce max_frames if we exceed after additions
            if len(current_frames) > self.config.max_frames:
                # For instance, remove oldest frames or remove extras. 
                # Example: keep only the first self.config.max_frames
                # Random keep the current frames
                random_indexes = random.sample(range(len(current_frames)), self.config.max_frames)
                current_frames = [current_frames[i] for i in random_indexes]
                # current_frames = current_frames[: self.config.max_frames]
                # current_timestamps = current_timestamps[: self.config.max_frames]
                # Or you can do more advanced logic.

            # 4) Rebuild the prompt
            new_prompt = self._create_prompt(
                question=batch.non_tensor_batch.get('question', 'Unknown question'),
                timestamps=current_timestamps
            )

            # 5) Re-tokenize: we mimic your __getitem__ approach on the new prompt
            new_gen_batch = self._retokenize_prompt(
                batch=batch,
                frames=current_frames,
                timestamps=current_timestamps,
                prompt_str=new_prompt
            )

            # Update the "initial_gen_batch" so next turn uses updated prompt
            initial_gen_batch = new_gen_batch

        # Return final DataProto after the refinement loop
        return initial_gen_batch

    def _create_prompt(self, question: str, timestamps: List[int]) -> str:
        """
        Similar to your existing generate_prompt function,
        but placed here for clarity. 
        """
        prompt = f"""
        You have a video with {len(timestamps)} frames (decoded at 1 fps).
        The sampled frame timestamps (in seconds) are: {timestamps}
        Please answer the following question:

        {question}

        If the available frames provide enough information, answer directly. Otherwise, 
        specify which frames (in seconds) to add or remove to ensure total does not exceed {self.config.max_frames} frames.

        Formatting Guidelines:
        - To add frames: +[frame1, frame2, ...]
        - To remove frames: -[frame1, frame2, ...]
        - If no changes are needed, simply provide the answer.
        - Use <think>...</think> for reasoning and <answer>...</answer> for the final response.

        <think>
        """
        return prompt

    def _retokenize_prompt(
        self,
        batch,
        frames: List[Any],
        timestamps: List[int],
        prompt_str: str
    ):
        """
        Rebuild a DataProto with new input_ids, attention_mask, position_ids, etc.
        after updating frames/timestamps and the textual prompt.
        Mimics your __getitem__ tokenization logic.
        """
        # 1) Possibly store frames & timestamps in the batch's non_tensor_batch
        batch.non_tensor_batch['multi_modal_data']['image'] = frames
        batch.non_tensor_batch['timestamps'] = timestamps

        # 2) If you have a specialized approach for inserting <image> tokens, do it here.
        # For example, if frames is non-empty, you might replace each with <image>.
        # We'll do a simplified approach:
        vision_placeholder = "<image>"
        num_images = len(frames)
        prompt_with_images = prompt_str
        if num_images > 0:
            # For demonstration: replace each frame with a single <image>, 
            # or you could do f"{vision_placeholder}...{vision_placeholder}" repeated
            prompt_with_images = prompt_with_images.replace(
                "<think>",
                f"{vision_placeholder * num_images}<think>"
            )

        # 3) Now tokenize
        token_out = self.tokenizer(
            prompt_with_images,
            add_special_tokens=False,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.config.max_prompt_length
        )
        input_ids = token_out['input_ids']
        attention_mask = token_out['attention_mask']

        # 4) Create position_ids (simple version)
        position_ids = torch.arange(0, input_ids.shape[1]).unsqueeze(0)
        # Or if your Qwen-VL approach needs special indexing, call your existing function:
        # position_ids = compute_position_id_with_mask(attention_mask[0])  # e.g.

        # 5) Build a new DataProto (or update the existing one)
        from verl import DataProto
        new_gen_batch = DataProto.from_dict({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        })
        new_gen_batch.meta_info = deepcopy(batch.meta_info)
        new_gen_batch.non_tensor_batch = deepcopy(batch.non_tensor_batch)

        return new_gen_batch
