##################################################################
# verl/utils/agents/generation.py                                   #
#                                                                    #
# Optimized LLMGenerationManager:                                   #
# - Merged `correction` & `execute_predictions` into core loop.    #
# - Reduced redundant loops; combined state updates.                #
# - Uses list comprehensions for collecting responses & ops.       #
# - Single pass over active indices per round for clarity & speed.  #
##################################################################

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig, ListConfig
from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.tracking import Tracking
from verl.utils.model import compute_position_id_with_mask
from verl.utils.agents.construct_prompt import generate_prompt
from verl.utils.agents.frames_sampler import sample_frames_from_next_obs
from verl.utils.agents.reward_function import extract_solution
from verl.utils.dataset.vision_utils import process_image
import verl.utils.torch_functional as verl_F
from verl.utils.agents.tensor_helper import TensorConfig, TensorHelper


@dataclass
class GenerationConfig:
    max_rounds: int
    max_frames: int
    max_prompt_length: int
    max_response_length: int
    num_gpus: int
    no_think_rl: bool = False


class LLMGenerationManager:
    """
    A loop‑based generator for multi‑round video‑QA and frame selection.
    """

    def __init__(
        self,
        tokenizer,
        processor,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        logger: Optional[Tracking] = None,
        ratio: float = 1.0,
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

        self.tensor_fn = TensorHelper(
            TensorConfig(
                pad_token_id=tokenizer.pad_token_id,
                max_prompt_length=config.max_prompt_length,
            )
        )
        self.max_prompt_length = config.max_prompt_length
        self.max_response_length = config.max_response_length
        self.truncation = "error"  # never silently truncate

        # Bookkeeping per run
        self.conversation_history: List[str] = []
        self.image_history: List[List[Any]] = []

    def run_llm_loop(
        self,
        gen_batch: DataProto,
        global_steps: Optional[int] = None,
    ) -> DataProto:
        """
        Main loop: runs up to `max_rounds` of generation + frame selection.
        """
        rollings = copy.deepcopy(gen_batch)
        batch_size = rollings.batch["input_ids"].shape[0]
        dones = [False] * batch_size
        final_outputs: List[Optional[DataProto]] = [None] * batch_size

        # Initialize conversation & image history
        self.conversation_history = [None] * batch_size
        self.image_history = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            self.conversation_history[i] = [
                {"role": "user", "content": [
                    {"type": "image", "image": rollings.non_tensor_batch["multi_modal_data"][i]["image"]},
                    {"type": "text", "text": rollings.non_tensor_batch["extra_info"][i]["prompt"]}
                ]}
            ]
            self.image_history[i] = rollings.non_tensor_batch["multi_modal_data"][i]["image"]

        for step in range(1, self.max_rounds + 1):
            active = [i for i, d in enumerate(dones) if not d]
            if not active:
                break

            sub_batch = self._create_sub_batch(rollings, active)
            gen_out = self._generate_with_gpu_padding(sub_batch)

            # Decode prompts & responses
            raw_ids = sub_batch.non_tensor_batch["raw_prompt_ids"]
            raw_prompts = self.tokenizer.batch_decode(raw_ids)
            responses = self.tokenizer.batch_decode(
                gen_out.batch["responses"], skip_special_tokens=True
            )

            # Prepare frame/time info
            current_times = [rollings.non_tensor_batch["extra_info"][i]["times"] for i in active]
            total_times   = [rollings.non_tensor_batch["extra_info"][i]["total_times"] for i in active]

            # Single pass: parse solutions, decide next_times & done flags
            next_times = []
            next_dones = []
            parsed_ops = []  # None=answer, ("replace", frames), ("error", msg)
            for resp, times, tot in zip(responses, current_times, total_times):
                etype, result = extract_solution(
                    solution_str=resp,
                    current_frames=times,
                    max_frames=self.max_frames,
                    total_times=tot,
                )

                if etype == "valid_answer":
                    parsed_ops.append(None)
                    next_times.append(None);
                    next_dones.append(True)
                elif etype == "valid_frames":
                    parsed_ops.append(("replace", result))
                    next_times.append(result);
                    next_dones.append(False)
                else:
                    parsed_ops.append(("error", f"{etype}: {result}"))
                    next_times.append(None);
                    next_dones.append(False)
            if random.randint(0, 64) == 1:
                print(f"response: {responses}")
                print(f"parsed_ops: {parsed_ops}")
                print(f"next_times: {next_times}")
                print(f"next_dones: {next_dones}")

            # Mark completed samples
            for idx, orig in enumerate(active):
                response_content = {
                    'role': 'assistant',
                    'content': responses[idx]
                }
                self.conversation_history[orig].append(response_content)

                if next_dones[idx] or step == self.max_rounds:
                    dones[orig] = True
                    final_outputs[orig] = gen_out[idx: idx+1]

            if all(dones):
                break

            # Sample frames for next round (skip errors/dones)
            sampled = [
                [] if next_dones[idx] or parsed_ops[idx][0] == "error"
                else sample_frames_from_next_obs(
                    rollings.non_tensor_batch["extra_info"][orig]["video_path"],
                    next_times[idx],
                    rollings.non_tensor_batch["extra_info"][orig]["height"],
                    rollings.non_tensor_batch["extra_info"][orig]["width"],
                    ratio=self.ratio,
                )
                for idx, orig in enumerate(active)
            ]

            # Update state & prompts
            rollings = self.update_rollings_state(
                rollings=rollings,
                active_indices=active,
                next_times=next_times,
                parsed_ops=parsed_ops,
                sampled_frames=sampled,
                new_round=step+1,
            )
        if random.randint(0, 64) == 1:
            print(f"run {step} rounds")
            for i in range(batch_size):
                print("="*20)
                print(self.conversation_history[i])
                print(rollings.non_tensor_batch["extra_info"][i])
                print("="*20)
         # Final assembly: ensure each DataProto carries its `extra_info`

        final_outputs= DataProto.concat(final_outputs)
        final_outputs.non_tensor_batch["extra_info"] = rollings.non_tensor_batch["extra_info"]
        return final_outputs

    def update_rollings_state(
        self,
        rollings: DataProto,
        active_indices: List[int],
        next_times: List[List[int]],
        parsed_ops: List[Optional[Tuple[str, Any]]],
        sampled_frames: List[List[Dict]],
        new_round: int,
    ) -> DataProto:
        """
        Updates `rollings` in-place for the next generation round.
        """
        for idx, orig in enumerate(active_indices):
            ops = parsed_ops[idx]
            # If error, build corrective prompt
            if isinstance(ops, tuple) and ops[0] == "error":
                prompt = f"Error ({ops[1]}). Please try again."
            # If answer, skip further prompting
            elif ops is None:
                # preserve times, no new prompt
                continue
            # Valid frame replacement
            else:
                prompt = generate_prompt(
                    question=rollings.non_tensor_batch["extra_info"][orig]["question"],
                    timestamps=next_times[idx],
                    total_times=rollings.non_tensor_batch["extra_info"][orig]["total_times"],
                    n_round=new_round,
                    max_rounds=self.max_rounds,
                    max_frames=self.max_frames,
                )
                rollings.non_tensor_batch["extra_info"][orig]["times"] = next_times[idx]
                rollings.non_tensor_batch["extra_info"][orig]["past_times"][new_round-1] = next_times[idx]
            rollings.non_tensor_batch["extra_info"][orig]["current_round"] = new_round-1

            # Add any new images for next prompt
            images = [process_image({"image": fr["image"]}) for fr in sampled_frames[idx]]
            self.image_history[orig].extend(images)

            # Build raw prompt with images + text
            user_content = [{"type":"image","image":im} for im in images] + [{"type":"text","text":prompt}]
            messages = [{"role":"user","content":user_content}]
            self.conversation_history[orig].append({"role": "user", "content": user_content})
            raw = self.tokenizer.apply_chat_template(
                self.conversation_history[orig], add_generation_prompt=True, tokenize=False
            ) + "<think>"

            # Re-tokenize for next round
            mi = self.processor(
                text=[raw],
                images=self.image_history[orig],
                return_tensors="pt",
            )
            input_ids = mi.pop("input_ids")
            attn_mask = mi.pop("attention_mask")
            ids, mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
            rollings.batch["input_ids"][orig] = ids[0]
            rollings.batch["attention_mask"][orig] = mask[0]
            rollings.non_tensor_batch["raw_prompt_ids"][orig] = self.tokenizer.encode(raw, add_special_tokens=False)

            # Re-attach images + position_ids
            rollings.non_tensor_batch["multi_modal_data"][orig] = {"image": self.image_history[orig]}
            if self.processor and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
                from verl.models.transformers.qwen2_vl import get_rope_index
                pos = get_rope_index(
                    self.processor,
                    input_ids=ids[0],
                    image_grid_thw=mi.get("image_grid_thw"),
                    video_grid_thw=mi.get("video_grid_thw"),
                    second_per_grid_ts=mi.get("second_per_grid_ts"),
                    attention_mask=mask[0],
                )
                rollings.batch["position_ids"][orig] = pos
            else:
                rollings.batch["position_ids"][orig] = compute_position_id_with_mask(mask[0])

        return rollings

    @staticmethod
    def _create_sub_batch(full_batch: DataProto, indices: List[int]) -> DataProto:
        tensors = {k: v[indices] for k, v in full_batch.batch.items()}
        non_tensors = {k: full_batch.non_tensor_batch[k][indices] for k in full_batch.non_tensor_batch}
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    def _generate_with_gpu_padding(self, gen_batch: DataProto) -> DataProto:
        padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
        out = self.actor_rollout_wg.generate_sequences(padded)
        return unpad_dataproto(out, pad_size=pad_size)
