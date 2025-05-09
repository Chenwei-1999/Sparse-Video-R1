# ------------------------------------------------------------
#  verl/utils/agents/generation.py   (FULL FILE)
# ------------------------------------------------------------
import copy
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


import torch
from omegaconf import DictConfig, ListConfig


from verl.utils.tracking import Tracking
from verl.utils.model import compute_position_id_with_mask
from verl.utils.agents.frames_sampler import (
   sample_frames_from_next_obs,
   sample_video_frames,
)
from verl.utils.agents.construct_prompt import generate_prompt, generate_prompt_for_force_round
from verl.utils.agents.reward_function import extract_solution
from verl.utils.agents.tensor_helper import TensorConfig, TensorHelper
from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.vision_utils import process_image, process_video
import verl.utils.torch_functional as verl_F




# -----------------------------------------------------------------------------




@dataclass
class GenerationConfig:
   max_rounds: int
   max_frames: int
   max_prompt_length: int
   max_response_length: int
   num_gpus: int
   no_think_rl: bool = False




# -----------------------------------------------------------------------------




class LLMGenerationManager:
   """
   A loop‑based generator that supports multi‑round interaction with an LLM
   for video‑QA / frame‑selection tasks.
   """


   # ---------------------------------------------------------------------


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
       # we *never* silently truncate – raise if too long
       self.truncation = "error"


       # conversation bookkeeping (re‑initialized every run)
       self.conversation_history: Optional[List[List[Dict[str, Any]]]] = None
       self.completion_rounds: Dict[int, int] = {}


   # ---------------------------------------------------------------------
   #  MAIN ENTRY
   # ---------------------------------------------------------------------


   def run_llm_loop(
       self, gen_batch: DataProto, global_steps: Optional[int] = None
   ) -> DataProto:
       """
       Run up to `max_rounds` of LLM ↔ environment interaction.


       Returns
       -------
       DataProto
           A batch containing responses **and** a faithful copy of
           `extra_info` for every sample so that downstream reward
           functions can safely index it.
       """
       rollings = copy.deepcopy(gen_batch)
       batch_size = rollings.batch["input_ids"].shape[0]


       dones = [False] * batch_size
       final_output_batch: List[Optional[DataProto]] = [None] * batch_size


       # initialise per‑sample chat history
       self.conversation_history = [[] for _ in range(batch_size)]
       self.completion_rounds = {i: self.max_rounds for i in range(batch_size)}


       # -----------------------------------------------------------------
       # seed the history with the *initial* prompt + any initial images
       # -----------------------------------------------------------------
       initial_mm = rollings.non_tensor_batch.get(
           "multi_modal_data", [None] * batch_size
       )


       for i in range(batch_size):
           mm_item = initial_mm[i]
           if mm_item and isinstance(mm_item, dict) and "image" in mm_item:
               for img in mm_item["image"]:
                   self.conversation_history[i].append({"type": "image", "image": img})


           prompt_text = rollings.non_tensor_batch["extra_info"][i]["prompt"]
           self.conversation_history[i].append({"type": "text", "text": prompt_text})


       # ------------------------- LOOP ----------------------------------
       for step in range(1, self.max_rounds + 1):
           active_indices = [i for i, done in enumerate(dones) if not done]
           if not active_indices:
               break  # all done


           # --------------------------------------------------------------
           # 1) prepare sub‑batch and generate
           # --------------------------------------------------------------
           active_batch = self._create_sub_batch(rollings, active_indices)
           gen_output = self._generate_with_gpu_padding(active_batch, step)


           responses_str = self.tokenizer.batch_decode(
               gen_output.batch["responses"], skip_special_tokens=True
           )


           # --------------------------------------------------------------
           # 2) error / correction checking
           # --------------------------------------------------------------
           current_times = [
               rollings.non_tensor_batch["extra_info"][i]["times"] for i in active_indices
           ]
           total_times_list = [
               rollings.non_tensor_batch["extra_info"][i]["total_times"]
               for i in active_indices
           ]


           correction_info = self.correction(
               responses_str, current_times, total_times_list, step
           )


           # --------------------------------------------------------------
           # 3) environment step (simulate frame add/remove)
           # --------------------------------------------------------------
           next_obs, step_dones = self.execute_predictions(
               current_times, correction_info
           )


           # --------------------------------------------------------------
           # 4) bookkeeping & store DataProto for finished samples
           # --------------------------------------------------------------
           for idx, orig_idx in enumerate(active_indices):
               # keep chat transcript
               self.conversation_history[orig_idx].append(
                   {"type": "text", "text": responses_str[idx]}
               )


               # update `past_times`:
               if not correction_info[idx]["needs_correction"] and not step_dones[idx]:
                   rollings.non_tensor_batch["extra_info"][orig_idx]["past_times"][
                       step - 1
                   ] = next_obs[idx]




               # mark completion
               if step_dones[idx] or step == self.max_rounds:
                   dones[orig_idx] = True
                   self.completion_rounds[orig_idx] = step
                   final_output_batch[orig_idx] = gen_output[idx : idx + 1]


           # --------------------------------------------------------------
           # 5) early exit?
           # --------------------------------------------------------------
           if all(dones) or step == self.max_rounds:
               # Fill any *still‑None* slots with the last gen_output slice
               for idx, orig_idx in enumerate(active_indices):
                   if final_output_batch[orig_idx] is None:
                       final_output_batch[orig_idx] = gen_output[idx : idx + 1]
               break


           # --------------------------------------------------------------
           # 6) sample new frames for the *next* round (where allowed)
           # --------------------------------------------------------------
           sampled_frames_batch: List[List[Dict]] = []
           for idx, orig_idx in enumerate(active_indices):
               if step_dones[idx] or correction_info[idx]["needs_correction"]:
                   sampled_frames_batch.append([])
               else:
                   sampled_frames_batch.append(
                       sample_frames_from_next_obs(
                           rollings.non_tensor_batch["extra_info"][orig_idx][
                               "video_path"
                           ],
                           next_obs[idx],
                           rollings.non_tensor_batch["extra_info"][orig_idx]["height"],
                           rollings.non_tensor_batch["extra_info"][orig_idx]["width"],
                           ratio=self.ratio,
                       )
                   )


           # --------------------------------------------------------------
           # 7) update the rolling batch in‑place ready for next turn
           # NOTE: new_round is *step+1* (next round index) – this number
           # is only used when rebuilding the *prompt* template.  We do
           # **not** rely on it for internal bookkeeping any more.
           # --------------------------------------------------------------
           rollings = self.update_rollings_state(
               rollings,
               active_indices=active_indices,
               sampled_frames_batch=sampled_frames_batch,
               new_times=next_obs,
               new_round=step + 1,  # prompt construction only
               responses_str=responses_str,
               correction_info=correction_info,
           )


       # -----------------------------------------------------------------
       # 8) Final assembly – make sure each DataProto carries `extra_info`
       # -----------------------------------------------------------------
       for i, dp in enumerate(final_output_batch):
           if dp is None:
               raise RuntimeError("internal error: missing DataProto slice")
           dp.non_tensor_batch["extra_info"] = rollings.non_tensor_batch[
               "extra_info"
           ][i : i + 1]     # ← slice, returns shape (1,) object array
       if random.random() < 0.3:
           print(self.conversation_history[0])
           responses = final_output_batch[0].batch["responses"][0]
           responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
           print(responses)
       return DataProto.concat(final_output_batch)


   # ---------------------------------------------------------------------
   #  Helper methods
   # ---------------------------------------------------------------------


   @staticmethod
   def _create_sub_batch(full_batch: DataProto, indices: List[int]) -> DataProto:
       new_tensors = {k: v[indices] for k, v in full_batch.batch.items()}
       new_non_tensors = {
           k: full_batch.non_tensor_batch[k][indices] for k in full_batch.non_tensor_batch
       }
       return DataProto.from_dict(tensors=new_tensors, non_tensors=new_non_tensors)


   # ---------------------------------------------------------------------


   def update_rollings_state(
       self,
       rollings: DataProto,
       active_indices: List[int],
       sampled_frames_batch: List[List[Dict]],
       new_times: List[List[int]],
       new_round: int,
       responses_str: Optional[List[str]] = None,
       correction_info: Optional[List[Dict]] = None,
   ) -> DataProto:
       """
       Mutates `rollings` in‑place for the *next* generation round.
       """


       for idx, orig_idx in enumerate(active_indices):
           # ----------------------------------------------------------
           # (a) frame / time history – change *only* if we really
           #     sampled fresh frames this turn
           # ----------------------------------------------------------
           if sampled_frames_batch[idx]:
               rollings.non_tensor_batch["extra_info"][orig_idx][
                   "frames"
               ] = sampled_frames_batch[idx]
               rollings.non_tensor_batch["extra_info"][orig_idx][
                   "times"
               ] = new_times[idx]


           # ----------------------------------------------------------
           # (b) advance round counter
           # ----------------------------------------------------------
           rollings.non_tensor_batch["extra_info"][orig_idx]["current_round"] = (
               new_round
           )


           # ----------------------------------------------------------
           # (c) build next‑round prompt
           # ----------------------------------------------------------
           if correction_info[idx]["needs_correction"]:
               prompt = correction_info[idx]["message"]
           else:
               prompt = generate_prompt_for_force_round(
                   question=rollings.non_tensor_batch["extra_info"][orig_idx][
                       "question"
                   ],
                   timestamps=new_times[idx] if new_times[idx] else [],
                   total_times=rollings.non_tensor_batch["extra_info"][orig_idx].get(
                       "total_times"
                   ),
                   n_round=new_round,
                   max_rounds=self.max_rounds,
                   max_frames=self.max_frames,
               )


           # ----------------------------------------------------------
           # (d) extend conversation history with any *new* images +
           #     the freshly built prompt
           # ----------------------------------------------------------
           current_frames = sampled_frames_batch[idx] if sampled_frames_batch[idx] else []
           current_images = [process_image({"image": fr["image"]}) for fr in current_frames]


           if current_images:
               for image in current_images:
                   self.conversation_history[orig_idx].append(
                       {"type": "image", "image": image}
                   )
           self.conversation_history[orig_idx].append(
               {"type": "text", "text": prompt}
           )


           # ----------------------------------------------------------
           # (e) rebuild raw prompt with the *full* history
           # ----------------------------------------------------------
           messages = [{"role": "user", "content": self.conversation_history[orig_idx]}]
           raw_prompt = self.processor.apply_chat_template(
               messages, add_generation_prompt=True, tokenize=False
           )
           all_images = [
               item["image"]
               for item in self.conversation_history[orig_idx]
               if item.get("type") == "image"
           ]


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


           # ----------------------------------------------------------
           # (f) write the new prompt tensors back into the rolling
           #     batch (in‑place)
           # ----------------------------------------------------------
           rollings.batch["input_ids"][orig_idx] = input_ids[0]
           rollings.batch["attention_mask"][orig_idx] = attention_mask[0]


           rollings.non_tensor_batch["multi_modal_data"][orig_idx] = {"image": all_images}
           rollings.non_tensor_batch["multi_modal_inputs"][orig_idx] = dict(model_inputs)
           rollings.non_tensor_batch["raw_prompt_ids"][orig_idx] = self.tokenizer.encode(
               raw_prompt, add_special_tokens=False
           )


           # ----------------------------------------------------------
           # (g) position_ids (Qwen2‑VL requires rope indices)
           # ----------------------------------------------------------
           if (
               self.processor is not None
               and self.processor.image_processor.__class__.__name__
               == "Qwen2VLImageProcessor"
           ):
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
               position_ids = compute_position_id_with_mask(attention_mask)


           rollings.batch["position_ids"][orig_idx] = position_ids[0]


       return rollings


   # ---------------------------------------------------------------------


   def _generate_with_gpu_padding(self, gen_batch: DataProto, step: int) -> DataProto:
       """
       Wrap the worker‑group `generate_sequences` call with padding /
       un‑padding so the batch is divisible by the DP world size.
       """
       padded_batch, pad_size = pad_dataproto_to_divisor(
           gen_batch, self.actor_rollout_wg.world_size
       )
       output_batch = self.actor_rollout_wg.generate_sequences(padded_batch, step, max_rounds=self.max_rounds)
       return unpad_dataproto(output_batch, pad_size=pad_size)


   # ---------------------------------------------------------------------


   def correction(
       self,
       responses_str: List[str],
       current_frames: List[List[int]],
       total_times: List[int],
       step: int,
   ) -> List[Dict[str, Any]]:
       """
       Validate responses; decide if a correction turn is needed.


       Returns
       -------
       List[Dict] – one per sample, with keys:
           * needs_correction (bool)
           * message          (str | dict)  – prompt for next turn, or details
           * error_type       (str)
       """
       corrections: List[Dict[str, Any]] = []


       for response, frames, total_time in zip(
           responses_str, current_frames, total_times
       ):
           info = {
               "needs_correction": False,
               "message": "",
               "error_type": None,
           }


           error_type, error_message = extract_solution(
               response, frames, self.max_frames, total_time, step, self.max_rounds
           )


           if error_type in ("valid_frames", "valid_answer"):
               # response is acceptable; maybe still return parsed ops
               info.update(needs_correction=False, message=error_message, error_type=error_type)
           elif error_type == "format_error":
               info.update(
                   needs_correction=True,
                   message=f"Invalid response format: {error_message} Please try again.",
                   error_type=error_type,
               )
           elif error_type == "frame_error":
               info.update(
                   needs_correction=True,
                   message=f"Invalid frame selection: {error_message} Please try again.",
                   error_type=error_type,
               )
           else:
               raise ValueError(f"Unexpected error_type {error_type}")


           corrections.append(info)


       return corrections


   # ---------------------------------------------------------------------


   def execute_predictions(
       self,
       current_frames: List[List[int]],
       correction_info: List[Dict[str, Any]],
   ) -> Tuple[List[List[int]], List[bool]]:
       """
       Apply the frame add/remove ops (or final answer) produced by the
       model, returning the *updated* frame list plus a flag indicating
       whether each sample is finished.
       """
       next_obs: List[List[int]] = []
       dones: List[bool] = []


       for frames, corr in zip(current_frames, correction_info):
           etype = corr["error_type"]
           msg = corr["message"]


           if etype == "valid_answer":
               next_obs.append(frames)
               dones.append(True)


           elif etype == "valid_frame_ops":
               # msg is a dict {add: [...], remove: [...]}
               next_obs.append(
                   update_frames(frames, msg["add"], msg["remove"])
               )
               dones.append(False)


           else:
               # format/frame error – no change
               next_obs.append(frames)
               dones.append(False)


       return next_obs, dones




# -------------------------------------------------------------------------
#  Utility – pure function
# -------------------------------------------------------------------------




def update_frames(
   current_frames: List[int], add_frames: List[int], remove_frames: List[int]
) -> List[int]:
   """
   Apply add/remove ops to the current frame list.
   """
   new_frames = [f for f in current_frames if f not in remove_frames]
   for f in add_frames:
       if f not in new_frames:
           new_frames.append(f)
   return new_frames





