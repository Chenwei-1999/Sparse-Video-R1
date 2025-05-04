import re
import random
import math
from typing import Dict, Any, Union, Tuple, Optional, List, Set
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper for parsing frame lists
def parse_frame_list(s: str) -> List[str]:
    return [item.strip() for item in s.split(',') if item.strip()]

# Refactored extract_solution function
def extract_solution(
    solution_str: str,
    current_frames: List[int] = None,
    max_frames: int = None,
    total_times: int = None,
    only_answer: bool = False
) -> Tuple[str, Union[str, dict]]:
    def _validate_answer(text: str) -> Tuple[str, str]:
        txt = text.strip()
        if not re.fullmatch(r"\d+", txt):
            return 'format_error', 'Answer must be a single number.'
        return 'valid_answer', txt

    def _validate_frame_ops(frame_text: str) -> Tuple[str, Union[dict, str]]:
        add_match = re.search(r"\+\[\s*(.*?)\s*\]", frame_text)
        remove_match = re.search(r"-\[\s*(.*?)\s*\]", frame_text)
        if not (add_match or remove_match):
            return 'format_error', 'Missing both add and remove operations.'

        added = parse_frame_list(add_match.group(1)) if add_match else []
        removed = parse_frame_list(remove_match.group(1)) if remove_match else []

        if any(not item.isdigit() for item in added + removed):
            return 'format_error', 'Frame indices must be digits.'
        added = [int(x) for x in added]
        removed = [int(x) for x in removed]

        if current_frames is not None and any(f not in current_frames for f in removed):
            return 'frame_error', 'Attempt to remove non-existent frames.'
        if total_times is not None and any(f < 0 or f > total_times for f in added):
            return 'frame_error', 'Attempt to add non-existent frames.'
        if current_frames is not None and any(f in current_frames for f in added):
            return 'frame_error', 'Duplicate frame addition.'

        new_count = (len(current_frames or []) - len(removed) + len(added))
        if max_frames is not None and new_count > max_frames:
            return 'frame_error', 'Exceeds maximum allowed frames.'
        if new_count < 1:
            return 'frame_error', 'Not enough frames left after operations.'

        return 'valid_frame_ops', {'add': added, 'remove': removed}

    think_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    frame_match = re.search(r"<frames>(.*?)</frames>", solution_str, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)

    if only_answer:
        if not answer_match:
            return 'format_error', 'Missing incomplete <answer> tag.'
        return _validate_answer(answer_match.group(1))

    errors = []
    if not think_match:
        errors.append('Missing <think> reasoning or incomplete <think> tag.')

    if answer_match:
        return _validate_answer(answer_match.group(1))

    if not frame_match:
        errors.append('Neither <frames> nor <answer> provided, or incomplete <frames> tag.')
        return 'format_error', '\n'.join(errors)

    return _validate_frame_ops(frame_match.group(1))


def discretize_time_intervals(intervals: List[List[float]]) -> Set[float]:
    """
    Convert time intervals into a set of discrete timestamps.
    
    Args:
        intervals: List of [start, end] time intervals
        step: Discretization step size in seconds
        
    Returns:
        Set of discrete timestamps
    """
    timestamps = set()
    for start, end in intervals:
        start = int(start)
        end = int(end)
        discrete_timestamps = np.arange(start, end + 1, 1)
        timestamps.update(discrete_timestamps)
    return timestamps

def convert_timestamps_to_set(timestamps: Union[List[float], np.ndarray]) -> Set[float]:
    """Convert timestamps array to set of floats."""
    return {float(t) for t in timestamps}

def calculate_jaccard_similarity(set1: Set[float], set2: Set[float]) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:  # Both empty
        return 0.0
    if not set1 or not set2:  # One is empty
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    status: str = 'running'
) -> float:
    """
    Compute reward score based on Jaccard similarity between selected frames and ground truth.
    
    Args:
        data_source: Source of the data
        solution_str: Model's solution string
        ground_truth: Ground truth frame timestamps
        extra_info: Additional information dictionary containing current frames
        
    Returns:
        float: Jaccard similarity score between 0 and 1
    """
    current_times = extra_info['times']
    ground_truth_times = extra_info['times_GT']
    current_times = convert_timestamps_to_set(current_times)
    ground_truth_times = convert_timestamps_to_set(ground_truth_times)
    jaccard_score = calculate_jaccard_similarity(current_times, ground_truth_times)
    

    status, answer = extract_solution(solution_str, only_answer=True)
    correct_score = 0.0
    if status == 'valid_answer':
        correct_score = 1.0 if answer == ground_truth else 0.0
    else:
        return 0.0
    
    final_score = 0.5 * jaccard_score + 0.5 * correct_score
    # Add debug logging occasionally
    if random.randint(1, 64) == 1:
        print("--------------------------------")
        print(f"Current Frames: {sorted(list(current_times))}")
        print(f"Ground Truth Frames: {sorted(list(ground_truth_times))}")
        print(f"Jaccard Score: {jaccard_score:.4f}")
        print(f"Correct Score: {correct_score:.4f}")
        print(f"Final Score: {final_score:.4f}")
        print(f"Past Times: {extra_info['past_times']}")
        print(f"Solution String: {solution_str}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Status: {status}")
        print(f"Answer: {answer}")
        print("--------------------------------")

    return final_score