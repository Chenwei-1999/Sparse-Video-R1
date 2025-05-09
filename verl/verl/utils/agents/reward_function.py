import re
from typing import List, Tuple, Union, Set, Optional, Dict, Any
import random
import numpy as np


def extract_solution(
    solution_str: str,
    current_frames: List[int] = None,
    max_frames: int = None,
    total_times: int = None,
    only_answer: bool = False, 
    step: int = None
) -> Tuple[str, Union[str, List[int]]]:
    
    def _validate_answer(text: str) -> Tuple[str, str]:
        txt = text.strip()
        if not re.fullmatch(r"\d+", txt):
            return 'format_error', 'Answer must be a single number.'
        return 'valid_answer', txt

    def _validate_frame_selection(frame_text: str) -> Tuple[str, Union[List[int], str]]:
        match = re.fullmatch(r"\s*(\d+(?:\s*,\s*\d+)*)\s*", frame_text)
        if not match:
            return 'format_error', 'Frames must be a comma-separated list of numbers.'

        try:
            frames = [int(x.strip()) for x in match.group(1).split(',')]
        except ValueError:
            return 'format_error', 'All frame indices must be integers.'

        if any(f < 0 for f in frames):
            return 'frame_error', 'Frame indices must be non-negative.'

        if current_frames is not None and frames == current_frames:
            return 'frame_error', 'Selected frames are identical to current frames.'

        if max_frames is not None and len(frames) > max_frames:
            return 'frame_error', f'Number of selected frames ({len(frames)}) exceeds max_frames ({max_frames}).'

        if total_times is not None and any(f >= total_times for f in frames):
            return 'frame_error', f'Some frames exceed total frame count ({total_times}).'

        return 'valid_frames', frames

    # === Extract think ===
    # think_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    # if not think_match:
    #     return 'format_error', 'Missing or incomplete <think> tag.'
  
    # === Extract all matches with their positions ===
    tag_pattern = re.compile(r"<(?P<tag>frames?|answer)>(.*?)</(?P=tag)>", re.DOTALL)
    matches = list(tag_pattern.finditer(solution_str))

    if only_answer:
        # Select first answer only
        for match in matches:
            if match.group("tag") == "answer":
                return _validate_answer(match.group(2))
        return 'format_error', 'Missing <answer> tag.'

    if not matches:
        return 'format_error', 'Missing <frames> or <answer> tag.'

    # Select the first occurring tag (frames or answer)
    first_tag = matches[0].group("tag")
    first_content = matches[0].group(2)

    if first_tag.startswith("frame"):
        return _validate_frame_selection(first_content)
    elif first_tag == "answer":
        return _validate_answer(first_content)
    else:
        return 'format_error', 'Unknown tag type encountered.'


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