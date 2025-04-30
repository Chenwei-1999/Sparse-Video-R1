import re
import random
import math
from typing import Dict, Any, Union, Tuple, Optional, List, Set
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_frame_list(frame_str: str) -> List[int]:
    """
    Parse a comma-separated string of frame numbers into a list of integers.
    
    Args:
        frame_str (str): Comma-separated string of frame numbers
        
    Returns:
        List[int]: List of parsed frame numbers, empty list if invalid
    """
    if not frame_str:
        return []
    try:
        return [int(f.strip()) for f in frame_str.split(',') if f.strip().isdigit()]
    except ValueError:
        logger.warning(f"Failed to parse frame list: {frame_str}")
        return []

def extract_solution(solution_str: str, total_frames: int, current_frames: List[int], simplified: bool = False) -> Tuple[str, Union[int, Dict[str, list]]]:
    """
    Enhanced error detection for multiple-choice responses with frame validation
    """
    # Tag validation
    think_match = re.search(r'<think>(.*?)</think>', solution_str, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    
    # Error type matrix
    if not think_match and not answer_match:
        return ('format_error', 'Missing both <think> and <answer> tags', solution_str)
    if not think_match:
        return ('think_error', 'Missing <think> reasoning section', solution_str)
    if not answer_match:
        return ('answer_error', 'Missing <answer> tag', solution_str)
        
    answer_content = answer_match.group(1).strip()
    
    # Multiple-choice validation
    if not re.match(r'^\d+$', answer_content):
        return ('format_error', 'Answer must be a single number (0-4)', solution_str)
    if answer_content not in {'0', '1', '2', '3', '4'}:
        return ('invalid_answer', f'Invalid choice {answer_content} - must be 0-3', solution_str)

    # Frame modification validation (original requirements)
    add_match = re.search(r'\+\[(.*?)\]', answer_content)
    remove_match = re.search(r'-\[(.*?)\]', answer_content)
    
    # Frame error cases
    if add_match or remove_match:
        added = parse_frame_list(add_match.group(1)) if add_match else []
        removed = parse_frame_list(remove_match.group(1)) if remove_match else []
        
        # Validate frame operations
        if any(f not in current_frames for f in removed):
            return ('frame_error', 'Trying to remove non-existent frames', solution_str)
        if len(current_frames) - len(removed) + len(added) > total_frames:
            return ('frame_error', 'Exceeds maximum allowed frames', solution_str)
        if any(f in current_frames for f in added):
            return ('frame_error', 'Adding duplicate frames', solution_str)

    return ('valid', answer_content, solution_str)

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
        return 1.0
    if not set1 or not set2:  # One is empty
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    # --- Add Tuning Parameters ---
) -> float:
    
    
    if extra_info is None:
        extra_info = {}
    
    required_fields = ['timestamps', 'max_frames', 'current_turn', 'max_turns', 'time', 'type']
    missing_fields = [field for field in required_fields if field not in extra_info]
    if missing_fields:
        # Consider returning a specific penalty or raising a more specific error
        raise ValueError(f"Missing required fields in extra_info: {missing_fields}")

    timestamps = extra_info['timestamps']
    time_intervals = extra_info.get('time', []) # Default to empty list if 'time' is missing

  
    if not isinstance(time_intervals, list) or not all(isinstance(i, list) and len(i) == 2 for i in time_intervals):
         logger.warning(f"Invalid time_intervals format: {time_intervals}. Assuming no relevant intervals.")
         time_intervals = []

    interval_timestamps = discretize_time_intervals(time_intervals)
    reference_timestamps = convert_timestamps_to_set(timestamps)
    similarity_score = calculate_jaccard_similarity(interval_timestamps, reference_timestamps)


    # ... (previous code: validate extra_info, calculate similarity_score) ...


    extraction_type, extracted_value, extracted_solution = extract_solution(
        solution_str,
        total_frames=extra_info['max_frames'],
        current_frames=extra_info['timestamps'],
        simplified=False
    )
    ground_truth_str = str(ground_truth).strip()

    if extraction_type == 'valid':
        answer_str = str(extracted_value).strip()
        is_correct = (answer_str.lower() == ground_truth_str.lower())
        if is_correct:
            final_score = similarity_score*0.5 + 1*0.5
        else:
            final_score = similarity_score
        # Optional: Add a small base reward if the format is correct, even if the answer is wrong?
        # score = max(score, -0.8) # Ensure score doesn't drop too low just for being wrong
    else: 
        final_score = 0.0
    # elif extraction_type == 'modify':
    #     score = (1.0 - similarity_score) * modification_bonus_factor - similarity_score * modification_penalty_factor

    # elif extraction_type in ['mistake', 'think error', 'answer error', 'both error']:
    #     # Penalize formatting/logic errors
    #     score = error_penalty 
    # else: # Should not happen, but catch all
    #     logger.error(f"Unknown extraction type: {extraction_type}. Assigning error penalty.")
    #     score = error_penalty

    # # Clamp score to a [-1.0, 1.0] range (or other desired range)
    # final_score = max(-1.0, min(score, 1.0)) 
    # Debug printing (consider making this conditional or removing in production)
    if random.randint(1, 64) == 1: # Keep infrequent debug prints
         print("--------------------------------")
         print(f"Ground_truth: {ground_truth_str}")
         print(f"Extracted Type: {extraction_type}")
         print(f"Extracted Value: {extracted_value}")
         print(f"Extracted Solution: {extracted_solution}")
         print(f"Sampled timestamps: {reference_timestamps}")
         print(f"GT intervals: {interval_timestamps}")
         print(f"Jaccard Similarity: {similarity_score:.4f}")
         print(f"Final Clamped Score: {final_score:.4f}")
         print("--------------------------------")

    return final_score