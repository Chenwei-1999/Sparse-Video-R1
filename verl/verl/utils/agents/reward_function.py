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

def extract_solution(solution_str: str, simplified: bool = False) -> Tuple[str, Union[int, Dict[str, list]]]:
    """
    Extract the solution type and value from the solution string.
    
    The solution string should contain:
    - <think> tags for reasoning
    - <answer> tags for the actual answer
    - Optional frame modifications in format +[frames] or -[frames]

    Or if it is simplified, it should contain:
    - <answer> tags for the actual answer
    
    Args:
        solution_str (str): The solution text from the model
        
    Returns:
        Tuple[str, Union[int, Dict[str, list]]]: A tuple containing:
            - The type of solution ('answer', 'modification', or error type)
            - The extracted value (answer, modification dict, or error message)
    """
    answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    if simplified:
        if not answer_match:
            return ('answer error', 'No <answer> tag found')
        return ('answer', answer_match.group(1).strip())
    
    think_match = re.search(r'<think>(.*?)</think>', solution_str, re.DOTALL)

    if not answer_match and not think_match:
        return ('both error', 'No <think> or <answer> tag found')
    
    if not think_match:
        return ('think error', 'No <think> tag found')
    
    if not answer_match:
        return ('answer error', 'No <answer> tag found')

    content = answer_match.group(1).strip()

    # Check for frame modification patterns
    add_match = re.search(r'\+\s*\[(.*?)\]', content)
    remove_match = re.search(r'-\s*\[(.*?)\]', content)

    # Determine if this is a modification request
    is_modification_request = (
        (add_match and remove_match) or
        (add_match and content.startswith('+[')) or
        (remove_match and content.startswith('-[')) or
        (add_match and not remove_match and '+' in content and '[' in content) or
        (remove_match and not add_match and '-' in content and '[' in content)
    )

    if is_modification_request:
        added_frames = parse_frame_list(add_match.group(1)) if add_match else []
        removed_frames = parse_frame_list(remove_match.group(1)) if remove_match else []

        # Validate frame lists
        if add_match and not added_frames and add_match.group(1).strip():
            return ('mistake', 'Invalid numbers in add list')
        if remove_match and not removed_frames and remove_match.group(1).strip():
            return ('mistake', 'Invalid numbers in remove list')

        # If only symbols without valid frames, treat as answer
        if not add_match and not remove_match:
            return ('answer', content)

        return ('modify', {'add': added_frames, 'remove': removed_frames})
    else:
        return ('answer', content)

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
    simplified: bool = False, 
    # --- Add Tuning Parameters ---
    correct_answer_reward_factor: float = 1.0,  # Scales reward for correct answer based on similarity
    incorrect_answer_base_penalty: float = 1.0, # Penalty subtracted when answer is incorrect
    modification_bonus_factor: float = 0.5,    # Reward multiplier for modifying when similarity is low
    modification_penalty_factor: float = 0.5,  # Penalty multiplier for modifying when similarity is high
    error_penalty: float = -1.0                # Fixed penalty for errors
) -> float:
    """
    Compute the score based on the solution string, ground truth, and rules.
    ... (rest of the docstring) ...
    """
    # ... (previous code: validate extra_info, calculate similarity_score) ...

    if extra_info is None:
        extra_info = {}
    
    required_fields = ['timestamps', 'max_frames', 'current_turn', 'max_turns', 'time']
    missing_fields = [field for field in required_fields if field not in extra_info]
    if missing_fields:
        # Consider returning a specific penalty or raising a more specific error
        logger.error(f"Missing required fields in extra_info: {missing_fields}")
        return error_penalty # Or raise ValueError as before

    timestamps = extra_info['timestamps']
    time_intervals = extra_info.get('time', []) # Default to empty list if 'time' is missing

    # Handle cases where time_intervals might be empty or invalid
    if not isinstance(time_intervals, list) or not all(isinstance(i, list) and len(i) == 2 for i in time_intervals):
         logger.warning(f"Invalid time_intervals format: {time_intervals}. Assuming no relevant intervals.")
         time_intervals = []
         # Decide behavior: force similarity to 0? Or let calculation proceed?
         # Let's proceed, similarity will likely be 0 if timestamps exist.

    interval_timestamps = discretize_time_intervals(time_intervals)
    reference_timestamps = convert_timestamps_to_set(timestamps)
    similarity_score = calculate_jaccard_similarity(interval_timestamps, reference_timestamps)

    extraction_type, extracted_value = extract_solution(solution_str, simplified=simplified)

    # --- New Reward Logic ---
    ground_truth_str = str(ground_truth).strip()

    if extraction_type == 'answer':
        answer_str = str(extracted_value).strip()
        is_correct = (answer_str.lower() == ground_truth_str.lower())
        
        if is_correct:
            # Reward based on similarity: Higher similarity -> Higher reward
            score = similarity_score * correct_answer_reward_factor
            logger.info(f"Correct answer. Similarity: {similarity_score:.2f}. Score: {score:.2f}")
        else:
            # Penalize incorrect answers. Penalty might be reduced slightly by high similarity
            # or increased by low similarity (guessing penalty).
            score = similarity_score - incorrect_answer_base_penalty 
            logger.info(f"Incorrect answer. Similarity: {similarity_score:.2f}. Score: {score:.2f}")
        
        # Optional: Add a small base reward if the format is correct, even if the answer is wrong?
        # score = max(score, -0.8) # Ensure score doesn't drop too low just for being wrong

    elif extraction_type == 'modify':
        # Reward modification when similarity is LOW, penalize when HIGH
        score = (1.0 - similarity_score) * modification_bonus_factor - similarity_score * modification_penalty_factor
        # Ensure modifications aren't excessively penalized/rewarded unless similarity is extreme
        # score = max(min(score, modification_bonus_factor), -modification_penalty_factor) 
        logger.info(f"Modification request. Similarity: {similarity_score:.2f}. Score: {score:.2f}")

    elif extraction_type in ['mistake', 'think error', 'answer error', 'both error']:
        # Penalize formatting/logic errors
        score = error_penalty 
        logger.warning(f"Extraction error: {extraction_type}. Value: {extracted_value}. Score: {score:.2f}")
        
    else: # Should not happen, but catch all
        logger.error(f"Unknown extraction type: {extraction_type}. Assigning error penalty.")
        score = error_penalty

    # Clamp score to a [-1.0, 1.0] range (or other desired range)
    final_score = max(-1.0, min(score, 1.0)) 
    
    # Debug printing (consider making this conditional or removing in production)
    if random.randint(1, 64) == 1: # Keep infrequent debug prints
         print("--------------------------------")
         print(f"Ground_truth: {ground_truth_str}")
         print(f"Extracted Type: {extraction_type}")
         print(f"Extracted Value: {extracted_value}")
         # print(f"Solution string: {solution_str}") # Can be long
         print(f"Sampled timestamps: {reference_timestamps}")
         print(f"GT intervals: {interval_timestamps}")
         print(f"Jaccard Similarity: {similarity_score:.4f}")
         print(f"Calculated Score: {score:.4f}")
         print(f"Final Clamped Score: {final_score:.4f}")
         print("--------------------------------")

    return final_score