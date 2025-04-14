import re
import random
import math
from typing import Dict, Any, Union, Tuple, Optional, List
import logging

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

def extract_solution(solution_str: str) -> Tuple[str, Union[int, Dict[str, list]]]:
    """
    Extract the solution type and value from the solution string.
    
    The solution string should contain:
    - <think> tags for reasoning
    - <answer> tags for the actual answer
    - Optional frame modifications in format +[frames] or -[frames]
    
    Args:
        solution_str (str): The solution text from the model
        
    Returns:
        Tuple[str, Union[int, Dict[str, list]]]: A tuple containing:
            - The type of solution ('answer', 'modification', or error type)
            - The extracted value (answer, modification dict, or error message)
    """
    answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
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

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    format_score: float = 0.0,
    correct_answer_score: float = 1.0,
    no_think_score: float = -2.0,
    modification_score: float = 0.5,
    error_score: float = -1.0
) -> float:
    """
    Compute the score based on the solution string, ground truth, and rules.
    
    The scoring system:
    1. Rewards correct answers with full points
    2. Penalizes incorrect answers
    3. Rewards valid frame modifications
    4. Penalizes invalid modifications or format errors
    
    Args:
        data_source (str): Source of the data (e.g., 'video', 'text')
        solution_str (str): The solution text from the model
        ground_truth (Any): The correct answer or ground truth
        extra_info (Dict[str, Any], optional): Additional context containing:
            - timestamps (list): Current frame timestamps
            - max_frames (int): Maximum allowed frames
            - current_turn (int): Current conversation turn
            - max_turns (int): Maximum allowed turns
            - type (str): Data type (e.g., 'val' for validation)
        format_score (float): Score for invalid format/answer/modification
        correct_answer_score (float): Score for a correct answer
        no_think_score (float): Penalty for not thinking/analyzing
        modification_score (float): Base score for valid modifications
        error_score (float): Base penalty for errors
        
    Returns:
        float: The calculated score in range [-2.0, 1.0]
        
    Raises:
        ValueError: If required extra_info fields are missing
    """
    # Validate extra_info
    if extra_info is None:
        extra_info = {}
    
    required_fields = ['timestamps', 'max_frames', 'current_turn', 'max_turns']
    missing_fields = [field for field in required_fields if field not in extra_info]
    if missing_fields:
        raise ValueError(f"Missing required fields in extra_info: {missing_fields}")
    
    # Extract values from extra_info
    timestamps = extra_info['timestamps']
    max_frames = extra_info['max_frames']
    current_turn = extra_info['current_turn']
    max_turns = extra_info['max_turns']
    
    # Extract solution type and value
    extraction_type, extracted_value = extract_solution(solution_str)
    
    # Debug printing (optional)
    if random.randint(1, 64) == 1:
        print("--------------------------------")
        print(f"Ground_truth: {ground_truth} ({type(ground_truth).__name__})")
        print(f"Extracted Type: {extraction_type}")
        print(f"Extracted Value: {extracted_value} ({type(extracted_value).__name__})")
        print(f"Solution string: {solution_str}")
        print(f"Max frames: {max_frames}")
        print(f"Current Turn: {current_turn}, Max Turns: {max_turns}")
        print("--------------------------------")
    
    # Compare answer with ground truth
    answer_str = str(extracted_value).strip()
    ground_truth_str = str(ground_truth).strip()
    
    if answer_str.lower() == ground_truth_str.lower():
        return correct_answer_score
    else:
        return error_score