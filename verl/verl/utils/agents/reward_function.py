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
    Extract solution from response with enhanced error detection
    """
    # Tag validation
    think_match = re.search(r'<think>(.*?)</think>', solution_str, re.DOTALL)
    frame_match = re.search(r'<frame>(.*?)</frame>', solution_str, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    
    # Error type matrix
    if not think_match:
        return ('think_error', 'Missing <think> reasoning section', solution_str)
    if not frame_match and not answer_match:
        return ('format_error', 'Missing both <frame> and <answer> tags', solution_str)
        
    # If we have an answer tag, validate it
    if answer_match:
        answer_content = answer_match.group(1).strip()
        if not re.match(r'^\d+$', answer_content):
            return ('format_error', 'Answer must be a single number', solution_str)
        return ('valid', answer_content, solution_str)
        
    # If we have a frame tag, validate frame operations
    frame_content = frame_match.group(1).strip()
    add_match = re.search(r'\+\[(.*?)\]', frame_content)
    remove_match = re.search(r'-\[(.*?)\]', frame_content)
    
    if not (add_match or remove_match):
        return ('frame_error', 'Invalid frame operation format', solution_str)
        
    # Frame error cases
    added = parse_frame_list(add_match.group(1)) if add_match else []
    removed = parse_frame_list(remove_match.group(1)) if remove_match else []
    
    # Validate frame operations
    if any(f not in current_frames for f in removed):
        return ('frame_error', 'Trying to remove non-existent frames', solution_str)
    if len(current_frames) - len(removed) + len(added) > total_frames:
        return ('frame_error', 'Exceeds maximum allowed frames', solution_str)
    if any(f in current_frames for f in added):
        return ('frame_error', 'Adding duplicate frames', solution_str)
        
    return ('valid', {'add': added, 'remove': removed}, solution_str)

def extract_answer(text: str) -> Optional[str]:
    """
    Extract either frame operations or final answer from the response.
    
    Args:
        text (str): Input text containing frame or answer tags
        
    Returns:
        Optional[str]: Extracted frame operations or answer text, or None if invalid
    """
    if not text:
        return None
        
    # First check for frame operations
    frame_match = re.search(r'<frame>(.*?)</frame>', text, re.DOTALL)
    if frame_match:
        frame_content = frame_match.group(1).strip()
        if not frame_content:
            return None
        return frame_content
        
    # Then check for final answer
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        if not answer_content:
            return None
        return answer_content
        
    return None

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
    if extra_info is None:
        extra_info = {}
    
    # Validate required fields
    required_fields = {
        'times': 'Current frame times',
        'max_frames': 'Maximum allowed frames',
    }
    
    missing_fields = []
    for field, desc in required_fields.items():
        if field not in extra_info:
            missing_fields.append(f"{field} ({desc})")
    
    if missing_fields:
        raise ValueError(f"Missing required fields in extra_info: {', '.join(missing_fields)}")

    # Get current frame selection
    current_frames = set(extra_info['times'])
    if not current_frames:
        logger.warning("Empty current frame selection")
        return 0.0

    # Convert ground truth to set of frames
    if isinstance(ground_truth, (list, np.ndarray)):
        ground_truth_frames = set(float(t) for t in ground_truth)
    else:
        logger.warning(f"Invalid ground truth format: {type(ground_truth)}")
        return 0.0

    # Calculate Jaccard similarity
    if not ground_truth_frames:  # Empty ground truth
        return 0.0
        
    # Calculate intersection and union
    intersection = len(current_frames.intersection(ground_truth_frames))
    union = len(current_frames.union(ground_truth_frames))
    
    # Compute Jaccard similarity
    jaccard_score = intersection / union if union > 0 else 0.0
    
    # Add debug logging occasionally
    if random.randint(1, 64) == 1:
        debug_info = {
            'Current Frames': sorted(list(current_frames)),
            'Ground Truth Frames': sorted(list(ground_truth_frames)),
            'Intersection Size': intersection,
            'Union Size': union,
            'Jaccard Score': f"{jaccard_score:.4f}",
        }
        logger.info("Score Computation Debug Info:\n" + 
                   "\n".join(f"{k}: {v}" for k, v in debug_info.items()))

    return jaccard_score