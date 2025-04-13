import re
import random
import math  # Import math for isnan check if needed later

def parse_frame_list(frame_str):
    """Helper function to parse comma-separated frame numbers safely."""
    if not frame_str:
        return []
    try:
        # Split, strip whitespace, convert to int, filter out invalid entries
        return [int(f.strip()) for f in frame_str.split(',') if f.strip().isdigit()]
    except ValueError:
        # Handle cases where conversion might fail unexpectedly, though isdigit should prevent this
        return []  # Or raise an error, depending on desired strictness

def extract_solution(solution_str):
    """
    Extracts the content within the <answer> tag.
    
    Returns a tuple: (type, value)
      - type: 'answer', 'modify', 'error'
      - value:
         - If type is 'answer': The string content of the answer.
         - If type is 'modify': A dict {'add': [list_of_ints], 'remove': [list_of_ints]}.
         - If type is 'error': An error message string.
    """
    answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)  # Use re.DOTALL for multiline content
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

    # Prioritize modification format if both '+' and '-' brackets are present,
    # or if at least one is present and it's the dominant part of the content.
    is_modification_request = (add_match and remove_match) or \
                              (add_match and content.startswith('+[')) or \
                              (remove_match and content.startswith('-[')) or \
                              (add_match and not remove_match and '+' in content and '[' in content) or \
                              (remove_match and not add_match and '-' in content and '[' in content)

    if is_modification_request:
        added_frames = parse_frame_list(add_match.group(1)) if add_match else []
        removed_frames = parse_frame_list(remove_match.group(1)) if remove_match else []

        # Ensure it's not *just* modification symbols without valid frames if matches were found
        if add_match and not added_frames and add_match.group(1).strip():
            return ('mistake', 'Invalid numbers in add list')
        if remove_match and not removed_frames and remove_match.group(1).strip():
            return ('mistake', 'Invalid numbers in remove list')

        # If we only found symbols like '+' or '-' without brackets, treat as answer
        if not add_match and not remove_match:
            return ('answer', content)

        return ('modify', {'add': added_frames, 'remove': removed_frames})

    else:
        return ('answer', content)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, 
                  format_score=0.0, correct_answer_score=1.0, no_think_score=-2.0,
                  modification_score=0.5, error_score=-1.0):
    """
    Computes the score based on the solution string, ground truth, and rules.
    
    This revised version uses extra_info to adjust scores based on conversation rounds.
    The idea is that:
      - A correct answer earns the same reward regardless of the turn.
      - A mistake (incorrect answer or invalid modification) at turn 0 is penalized more heavily.
      - A valid frame modification is rewarded, but less so if it's the first turn.
    
    Args:
        solution_str (str): The solution text from the model.
        ground_truth (any): The correct answer.
        extra_info (dict, optional): Contains additional info such as:
            'timestamps' (list): Current frame timestamps.
            'max_frames' (int): Maximum allowed frames.
            'current_turn' (int): The current round number.
            'max_turns' (int): Maximum allowed rounds.
            'type' (str): Type of data source (e.g., 'val' for validation).
        iter_decay (float): Decay factor (unused in this revision but left for compatibility).
        format_score (float): Score for invalid format/answer/modification.
        correct_answer_score (float): Score for a correct answer.
        modification_score (float): Base score for a valid modification request.
        error_score (float): Base penalty score for errors.
        mode (str): Mode indicator (e.g., 'train').
    
    Returns:
        float: The calculated score.
    """
    # Default extra_info if not provided
    if extra_info is None:
        extra_info = {}
    # Retrieve required info from extra_info
    # timestamps = extra_info.get('timestamps', None)
    # max_frames = extra_info.get('max_frames', 5)
    # current_turn = extra_info.get('current_turn', 0)
    # max_turns = extra_info.get('max_turns', 5)
    timestamps = extra_info['timestamps']
    max_frames = extra_info['max_frames']
    current_turn = extra_info['current_turn']
    max_turns = extra_info['max_turns']

    # Additional constants to encourage multi-round conversation
    # early_mistake_multiplier = 2.0    # Mistakes on turn 0 are penalized more heavily.

    # --- Extraction ---
    extraction_type, extracted_value = extract_solution(solution_str)

    # --- Debug Printing (optional) ---
    do_print = random.randint(1, 64) == 1
    if do_print:
        print("--------------------------------")
        print(f"Ground_truth: {ground_truth} ({type(ground_truth).__name__})")
        print(f"Extracted Type: {extraction_type}")
        print(f"Extracted Value: {extracted_value} ({type(extracted_value).__name__})")
        print(f"Solution string: {solution_str}")
        print(f"Max frames: {max_frames}")
        print(f"Current Turn: {current_turn}, Max Turns: {max_turns}")
        print("--------------------------------")
    # if extra_info.get('type', None) == 'val':
    answer_str = str(extracted_value).strip()
    ground_truth_str = str(ground_truth).strip()

    answer_str = str(extracted_value).strip()
    ground_truth_str = str(ground_truth).strip()
    if answer_str.lower() == ground_truth_str.lower():
        return 1
    else:
        return 0