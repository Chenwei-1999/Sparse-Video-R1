import re
import random
import math # Import math for isnan check if needed later

def parse_frame_list(frame_str):
    """Helper function to parse comma-separated frame numbers safely."""
    if not frame_str:
        return []
    try:
        # Split, strip whitespace, convert to int, filter out invalid entries
        return [int(f.strip()) for f in frame_str.split(',') if f.strip().isdigit()]
    except ValueError:
        # Handle cases where conversion might fail unexpectedly, though isdigit should prevent this
        return [] # Or raise an error, depending on desired strictness

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
    match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL) # Use re.DOTALL for multiline content

    if not match:
        return ('error', 'No <answer> tag found')

    content = match.group(1).strip()

    # Check for frame modification patterns
    add_match = re.search(r'\+\s*\[(.*?)\]', content)
    remove_match = re.search(r'-\s*\[(.*?)\]', content)

    # Prioritize modification format if both '+' and '-' brackets are present,
    # or if at least one is present and it's the dominant part of the content.
    # A simple check: does the content *start* with '+' or '-' followed by '['?
    # Or contains both patterns?
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
             # Found +[] but content inside wasn't valid numbers
             return ('error', 'Invalid numbers in add list')
        if remove_match and not removed_frames and remove_match.group(1).strip():
             # Found -[] but content inside wasn't valid numbers
             return ('error', 'Invalid numbers in remove list')

        # If we only found symbols like '+' or '-' without brackets, treat as answer
        if not add_match and not remove_match:
             return ('answer', content)

        # If we found at least one valid list or an empty list marker
        return ('modify', {'add': added_frames, 'remove': removed_frames})

    else:
        # Treat as a direct answer
        return ('answer', content)


def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  format_score=0.0, correct_answer_score=1.0,
                  modification_score=0.1, error_score=-1.0, mode='train'):
    """
    Computes the score based on the solution string, ground truth, and rules.

    Args:
        solution_str (str): The solution text from the model.
        ground_truth (any): The correct answer. Can be string, int, etc.
        extra_info (dict, optional): Dictionary containing additional info like:
            'timestamps' (list): Current frame timestamps. Required for modification checks.
            'max_frames' (int): Maximum allowed frames. Required for modification checks.
            'current_turn' (int): The current turn number. Required for max_turns check.
            'max_turns' (int): The maximum number of turns allowed. Required for max_turns check.
            'n_iter' (int): Number of iterations used (optional, for penalty).
            'max_iter' (int): Max iterations allowed without penalty (optional).
            'iter_decay' (float): Decay factor for iterations (optional).
            'n_frames' (int): Number of frames used (optional, for penalty, distinct from len(timestamps)).
            'frame_decay' (float): Decay factor for frames used (optional).
            'row_chat' (str): Raw chat log (optional, for debugging).
            'prompt' (str): Prompt used (optional, for debugging).
            'type' (str): Type of the data source, val for validation data.
        format_score (float): Score for invalid format/answer/modification (Case 1). Defaults to 0.0.
        correct_answer_score (float): Base score for a correct answer (Case 3). Defaults to 1.0.
        modification_score (float): Score for a valid modification request (Case 2). Defaults to 0.1.
        error_score (float): Score for missing <answer> tag (Case 4). Defaults to -1.0.


    Returns:
        float: The calculated score based on the rules.
    """
    if extra_info is None:
        extra_info = {}


    # Ensure necessary info is present for certain checks
    timestamps = extra_info.get('timestamps', None)
    max_frames = extra_info.get('max_frames', 5)
    current_turn = extra_info.get('current_turn', 0)
    max_turns = extra_info.get('max_turns', 5)

    # --- Extraction ---
    extraction_type, extracted_value = extract_solution(solution_str)

    # --- Debug Printing ---
    do_print = random.randint(1, 64) == 1 # Keep random printing?
    if do_print:
        print(f"--------------------------------")
        print(f"Ground_truth: {ground_truth} ({type(ground_truth).__name__})")
        print(f"Extracted Type: {extraction_type}")
        print(f"Extracted Value: {extracted_value} ({type(extracted_value).__name__})")
        print(f"Solution string: {solution_str}")
        if 'row_chat' in extra_info: print(f"Raw Chat: \n {extra_info['row_chat']}")
        if 'prompt' in extra_info: print(f"Prompt: \n {extra_info['prompt']}")
        # print(f"Solution string: {solution_str}")
        print(f"Max frames: {max_frames}")
        print(f"Current Turn: {current_turn}, Max Turns: {max_turns}")
        print(f"--------------------------------")


    # --- Scoring Logic ---
    if extra_info['type'] == 'val':
        answer_str = str(extracted_value).strip()
        ground_truth_str = str(ground_truth).strip()
        if answer_str.lower() == ground_truth_str.lower():
            return 1
        else:
            return 0
        
    # Case 4: No <answer> tag found or other critical extraction error
    if extraction_type == 'error':
        # You might want finer control here based on the error message in extracted_value
        # For now, map all errors to the 'do nothing' score
        return error_score # -1

    # --- Check Max Turns Condition ---
    # If it's the last turn, only a direct answer is acceptable. Modifications fail.
    is_last_turn = (current_turn is not None and max_turns is not None and current_turn >= max_turns)
    if is_last_turn and extraction_type == 'modify':
        print("DEBUG: Modification request on last turn - returning format_score") # Debug helper
        return format_score # 0 (Case 1)

    # Case 3: Direct Answer Provided
    if extraction_type == 'answer':
        # Convert both to string for comparison, handling potential type mismatches
        # Be careful if ground_truth could be None or NaN etc.
        answer_str = str(extracted_value).strip()
        ground_truth_str = str(ground_truth).strip()

        # Simple string comparison - adjust if more complex logic (e.g., numeric ranges) is needed
        if answer_str.lower() == ground_truth_str.lower():
            # Correct Answer! Start with the base score for correctness
            score = correct_answer_score # 1.0

            # Apply optional penalties for resource usage (iter/frames)
            n_iter = extra_info.get("n_iter")
            max_iter = extra_info.get("max_iter")
            iter_decay = extra_info.get("iter_decay")
            if n_iter and max_iter and iter_decay and iter_decay < 1.0:
                iter_penalty_count = max(0, n_iter - max_iter)
                score *= iter_decay ** iter_penalty_count

            # Note: n_frames in extra_info might be different from len(timestamps)
            # Use the one relevant for penalty calculation
            n_frames_used = extra_info.get("n_frames")
            frame_decay = extra_info.get("frame_decay")
            # Using max_frames from extra_info as the threshold for frame penalty
            if n_frames_used and max_frames and frame_decay and frame_decay < 1.0:
                frame_penalty_count = max(0, n_frames_used - max_frames)
                score *= frame_decay ** frame_penalty_count

            return score # Case 3 (potentially penalized)
        else:
            # Incorrect Answer
            return format_score # 0 (Case 1)

    # Case 2 & 1 (partially): Frame Modification Request
    if extraction_type == 'modify':
        # Need timestamps and max_frames to validate
        if timestamps is None or max_frames is None:
             print("DEBUG: Missing timestamps or max_frames for modification check - returning format_score") # Debug helper
             return format_score # Cannot validate, treat as invalid (Case 1)

        added = extracted_value['add']
        removed = extracted_value['remove']
        current_frame_count = len(timestamps)
        new_frame_count = current_frame_count + len(added) - len(removed)

        # --- Validation Checks for Modification ---
        # 1. Must add frames if requesting modification? (User prompt implies this: "only remove no add, return 0")
        #    Let's assume modification means *some* change intended, but *only* removing is invalid.
        #    However, +[] -[valid] could be valid if the *intent* was modification.
        #    Stricter check: Must add at least one frame? Let's go with "Cannot *only* remove".
        only_removing = len(added) == 0 and len(removed) > 0

        # 2. Does the resulting number of frames exceed the maximum?
        exceeds_max_frames = new_frame_count > max_frames

        # 3. Are there any frames to add or remove? (An empty modification like '+[] -[]' is likely invalid)
        no_change_requested = len(added) == 0 and len(removed) == 0

        # Apply Case 1 rules (return format_score = 0)
        if only_removing or exceeds_max_frames or no_change_requested:
             print(f"DEBUG: Invalid modification - only_removing:{only_removing}, exceeds_max:{exceeds_max_frames}, no_change:{no_change_requested} - returning format_score") # Debug helper
             return format_score # 0 (Case 1)
        else:
             # Valid modification request
             return modification_score # 0.1 (Case 2)

    # Fallback - should not be reached if extract_solution is correct
    print("DEBUG: Reached unexpected fallback in compute_score - returning error_score") # Debug helper
    return error_score # -1