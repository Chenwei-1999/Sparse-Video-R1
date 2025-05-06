import textwrap
from typing import List, Optional, Dict, Any


def generate_prompt(
    question: str,
    timestamps: List[int],
    total_times: Optional[int] = None,
    n_round: int = 1,
    max_rounds: int = 5,
    max_frames: int = 5,
    first_round: bool = True
) -> str:
    """
    Generate a prompt string with improved structure and error handling.
    
    Args:
        question: The question to be answered
        timestamps: List of frame timestamps (in seconds)
        total_times: Total number of frames in video (optional)
        n_round: Current round number (default: 1)
        max_rounds: Maximum number of rounds (default: 5)
        max_frames: Maximum number of frames allowed (default: 5)

    Returns:
        str: Formatted prompt string
    """
    if not timestamps:
        raise ValueError("timestamps list cannot be empty")
    if n_round < 1 or n_round > max_rounds:
        raise ValueError(f"n_round must be between 1 and {max_rounds}")
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")
        
    # Format video info
    video_info = f"sampled from total {total_times} frames (decoded at 1 fps)"


    format_prompt = f"""  
        Notice:
        - If the available frames provide enough information, answer directly in <answer></answer>, for example, <answer>1</answer>.
        - Otherwise, specify which frames to add/remove in <frames></frames>, for example, <frames>+[1, 2, 3]</frames>.
        - This is round {n_round} of {max_rounds}. Try to answer before the final round.
        - You are allowed to add/remove frames and made total frames up to {max_frames} frames.
        - Use <think></think> for reasoning before you answer or add/remove frames, for example, <think>I need to add frames 1, 2, 3 to answer the question.</think>
        """ if first_round else ""
    info_prompt = f"You have a video with {len(timestamps)} {video_info}." if first_round else ""


    return f"""
        {info_prompt}

        The sampled frame timestamps (in seconds) are: {timestamps}.
        Please answer the following question or adjust the frames to help you answer the question:
        
        {question}
      
        {format_prompt}
    """
