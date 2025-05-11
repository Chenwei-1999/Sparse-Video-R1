import textwrap
from typing import List, Optional, Dict, Any


def generate_prompt(
    question: str,
    timestamps: List[int],
    total_times: Optional[int] = None,
    n_round: int = 1,
    max_rounds: int = 5,
    max_frames: int = 5,
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
        - Otherwise, specify which frames to use in <frames></frames>, for example, <frames>1, 2, 3</frames> means use frames in 1, 2, 3 seconds.
        - You are allowed to select up to {max_frames} frames in this round.
        - Use <think></think> for reasoning before you answer the question or select frames, for example, <think>I need to add frames from 1, 2, 3 seconds to answer the question.</think>
        """ if n_round==1 else ""
    info_prompt = f"You have a video with {len(timestamps)} {video_info}." if n_round==1  else ""


    return f"""
        {info_prompt}

        The sampled frame timestamps (in seconds) are: {timestamps}.
        Please answer the following question or adjust the frames to help you answer the question:
        
        {question}
      
        {format_prompt}
        - This is round {n_round} of {max_rounds}. Try to answer before the final round.
    """

def generate_prompt_for_force_round(
    question: str,
    timestamps: List[int],
    total_times: Optional[int] = None,
    n_round: int = 1,
    max_rounds: int = 5,
    max_frames: int = 5,
) -> str:
    """
    Generate a prompt string for force round.
    For previous 4 rounds, the agent will ask for only select frames.
    For the final round, the agent will answer the question directly.
    """
    video_info = f"sampled from total {total_times} frames (decoded at 1 fps)"

    if n_round == max_rounds:
        round_prompt = f"""
        This is the final round, so you can answer the question directly.
        You should use <answer></answer> to answer the question, for example, <answer>1</answer>. 
        """
    else:
        round_prompt = f"""
        This is round {n_round} of {max_rounds}, so you should use <think></think> to think about the question and <frames></frames> to select frames.
        For example, <think>I need to add frames from 1, 2, 3 seconds to answer the question.</think><frames>1, 2, 3</frames>.
        In this round, you are allowed to select up to {max_frames} frames.
        """
    video_info = f"sampled from total {total_times} frames (decoded at 1 fps)"


    return f"""
    You have a video with {len(timestamps)} {video_info}.
    The current sampled frame timestamps (in seconds) are: {timestamps}.
    Please answer the following question or adjust the frames to help you answer the question:
    
    {question}
    {round_prompt}
    """
