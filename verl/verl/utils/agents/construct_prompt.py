import textwrap

import textwrap

def generate_prompt(question, timestamps, total_frames=None, n_round=1, max_rounds=5, max_frames=5, previous_frames=[]):
    """
    Generate a prompt string based on the provided video frames, question, and round details.

    Args:
        question (str): The question to be answered.
        timestamps (list): List of frame timestamps (in seconds).
        n_round (int): The current round number (default is 1).
        max_rounds (int): The total number of rounds available (default is 5).
        max_frames (int): The maximum number of frames allowed (default is 5).
        previous_frames (list of tuple, optional): Each tuple contains the frames selected in that round.
    Returns:
        str: A formatted prompt string.
    """
    
    def format_previous_history(previous_frames):
        """
        Format the history of previous rounds.
        
        For round 1, just show the selected frames.
        For subsequent rounds, show the actions taken and the resulting selected frames.
        """
        if len(previous_frames) == 0:
            return ""
        
        lines = ["This is the history of previous rounds:"]
        # Round 1: no actions, just the selected frames.
        lines.append(f"Round 1: the frames selected are: [{', '.join(map(str, previous_frames[0]))}]")
        
        # For rounds 2 and onward, show the action and resulting frames.
        if len(previous_frames) >= 2:
            # It is assumed that len(previous_frames) == len(previous_rounds) + 1.
            for i in range(1, len(previous_frames)):
                # add_frames, remove_frames = previous_rounds[i - 1]
                # add_str = f"add [{', '.join(map(str, add_frames))}]" if add_frames else ""
                # remove_str = f"remove [{', '.join(map(str, remove_frames))}]" if remove_frames else ""
                # action_str = f"you {add_str} {remove_str}".strip()
                # lines.append(f"Round {i + 1}: {action_str}, so the frames selected are: [{', '.join(map(str, previous_frames[i]))}]")
                lines.append(f"Round {i}: the frames selected are: [{', '.join(map(str, previous_frames[i]))}]")
        return "\n".join(lines)
    video_info = f"sampled from total {total_frames} frames (decoded at 1 fps)" if total_frames else "frames (decoded at 1 fps)"
    previous_history_str = format_previous_history(previous_frames)

    prompt = textwrap.dedent(f"""
        You have a video with {len(timestamps)} {video_info}.
        The sampled frame timestamps (in seconds) are: {timestamps}
        Please answer the following question:

        {question}

        Notice:
        If the available frames provide enough information, answer directly. Otherwise,
        specify which frames (in seconds) to add or remove to ensure the total does not exceed {max_frames} frames.
        This is round {n_round} out of {max_rounds} rounds. Please try to answer the question before the final round.

        {previous_history_str}

        You can use the following guidelines to help you decide:
        Formatting Guidelines:
        - To add frames: +[frame1, frame2, ...]
        - To remove frames: -[frame1, frame2, ...]
        - If no changes are needed, simply provide the answer.
        - Use <think>...</think> for reasoning and <answer>...</answer> for the final response.

        Examples:
        - <answer>0</answer> (if the current frames are sufficient and the answer is 0)
        - <answer>1</answer> (if the current frames are sufficient and the answer is 1)
        - <answer>chair</answer> (if the current frames are sufficient and the answer is "chair")
        - <answer>+[3,4,10]-[2,5]</answer> (to add frames 3, 4, and 10 and remove frames 2 and 5)

    """).strip()

    return prompt
