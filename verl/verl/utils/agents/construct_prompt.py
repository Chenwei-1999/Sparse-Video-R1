def generate_prompt(question, timestamps, n_turns=1, max_turns=5, max_frames=5):
    prompt = f"""
    You have a video with {len(timestamps)} frames (decoded at 1 fps).
    The sampled frame timestamps (in seconds) are: {timestamps}
    Please answer the following question:

    {question}

    Notice:
    If the available frames provide enough information, answer directly. Otherwise, specify which frames (in seconds) to add or remove to ensure the total does not exceed {max_frames} frames.
    This is your {n_turns} turn of interaction, and you can interact up to {max_turns} times. Please finish your answer as quick as possible.
    You can use the following guidelines to help you decide:

    Formatting Guidelines:
    - To add frames: +[frame1, frame2, ...]
    - To remove frames: -[frame1, frame2, ...]
    - If no changes are needed, simply provide the answer.
    - Use <think>...</think> for reasoning and <answer>...</answer> for the final response.

    Examples:
    - <answer>0</answer> (current frames are sufficient, suppose the answer is 0)
    - <answer>1</answer> (current frames are sufficient, suppose the answer is 1)
    - <answer>chair</answer> (current frames are sufficient, suppose the answer is chair)
    - <answer>+[3,4,10]-[2,5]</answer> (add frames 3, 4, and 10; remove frames 2 and 5)

    <think>
    """

    return prompt