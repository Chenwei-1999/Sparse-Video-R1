def generate_prompt(question, timestamps, max_frames=5):
    prompt = f"""
    You have a video with {len(timestamps)} frames (decoded at 1 fps).
    The sampled frame timestamps (in seconds) are: {timestamps}
    Please answer the following question:

    {question}

    If the available frames provide enough information, answer directly. Otherwise, specify which frames (in seconds) to add or remove to ensure the total does not exceed {max_frames} frames.

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