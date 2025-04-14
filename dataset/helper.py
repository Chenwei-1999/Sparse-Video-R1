import cv2 
import base64
import numpy as np
from openai import OpenAI

client = OpenAI()  # Initialize OpenAI client

def sample_frames(video_path, num_frames=None, frames=None, max_frames=10, method='random'):
    """
    Load a video and sample frames, returning base64-encoded images.

    Args:
      video_path (str): Path to the video file.
      num_frames (int or None): Number of frames to sample. If None, will be determined based on `frames` or randomly.
      frames (list or None): List of specific frame timestamps (in seconds) to sample. If None, frames are selected randomly.
      max_frames (int): Maximum number of frames allowed.
      method (str): Sampling method ('random' or 'uniform'). Default is 'random'.

    Returns:
      sampled_frames_base64 (list): List of sampled frames as base64-encoded strings.
      times (list): List of timestamps (in seconds) for sampled frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    sampled_frames_base64 = []
    times = []

    if total_frames == 0 or fps == 0:
        print("Error: Invalid video file.")
        cap.release()
        return [], []

    available_frames = np.arange(0, total_frames)  # All frames in video

    # If frames are provided, convert timestamps to frame indices
    if frames is not None:
        frame_indices = np.clip(np.array([int(f * fps) for f in frames]), 0, total_frames - 1)
    else:
        if num_frames is None:
            num_frames = max_frames 
        if method == 'random':
            frame_indices = np.sort(np.random.choice(available_frames, min(num_frames, total_frames), replace=False))
        elif method == 'uniform':
            frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
        else:
            print("Error: Unsupported method. Choose 'random' or 'uniform'.")
            cap.release()
            return [], []

    # Ensure frame count does not exceed max_frames
    if len(frame_indices) > max_frames:
        print(f"Warning: Reducing frames from {len(frame_indices)} to max_frames={max_frames}.")
        frame_indices = np.sort(np.random.choice(frame_indices, max_frames, replace=False))

    # Extract frames
    for frame_idx in frame_indices[:max_frames]:  # Limit to max_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            sampled_frames_base64.append(frame_to_base64(frame))

            times.append(round(frame_idx / fps, 1))  # Convert to seconds and round
        else:
            print(f"Warning: Could not read frame {frame_idx}")

    cap.release()

    return sampled_frames_base64, times


def frame_to_base64(frame):
    """
    Convert a frame (NumPy array) to a base64-encoded string.
    
    Args:
      frame (np.ndarray): The image frame.
      
    Returns:
      str: Base64-encoded string of the image.
    """
    _, buffer = cv2.imencode('.jpg', frame)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str


import re 
def extract_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text)
    
    if not match:
        return -1  # No <answer> tag found

    answer_content = match.group(1).strip()

    # Case 1: Direct numeric answer in the range 0-4
    if answer_content in ["0", "1", "2", "3", "4"]:
        return int(answer_content)
    
    # Case 2: Extracting frame modifications in the format +[x,y,z]-[a,b]
    add_match = re.search(r'\+\[(.*?)\]', answer_content)
    remove_match = re.search(r'\-\[(.*?)\]', answer_content)

    def extract_numbers(text):
        """Extract valid integers from a comma-separated string, ignoring non-numeric values."""
        if not text:
            return []
        return [int(num) for num in re.findall(r'\b\d+\b', text)]  # Extracts only valid numbers

    add_frames = extract_numbers(add_match.group(1)) if add_match else []
    remove_frames = extract_numbers(remove_match.group(1)) if remove_match else []

    # If no valid frames were found, return -1
    if not add_frames and not remove_frames:
        return -1

    return {"add": add_frames, "remove": remove_frames}

def generate_prompt(question, times):
    """
    Generate a prompt for the model based on the question and frame times.
    
    Args:
        question (str): The question to be answered
        times (list): List of timestamps for the frames
        
    Returns:
        str: Formatted prompt for the model
    """
    return f"Question: {question}\nFrames at times: {', '.join(map(str, times))}"

def get_answer(question, video_path, max_frames=5, frames=None):
    """
    Get answer for a question about a video using GPT-4.
    
    Args:
        question (str): The question to be answered
        video_path (str): Path to the video file
        max_frames (int): Maximum number of frames to sample
        frames (list, optional): Specific frames to use
        
    Returns:
        dict or int: The extracted answer
    """
    sampled_frames_base64, times = sample_frames(video_path, max_frames=max_frames, frames=frames)
    text_message = {"type": "text", "text": generate_prompt(question, times)}
    image_messages = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        for base64_image in sampled_frames_base64
    ]
    combined_content = [text_message] + image_messages
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",  # Updated to correct model name
        messages=[
            {"role": "user", "content": combined_content},
        ],
        max_tokens=10000,
    )
    answer = response.choices[0].message.content
    answer = extract_answer(answer)
    return answer
    