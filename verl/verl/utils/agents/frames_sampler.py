import cv2
import random
import base64
from io import BytesIO
from PIL import Image
import os
import re
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


def encode_image_to_base64(image_bytes):
    return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"


import cv2
import random
from PIL import Image
from io import BytesIO

def sample_video_frames(video_path, height=None, width=None, num_frames=5, strategy='uniform', ratio=1.0):
    """
    Args:
        video_path (str): Path to the video file.
        height (int or None): Desired height of the output image (None uses original).
        width (int or None): Desired width of the output image (None uses original).
        num_frames (int): Maximum number of frames to sample from the 1fps candidates (ignored if strategy is 'all').
        strategy (str): Sampling strategy: 'uniform', 'random', or 'all'.
        ratio (float): A scaling factor applied to width and height.
        
    Returns:
        sampled_frames: List of dicts, each containing JPEG bytes and image dimensions.
        sampled_times: List of timestamps (seconds) corresponding to the sampled frames.
        total_1fps_frames: Total number of candidate 1fps frames available in the video.
    """
    # Adjust dimensions using the ratio (scaling factor)
    if width is not None:
        width = int(ratio * width)
    if height is not None:
        height = int(ratio * height)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("Invalid or unreadable FPS value from video.")
    
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames_in_video / fps  # Duration in seconds.
    
    # Build candidate list: for each whole second, store (frame_index, timestamp)
    candidates = []
    for sec in range(0, int(video_duration) + 1):
        frame_idx = int(round(sec * fps))
        if frame_idx < total_frames_in_video:
            candidates.append((frame_idx, float(sec)))
        else:
            break

    total_1fps_frames = len(candidates)

    # Select a subset of the candidates based on the strategy.
    if strategy == 'random':
        if num_frames < total_1fps_frames:
            sampled_candidates = sorted(random.sample(candidates, num_frames), key=lambda x: x[0])
        else:
            sampled_candidates = candidates
    elif strategy == 'uniform':
        if num_frames >= total_1fps_frames:
            sampled_candidates = candidates
        else:
            step = total_1fps_frames / float(num_frames)
            sampled_candidates = [candidates[min(int(i * step), total_1fps_frames - 1)] for i in range(num_frames)]
    elif strategy == 'all':
        sampled_candidates = candidates
    else:
        raise ValueError("Invalid sampling strategy. Choose 'random', 'uniform', or 'all'.")
    
    sampled_frames = []
    sampled_times = []
    
    # Extract each sampled frame from the video.
    for frame_idx, timestamp in sampled_candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert the frame from BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        orig_width, orig_height = pil_img.size
        
        # Resize if desired dimensions are provided.
        if width is not None or height is not None:
            if width is None:
                ratio_val = height / float(orig_height)
                new_width = int(orig_width * ratio_val)
                new_height = height
            elif height is None:
                ratio_val = width / float(orig_width)
                new_width = width
                new_height = int(orig_height * ratio_val)
            else:
                new_width, new_height = width, height
            pil_img = pil_img.resize((new_width, new_height))
        else:
            new_width, new_height = orig_width, orig_height
        
        # Convert the PIL image to JPEG bytes.
        buf = BytesIO()
        pil_img.save(buf, format='JPEG')
        image_bytes = buf.getvalue()
        buf.close()
        
        frame_info = {
            'bytes': image_bytes,
            'width': new_width,
            'height': new_height,
        }
        sampled_frames.append(frame_info)
        sampled_times.append(timestamp)
    
    cap.release()
    return sampled_frames, sampled_times, total_1fps_frames


def sample_frames_from_next_obs(video_path: str, timestamps: list, height: int = None, width: int = None, ratio=1.0) -> list:
    """
    Sample frames from a video at specified timestamps.
    Args:
        video_path (str): Path to the video file.
        timestamps (list): List of timestamps in seconds where frames should be sampled.
        height (int or None): Desired height of the output image (None uses original).
        width (int or None): Desired width of the output image (None uses original).
        ratio (float): Scaling factor for width and height.
    Returns:
        sampled_frames: List of dicts containing JPEG bytes and image dimensions.
    """
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("Invalid or unreadable FPS value from video.")
    sampled_frames = []
    sampled_times = []
    for ts in timestamps:
        # Convert the timestamp (in seconds) to a frame index.
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        orig_width, orig_height = pil_img.size

        # Resize the image if either height or width is provided.
        if width is not None or height is not None:
            if width is None:
                ratio = height / float(orig_height)
                new_width = int(orig_width * ratio)
                new_height = height
            elif height is None:
                ratio = width / float(orig_width)
                new_width = width
                new_height = int(orig_height * ratio)
            else:
                new_width, new_height = width, height
            pil_img = pil_img.resize((new_width, new_height))
        else:
            new_width, new_height = orig_width, orig_height

        buf = BytesIO()
        pil_img.save(buf, format='JPEG')
        image_bytes = buf.getvalue()
        buf.close()

        frame_info = {
            "bytes": image_bytes,
            "width": new_width,
            "height": new_height,
        }
        sampled_frames.append(frame_info)
    
    cap.release()
    return sampled_frames

