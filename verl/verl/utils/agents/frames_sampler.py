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


def sample_video_frames(video_path, height=None, width=None, num_frames=5, strategy='uniform', ratio=1.0):
    """
    Args:
        video_path (str): Path to the video file.
        height (int or None): Desired height of the output image (None uses original).
        width (int or None): Desired width of the output image (None uses original).
        num_frames (int): Maximum number of frames to sample (ignored if strategy is 'all').
        strategy (str): Sampling strategy: 'uniform', 'random', or 'all'.

    Returns:
        sampled_frames: List of dicts containing JPEG bytes and image dimensions.
        sampled_times: List of timestamps in seconds (rounded to 0.1).
    """
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = int(fps)
    one_fps_frames = list(range(0, total_frames, frame_interval))

    if not one_fps_frames:
        raise ValueError("Video too short or FPS too high to sample at 1fps.")

    if strategy == 'random':
        sampled_indices = sorted(random.sample(one_fps_frames, min(num_frames, len(one_fps_frames))))
    elif strategy == 'uniform':
        if len(one_fps_frames) <= num_frames:
            sampled_indices = one_fps_frames
        else:
            step = len(one_fps_frames) / float(num_frames)
            sampled_indices = [one_fps_frames[min(int(i * step), len(one_fps_frames) - 1)] for i in range(num_frames)]
    elif strategy == 'all':
        sampled_indices = one_fps_frames[:min(num_frames, len(one_fps_frames))]
    else:
        raise ValueError("Invalid sampling strategy. Choose 'random', 'uniform', or 'all'.")

    sampled_frames = []
    sampled_times = []

    for frame_id in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        orig_width, orig_height = pil_img.size

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
            'bytes': image_bytes,
            'width': new_width,
            'height': new_height,
        }
        sampled_frames.append(frame_info)
        timestamp = round(frame_id / fps, 1)
        sampled_times.append(timestamp)
    # sort f
    cap.release()
    return sampled_frames, sampled_times

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

