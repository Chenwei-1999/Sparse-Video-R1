import cv2
import random
import base64
from io import BytesIO
from PIL import Image
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_image_to_base64(image_bytes: bytes) -> str:
    """
    Encode image bytes to base64 string.
    
    Args:
        image_bytes (bytes): Raw image bytes
        
    Returns:
        str: Base64 encoded image string
    """
    return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

def validate_video_file(video_path: str) -> Tuple[float, float]:
    """
    Validate video file and get its properties.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        Tuple[float, float]: (fps, duration)
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video file is invalid or unreadable
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError("Invalid or unreadable FPS value from video.")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    cap.release()
    return fps, duration

def sample_video_frames(
    video_path: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: int = 5,
    strategy: str = 'uniform',
    ratio: float = 1,
    cache_dir: Optional[str] = None,
    use_cache: bool = True
) -> Tuple[List[Dict[str, Any]], List[float], int]:
    """
    Sample frames from a video with improved caching and error handling.
    
    Args:
        video_path: Path to the video file
        height: Desired height of output frames
        width: Desired width of output frames
        num_frames: Number of frames to sample
        strategy: Sampling strategy ('uniform', 'random', 'all')
        ratio: Scaling factor for width and height
        cache_dir: Directory to store cached frames
        use_cache: Whether to use frame caching
        
    Returns:
        Tuple containing:
            - List of frame dictionaries with 'image', 'width', 'height'
            - List of timestamps
            - Total number of 1fps frames available
    """
    # Parameter validation
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1")
    if strategy not in ['uniform', 'random', 'all']:
        raise ValueError("strategy must be 'uniform', 'random', or 'all'")
    if ratio <= 0 or ratio > 1:
        raise ValueError("ratio must be between 0 and 1")
    
    # Get video properties with retry
    max_retries = 3
    for attempt in range(max_retries):
        try:
            fps, duration = validate_video_file(video_path)
            total_frames = int(fps * duration)
            break
        except (ValueError, FileNotFoundError) as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {str(e)}")
            continue
    
    # Apply scaling to dimensions
    if ratio != 1:
        width = int(width * ratio) if width else None
        height = int(height * ratio) if height else None
    
    # Build frame candidates list
    candidates = []
    for sec in range(0, int(duration) + 1):
        frame_idx = int(round(sec * fps))
        if frame_idx < total_frames:
            candidates.append((frame_idx, float(sec)))
    
    total_1fps_frames = len(candidates)
    if not candidates:
        raise ValueError("No valid frames found in video")
    
    # Select frames based on strategy
    if strategy == 'all':
        strategy = random.choice(['random', 'uniform'])
    
    if strategy == 'random':
        sampled_candidates = sorted(
            random.sample(candidates, min(num_frames, total_1fps_frames)),
            key=lambda x: x[0]
        )
    else:  # uniform
        if num_frames >= total_1fps_frames:
            sampled_candidates = candidates
        else:
            step = total_1fps_frames / float(num_frames)
            sampled_candidates = [
                candidates[min(int(i * step), total_1fps_frames - 1)]
                for i in range(num_frames)
            ]
    
    # Extract and process frames
    sampled_frames = []
    sampled_times = []
    cap = cv2.VideoCapture(video_path)
    
    try:
        for frame_idx, timestamp in sampled_candidates:
            # Check cache first if enabled
            cache_key = None
            if use_cache and cache_dir:
                cache_key = f"{os.path.basename(video_path)}_{frame_idx}_{width}x{height}.jpg"
                cache_path = os.path.join(cache_dir, cache_key)
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, 'rb') as f:
                            image_bytes = f.read()
                        sampled_frames.append({
                            'image': encode_image_to_base64(image_bytes),
                            'width': width or 0,
                            'height': height or 0,
                        })
                        sampled_times.append(timestamp)
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to read cached frame: {str(e)}")
            
            # Read frame from video
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None or frame.size == 0:
                logger.warning(f"Failed to read frame {frame_idx}")
                continue
            
            try:
                # Convert and resize frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Calculate dimensions
                if width or height:
                    orig_width, orig_height = pil_img.size
                    if width is None:
                        width = int(orig_width * (height / orig_height))
                    elif height is None:
                        height = int(orig_height * (width / orig_width))
                    pil_img = pil_img.resize((width, height))
                else:
                    width, height = pil_img.size
                
                # Convert to bytes
                buf = BytesIO()
                pil_img.save(buf, format='JPEG', quality=95)
                image_bytes = buf.getvalue()
                buf.close()
                
                # Cache frame if enabled
                if use_cache and cache_dir and cache_key:
                    os.makedirs(cache_dir, exist_ok=True)
                    try:
                        with open(os.path.join(cache_dir, cache_key), 'wb') as f:
                            f.write(image_bytes)
                    except Exception as e:
                        logger.warning(f"Failed to cache frame: {str(e)}")
                
                sampled_frames.append({
                    'image': encode_image_to_base64(image_bytes),
                    'width': width,
                    'height': height,
                })
                sampled_times.append(timestamp)
                
            except Exception as e:
                logger.warning(f"Error processing frame {frame_idx}: {str(e)}")
                continue
    
    finally:
        cap.release()
    
    if not sampled_frames:
        raise ValueError(f"No frames could be successfully sampled from video {video_path}")
    
    return sampled_frames, sampled_times, total_1fps_frames

def sample_frames_from_next_obs(
    video_path: str,
    timestamps: List[float],
    height: Optional[int] = None,
    width: Optional[int] = None,
    ratio: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Sample specific frames from a video at given timestamps.
    
    Args:
        video_path (str): Path to the video file
        timestamps (List[float]): List of timestamps to sample
        height (int, optional): Desired height of output frames
        width (int, optional): Desired width of output frames
        ratio (float): Scaling factor for width and height
        
    Returns:
        List of frame dictionaries with 'image' (base64 encoded), 'width', 'height'
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video is invalid or timestamps are invalid
    """
    # Validate parameters
    if len(timestamps) == 0:
        raise ValueError("timestamps list cannot be empty")
    if ratio <= 0 or ratio > 1:
        raise ValueError("ratio must be between 0 and 1")
    
    # Get video properties
    fps, duration = validate_video_file(video_path)
    
    # Validate timestamps and filter out invalid ones
    valid_timestamps = []
    invalid_timestamps = []
    for ts in timestamps:
        if 0 <= ts <= duration:
            valid_timestamps.append(ts)
        else:
            invalid_timestamps.append(ts)
            logger.warning(f"Timestamp {ts} is outside video duration {duration}, skipping")
    
    if not valid_timestamps:
        raise ValueError(f"No valid timestamps found. All timestamps were invalid: {invalid_timestamps}")
    
    # Adjust dimensions
    if width is not None:
        width = int(ratio * width)
    if height is not None:
        height = int(ratio * height)
    
    # Sample frames
    sampled_frames = []
    cap = cv2.VideoCapture(video_path)
    
    try:
        for ts in valid_timestamps:
            frame_idx = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame at timestamp {ts}")
                continue
            
            # Convert and resize frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            if width is not None or height is not None:
                orig_width, orig_height = pil_img.size
                if width is None:
                    new_width = int(orig_width * (height / orig_height))
                    new_height = height
                elif height is None:
                    new_width = width
                    new_height = int(orig_height * (width / orig_width))
                else:
                    new_width, new_height = width, height
                pil_img = pil_img.resize((new_width, new_height))
            else:
                new_width, new_height = pil_img.size
            
            # Convert to bytes and encode to base64
            buf = BytesIO()
            pil_img.save(buf, format='JPEG', quality=95)  # Match quality with sample_video_frames
            image_bytes = buf.getvalue()
            buf.close()
            
            sampled_frames.append({
                'image': encode_image_to_base64(image_bytes),  # Use base64 encoding like sample_video_frames
                'width': new_width,
                'height': new_height,
            })
    
    finally:
        cap.release()
    
    if not sampled_frames:
        raise ValueError("No frames were successfully sampled from the video")
    
    return sampled_frames

