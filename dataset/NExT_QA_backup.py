import os
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import cv2
import base64
import numpy as np

from io import BytesIO

def generate_prompt(question, timestamps, frames_selection=True, max_frames=5):
    if frames_selection:
        prompt = f"""
        {"<image>" * len(timestamps)}
        You have a video with {len(timestamps)} frames (decoded at 1 fps).
        The sampled frame timestamps (in seconds) are: {timestamps}
        Please answer the following question:

        {question}

        If the available frames provide enough information, answer directly. Otherwise, specify which frames (in seconds) to add or remove to ensure the total does not exceed {max_frames} frames.

        **Formatting Guidelines:**
        - To add frames: +[frame1, frame2, ...]
        - To remove frames: -[frame1, frame2, ...]
        - If no changes are needed, simply provide the answer.
        - Use <think>...</think> for reasoning and <answer>...</answer> for the final response.

        **Examples:**
        - <answer>0</answer> (current frames are sufficient, suppose the answer is 0)
        - <answer>1</answer> (current frames are sufficient, suppose the answer is 1)
        - <answer>+[3,4,10]-[2,5]</answer> (add frames 3, 4, and 10; remove frames 2 and 5)

        <think>
        """
    else:
        prompt = f"""
        {"<image>" * len(timestamps)}
        You have a video with {len(timestamps)} frames (decoded at 1 fps).
        The sampled frame timestamps (in seconds) are: {timestamps}
        Please answer the following question:

        {question}

        <think>
        """
    return prompt

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.png', frame)
    return base64.b64encode(buffer).decode('utf-8')

def frame_to_bytes(frame):
    _, buffer = cv2.imencode('.png', frame)
    return buffer.tobytes()
def sample_frames(video_path, num_frames=None, frames=None, max_frames=10, method='random'):
    """
    Load a video and sample frames, returning base64-encoded PNG images and their timestamps.
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

    available_frames = np.arange(0, total_frames)

    # Decide how to pick the frame indices
    if frames is not None:
        frames = [frame for frame in frames if frame != -1]
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

    # Enforce max_frames if needed
    if len(frame_indices) > max_frames:
        print(f"Warning: Reducing frames from {len(frame_indices)} to max_frames={max_frames}.")
        frame_indices = np.sort(np.random.choice(frame_indices, max_frames, replace=False))

    # Read the frames we want
    for frame_idx in frame_indices[:max_frames]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_base64 = frame_to_bytes(frame)
            if frame_base64:
                sampled_frames_base64.append(frame_base64)
                times.append(round(frame_idx / fps, 1))
        else:
            print(f"Warning: Could not read frame {frame_idx}")
    cap.release()
    return sampled_frames_base64, times

def create_data(QA_data, vid_to_vidor, parent_directory, method='random', num_frames=5, max_frames=5):
    dataset = []

    for video_id in QA_data['video']:
        video_vidor = vid_to_vidor.get(str(video_id))
        if not video_vidor:
            continue  # Skip if video_id is not found
        video_path = os.path.join(parent_directory, video_vidor) + ".mp4"

        if os.path.exists(video_path):
            frame_base64_list, times = sample_frames(
                video_path,
                max_frames=num_frames,
                method=method
            )
            # print(f"video_id: {video_id}, video_path: {video_path}, frames: {len(frame_base64_list)}")

            qa_entry = QA_data[QA_data['video'] == video_id]
            # Extract question, answer, and answer choices.
            question = qa_entry['question'].values[0]
            answer = qa_entry['answer'].values[0]
            a0 = qa_entry['a0'].values[0]
            a1 = qa_entry['a1'].values[0]
            a2 = qa_entry['a2'].values[0]
            a3 = qa_entry['a3'].values[0]
            a4 = qa_entry['a4'].values[0]

            # Build a prompt that includes your original multi-GPU (video/frame) info.
            q_prompt = f"{question} \n 0: {a0} \n 1: {a1} \n 2: {a2} \n 3: {a3} \n 4: {a4}"
            full_prompt = generate_prompt(q_prompt, times, max_frames=num_frames)

            # Create content list with multiple images followed by the text prompt
            images = [frame_base64 for frame_base64 in frame_base64_list]

            # Additional info
            frame_count = qa_entry['frame_count'].values[0]
            width = qa_entry['width'].values[0]
            height = qa_entry['height'].values[0]
            qid = qa_entry['qid'].values[0]
            q_type = qa_entry['type'].values[0]
            extra_info = {
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "qid": qid,
                "type": q_type
            }
            frames_total = [-1] * max_frames
            frames_total[:len(times)] = times
            # Build the final data sample.
            sample = {
                "id": qid,
                "prompt": full_prompt,
                "q_prompt": q_prompt,
                "images": images,
                "answer": answer,
                "data_source": "VideoQA",
                "num_frames": num_frames,
                "times": frames_total,
                "extra_info": extra_info,
                "video_path": video_path,
                # "prompt": full_prompt,  # We won't include "prompt" in the final columns if you want the 4-column base
            }
            dataset.append(sample)

  
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="/shares/hlw3876/chenwei/NExT-QA/parquet")
    parser.add_argument("--parent_directory", type=str, default='/shares/hlw3876/chenwei/NExT-QA')
    parser.add_argument("--method", type=str, default='random', choices=['random', 'uniform', 'importance'])
    parser.add_argument("--frames_selection", type=bool, default=True, help="Let the llm select frames or not")
    parser.add_argument("--num_frames", type=int, default=3, help="Number of frames to sample from the video")
  
    args = parser.parse_args()
    with open(os.path.join(args.parent_directory, "map_vid_vidorID.json"), "r") as f:
        vid_to_vidor = json.load(f)
    
    video_directory = os.path.join(args.parent_directory, "NExTVideo")
    
    if args.mode == "train":
        # Create train data.
        QA_file = os.path.join(args.parent_directory, "nextqa/train.csv")
        QA_data = pd.read_csv(QA_file)
        df = create_data(QA_data, vid_to_vidor, video_directory, method=args.method, num_frames=args.num_frames)
        table = pa.Table.from_pylist(df)
        pq.write_table(table, os.path.join(args.output_dir, "train.parquet"))
        print(f"Train data created with {len(df)} samples")
        
        # Create validation data.
        QA_file = os.path.join(args.parent_directory, "nextqa/val.csv")
        QA_data = pd.read_csv(QA_file)
        df = create_data(QA_data, vid_to_vidor, video_directory, method=args.method, num_frames=args.num_frames)
        table = pa.Table.from_pylist(df)
        pq.write_table(table, os.path.join(args.output_dir, "val.parquet"))
        print(f"Validation data created with {len(df)} samples")
 
