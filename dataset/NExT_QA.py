import os
import argparse
import pandas as pd
import json


def create_data(QA_data, vid_to_vidor, parent_directory, mode='train', sample_size=None):
    """
    Create dataset entries from QA data and video information.
    
    Args:
        QA_data (pd.DataFrame): DataFrame containing question-answer pairs and video information
        vid_to_vidor (dict): Mapping from video IDs to video paths
        parent_directory (str): Base directory containing video files
        mode (str): Dataset mode ('train', 'val', or 'test')
        sample_size (int, optional): Number of samples to create. If None, uses all data.
        
    Returns:
        list: List of dataset entries with video, question, and answer information
    """
    if sample_size is not None:
        QA_data = QA_data.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} entries from the dataset for {mode} mode.")
    else:
        print(f"Using the full dataset for {mode} mode.")
    dataset = []
    id = 0
    for video_id in QA_data['video']:
        video_vidor = vid_to_vidor.get(str(video_id))
        if not video_vidor:
            continue  # Skip if video_id is not found
        video_path = os.path.join(parent_directory, video_vidor) + ".mp4"

        if os.path.exists(video_path):
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
            q_prompt = f"{question}? 0: {a0}; 1: {a1}; 2: {a2}; 3: {a3}; 4: {a4}"

            # Additional info
            frame_count = qa_entry['frame_count'].values[0]
            width = qa_entry['width'].values[0]
            height = qa_entry['height'].values[0]
            original_id = qa_entry['video'].values[0]    
            # Build the final data sample.
            sample = {
                "height": int(height),
                "width": int(width),
                "id": int(id),
                "original_id": int(original_id),
                "dataset_name": "NExT-QA",
                "num_frames": int(frame_count),
                "problem": str(q_prompt),
                "video": str(video_path),
                "data_source": "video",  # This is used to identify the dataset for reward model evaluation
                "reward_model": {
                  "ground_truth": str(answer)  # This is used for reward model evaluation
                },
                "extra_info": {
                    "type": mode,
                    'index': int(id),
                }
            }
            dataset.append(sample)
            id += 1
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="/shares/chenwei/VLM-R1")
    parser.add_argument("--parent_directory", type=str, default='/shares/chenwei/NExT-QA')
    parser.add_argument("--n", type=int, default=None)
  
    args = parser.parse_args()
    with open(os.path.join(args.parent_directory, "map_vid_vidorID.json"), "r") as f:
        vid_to_vidor = json.load(f)
    
    video_directory = os.path.join(args.parent_directory, "NExTVideo")
    
    if args.mode == "train":
        # Create train data.
        QA_file = os.path.join(args.parent_directory, "nextqa/train.csv")
        QA_data = pd.read_csv(QA_file)
        df = create_data(QA_data, vid_to_vidor, video_directory, args.mode, args.n)
    elif args.mode == "val":
        # Create val data.
        QA_file = os.path.join(args.parent_directory, "nextqa/val.csv")
        QA_data = pd.read_csv(QA_file)
        df = create_data(QA_data, vid_to_vidor, video_directory, args.mode, args.n)
    elif args.mode == "test":
        # Create test data.
        QA_file = os.path.join(args.parent_directory, "nextqa/test.csv")
        QA_data = pd.read_csv(QA_file)
        df = create_data(QA_data, vid_to_vidor, video_directory, args.mode, args.n)
    output_path = os.path.join(args.output_dir, args.mode)
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "nextqa.json"), "w") as f:
        json.dump(df, f, indent=4)