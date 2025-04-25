import json
import os
import cv2
import argparse

def get_video_metadata(video_path):
    if not os.path.exists(video_path):
        return None, None, None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return height, width, frame_count


def build_nextqa_from_subset_with_video_dims(
    raw_questions_path,
    subset_answer_path,
    video_base_dir,
    mode,
    output_dir
):
    with open(raw_questions_path, "r") as f:
        questions = json.load(f)

    with open(subset_answer_path, "r") as f:
        subset_answer = json.load(f)

    dataset = []
    valid_id = 0

    for q in questions:
        q_uid = q.get("q_uid")
        if q_uid not in subset_answer:
            continue

        gt = subset_answer[q_uid]

        question = q["question"]
        options = [q.get(f"option {i}", "") for i in range(5)]
        problem = f"{question}? " + "; ".join([f"{i}: {opt}" for i, opt in enumerate(options)])

        video_path = os.path.join(video_base_dir, f"{q_uid}.mp4")
        height, width, frame_count = get_video_metadata(video_path)

        if height is None:
            print(f"⚠️ Skipping: Cannot read video {video_path}")
            continue

        sample = {
            "height": height,
            "width": width,
            "id": valid_id,
            "original_id": q_uid,
            "dataset_name": "EgoSchema",
            "num_frames": frame_count,
            "problem": problem,
            "video": video_path,
            "data_source": "video",
            "reward_model": {
                "ground_truth": str(gt)
            },
            "extra_info": {
                "type": mode,
                "index": valid_id
            }
        }
        dataset.append(sample)
        valid_id += 1

    output_folder = os.path.join(output_dir, mode)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "egoschema.json")

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f" Saved {len(dataset)} valid samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_questions", type=str, default="/home/zye52/scr4_hlee283/zye52/EgoSchema/questions.json")
    parser.add_argument("--subset_answer", type=str, default="/home/zye52/scr4_hlee283/zye52/EgoSchema/subset_answers.json")
    parser.add_argument("--video_dir", type=str, default="/home/zye52/scr4_hlee283/zye52/EgoSchema/videos/videos")
    parser.add_argument("--mode", type=str, default="val")
    parser.add_argument("--output_dir", type=str, default="/home/zye52/scr4_hlee283/zye52/EgoSchema-processed-data")

    args = parser.parse_args()

    build_nextqa_from_subset_with_video_dims(
        raw_questions_path=args.raw_questions,
        subset_answer_path=args.subset_answer,
        video_base_dir=args.video_dir,
        mode=args.mode,
        output_dir=args.output_dir
    )
