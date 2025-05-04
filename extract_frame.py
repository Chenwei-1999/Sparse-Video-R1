import cv2
import os
from PIL import Image

# ===== 参数设置 =====
VIDEO_PATH = "/home/zye52/scr4_hlee283/zye52/NExT-QA/NExTVideo/1103/8557532213.mp4"
ROUND_TIMESTAMPS = {
    1: [0, 4,10],
    2: [4,5,10],
    3: [2,3,4,5]
}
HEIGHT = 512
WIDTH = 512
SAVE_DIR = "/home/zye52/scr4_hlee283/zye52/extract_frame/extracted_by_round_450"

def extract_frames(video_path, round_timestamps, save_root):
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    os.makedirs(save_root, exist_ok=True)

    for round_id, timestamps in round_timestamps.items():
        round_dir = os.path.join(save_root, f"round_{round_id}")
        os.makedirs(round_dir, exist_ok=True)

        print(f"⏳ Extracting Round {round_id}...")

        for i, ts in enumerate(timestamps):
            if ts < 0 or ts > duration:
                print(f"⏭️ Skipping invalid timestamp {ts}s (video duration = {duration:.2f}s)")
                continue

            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            success, frame = cap.read()
            if not success:
                print(f"⚠️ Failed to read frame at {ts:.1f}s")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb).resize((WIDTH, HEIGHT))
            filename = f"frame_{i}_t{ts:.1f}.png"
            pil_img.save(os.path.join(round_dir, filename))

        print(f"✅ Saved {len(timestamps)} frames to {round_dir}")

    cap.release()

if __name__ == "__main__":
    extract_frames(VIDEO_PATH, ROUND_TIMESTAMPS, SAVE_DIR)
