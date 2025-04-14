import cv2
import os

def test_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")

    # Test frames around the problematic area
    test_frames = list(range(385, 393))  # Test frames 385-392
    print("\nTesting frame accessibility:")
    print("Frame | Success | Position (ms)")
    
    for frame_idx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        print(f"{frame_idx:5d} | {str(ret):7s} | {position_ms:10.2f}")

    cap.release()

if __name__ == "__main__":
    video_path = "/shares/hlw3876/chenwei/NExT-QA/NExTVideo/1013/2401167740.mp4"
    test_video_frames(video_path) 