# -*- coding: utf-8 -*-
"""Inspect video metadata and properties."""

import sys
import argparse
from pathlib import Path
import cv2


def inspect_video(video_path: str):
    path = Path(video_path)
    if not path.is_file():
        print(f"ERROR: Video file not found at: {video_path}")
        return False

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"ERROR: Failed to open video: {video_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0.0

    ret, first_frame = cap.read()
    cap.release()

    print("=" * 60)
    print("VIDEO INSPECTION REPORT")
    print("=" * 60)
    print(f"File Path:       {path.resolve()}")
    print(f"File Size:       {path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"Resolution:      {width} x {height}")
    print(f"Framerate:       {fps:.2f} FPS")
    print(f"Total Frames:    {frame_count}")
    print(f"Duration:        {duration_sec:.2f} seconds")
    print(f"First Frame OK:  {ret and first_frame is not None}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect video file properties")
    parser.add_argument("video", nargs="?", default="videos/deer_test.mp4", help="Path to video file")
    args = parser.parse_args()
    inspect_video(args.video)
