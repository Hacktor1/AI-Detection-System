#!/usr/bin/env python3
"""
Camera I/O module for reading video streams (thermal cameras).

For development/testing: reads from MP4 files.
Later: can be extended for Jetson CSI/MIPI camera input.
"""
import argparse
import os

import cv2


def open_video_source(source: str = "0") -> cv2.VideoCapture:
    """Open a video source which can be a file path or camera index."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Video file not found: {source}")
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    return cap


def read_frame(cap: cv2.VideoCapture):
    """Read one frame. Returns (ret, frame_bgr) — frame is None on failure."""
    ret, frame = cap.read()
    return ret, frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test camera/video input")
    parser.add_argument("--source", type=str, default="0",
                        help="0 = webcam, path = video file")
    args = parser.parse_args()

    cap = open_video_source(args.source)
    idx = 0
    while True:
        ret, frame = read_frame(cap)
        if not ret:
            print(f"End of stream ({idx} frames read)")
            break
        idx += 1
        if idx % 10 == 0:
            print(f"Frame {idx}")
    cap.release()
    print("Done.")
