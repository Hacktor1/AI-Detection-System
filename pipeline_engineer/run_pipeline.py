#!/usr/bin/env python3
"""
Main pipeline entry point.

Reads a video stream, runs object detection (YOLO), renders bounding boxes,
and displays/saves the result.

Usage:
    python run_pipeline.py --video data/sample.mp4 --model models/best.onnx
"""
import argparse
import os
import time

import cv2

from camera_io import open_video_source, read_frame
from detector import PersonDetector
from renderer import draw_detections


def parse_args():
    parser = argparse.ArgumentParser(description="Drone AI Detection Pipeline")
    parser.add_argument("--video", type=str, default="0",
                        help="0 = webcam, path = video file")
    parser.add_argument("--model", type=str, default="",
                        help="Path to model (.onnx / .engine / .pt)")
    parser.add_argument("--output", type=str, default="",
                        help="Optional output video file path")
    parser.add_argument("--conf-thres", type=float, default=0.4,
                        help="Confidence threshold for detections")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Camera/source ---
    cap = open_video_source(args.video)
    print(f"[pipeline] Source opened: {args.video}")

    # --- Detector ---
    detector = PersonDetector(model_path=args.model, conf_thres=args.conf_thres)
    print(f"[pipeline] Detector loaded: {args.model}")

    # --- Output writer (optional) ---
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = read_frame(cap)
        if not ret:
            print(f"[pipeline] End of stream at frame {frame_idx}")
            break

        # --- Detect ---
        t0 = time.time()
        boxes = detector.infer(frame)
        dt = time.time() - t0

        # --- Render ---
        vis = draw_detections(frame, boxes)

        # Show in window
        cv2.imshow("Drone AI — Thermal Detection", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if out is not None:
            out.write(vis)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"[pipeline] Frame {frame_idx} | inference: {dt:.3f}s | boxes: {len(boxes)}")

    # cleanup
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print("[pipeline] Finished.")


if __name__ == "__main__":
    main()
