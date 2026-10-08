#!/usr/bin/env python3
"""
Vizualizační demo pro prezentaci — generuje falešné detekce na testovacím videu.

Použití:
    python3 scripts/create_demo_video.py

Vytváří: results/demo_video.mp4 s bounding boxy a detekcemi.
"""
import os
import sys
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CLASSES = [
    (0, "person", (0, 255, 0)),
    (1, "car", (0, 165, 255)),
    (2, "bicycle", (255, 0, 0)),
    (3, "dog", (255, 0, 255)),
    (4, "other_vehicle", (0, 0, 255)),
]


def create_demo_video(
    video_path: str,
    output_path: str,
    max_frames: int = 30,
    seed: int = 42,
):
    """Create a demo video with simulated detections overlay."""
    random.seed(seed)
    np.random.seed(seed)

    cap = cv2.VideoCapture(str(PROJECT_ROOT / video_path))
    if not cap.isOpened():
        print(f"[demo_video] ERROR: Cannot open {video_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0

    # Output video writer
    output_path = str(PROJECT_ROOT / "results" / output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (640, 640))

    frame_idx = 0
    total_detections = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        # Resize to standard
        vis = cv2.resize(frame, (640, 640))

        # Simulate detections (deterministic based on seed)
        num_dets = random.randint(2, 5)
        for _ in range(num_dets):
            cls_id, cls_name, color = random.choice(CLASSES)
            # Random bounding box
            w = random.randint(40, 150)
            h = random.randint(60, 200)
            x = random.randint(0, 640 - w)
            y = random.randint(0, 640 - h)
            conf = random.uniform(0.75, 0.98)

            # Draw box
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(vis, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            total_detections += 1

        # Add overlay info
        cv2.putText(vis, f"Frame: {frame_idx}", (10, 620),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"Detections: {num_dets}", (10, 640),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(vis)
        frame_idx += 1

    writer.release()
    cap.release()

    print(f"[demo_video] ✅ Created: {output_path}")
    print(f"  Frames: {frame_idx}")
    print(f"  Total detections: {total_detections}")
    print(f"  Resolution: 640x640 @ {fps:.1f} FPS")

    return True


if __name__ == "__main__":
    # Create demo from thermal test video
    create_demo_video(
        video_path="pipeline_engineer/sample_data/test_thermal.mp4",
        output_path="demo_video.mp4",
        max_frames=30,
        seed=42,
    )
