#!/usr/bin/env python3
"""
Generate a synthetic visible-light video from a thermal video file.

This simulates a dual-camera setup (thermal + visible) when only a
thermal video is available. The "visible" stream is derived by applying
a colour map and adding realistic visual noise/lighting variation.

Usage:
    python -m jetson_sim.make_sample_video \
        --thermal-input pipeline_engineer/sample_data/test_thermal.mp4 \
        --visible-output pipeline_engineer/sample_data/test_visible.mp4 \
        --frames 30

The generated video matches the thermal source's resolution and frame count.
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def generate_visible_frame(thermal_frame: np.ndarray, frame_idx: int) -> np.ndarray:
    """
    Transform a thermal (grayscale / BGR) frame into a synthetic
    visible-light frame by applying a colour map and simulating
    lighting variation, sensor noise, and motion blur.

    Thermal cameras typically show a monochrome or ironbow image.
    A visible camera would show RGB colours with more texture detail.
    """
    # Convert to grayscale if it's already BGR
    if len(thermal_frame.shape) == 3:
        gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = thermal_frame

    # Apply a colour map (JET or INFERNO) to simulate visible-light false colour
    # In a real scenario, the visible camera would show different contrast/colour
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Add synthetic lighting variation (sinusoidal brightness modulation)
    # Simulates drone movement / changing scene illumination
    height, width = colored.shape[:2]
    light_factor = 0.85 + 0.15 * np.sin(frame_idx * 0.3)  # 85%-100% brightness
    colored = np.clip(colored.astype(np.float32) * light_factor, 0, 255).astype(np.uint8)

    # Add subtle Gaussian noise to simulate sensor noise
    noise = np.random.normal(0, 8, colored.shape).astype(np.float32)
    colored = np.clip(colored.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Add a subtle vignette to simulate lens effect
    X_kernel = cv2.getGaussianKernel(width, width / 2).flatten()
    Y_kernel = cv2.getGaussianKernel(height, height / 2).flatten()
    kernel = np.outer(Y_kernel.astype(np.float32), X_kernel.astype(np.float32))
    vignettting = kernel / kernel.max()
    vignettting = vignettting * 0.2 + 0.8  # Range 0.8-1.0
    colored = np.clip(colored.astype(np.float32) * vignettting[:, :, np.newaxis], 0, 255).astype(np.uint8)

    # Add a simple reticle/overlay to distinguish from thermal visually
    cx, cy = width // 2, height // 2
    cv2.circle(colored, (cx, cy), 20, (0, 255, 255), 1)  # Yellow circle

    return colored


def generate_visible_video(thermal_path: str, output_path: str, max_frames: int = 0):
    """
    Read a thermal video and produce a synthetic visible-light video.

    The visible video has the same resolution, FPS, and frame count
    (up to max_frames) as the thermal source.
    """
    cap = cv2.VideoCapture(thermal_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open thermal video: {thermal_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        frame_count = min(frame_count, max_frames)

    print(f"[make_sample_video] Thermal source: {thermal_path}")
    print(f"  Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {frame_count}")
    print(f"[make_sample_video] Generating visible-light video: {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    generated = 0
    while generated < frame_count:
        ret, frame = cap.read()
        if not ret:
            print(f"  End of thermal stream at frame {generated}")
            break

        visible_frame = generate_visible_frame(frame, generated)
        out.write(visible_frame)
        generated += 1

        if generated % 10 == 0:
            print(f"  Generated {generated}/{frame_count} frames")

    cap.release()
    out.release()
    print(f"[make_sample_video] Done — {generated} frames written to {output_path}")

    # Verify output
    check = cv2.VideoCapture(output_path)
    if check.isOpened():
        print(f"  Output: {int(check.get(cv2.CAP_PROP_FRAME_COUNT))} frames, "
              f"{int(check.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    check.release()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic visible-light video from thermal")
    parser.add_argument("--thermal-input", type=str, required=True,
                        help="Path to thermal video file")
    parser.add_argument("--visible-output", type=str, required=True,
                        help="Path for output visible-light video")
    parser.add_argument("--frames", type=int, default=0,
                        help="Max frames to copy (0 = all)")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.visible_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    generate_visible_video(args.thermal_input, args.visible_output, args.frames)


if __name__ == "__main__":
    main()
