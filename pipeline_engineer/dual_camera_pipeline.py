#!/usr/bin/env python3
"""
Dual-camera detection pipeline for UAV thermal + visible-light streams.

Reads from two video sources simultaneously (thermal + visible), runs
YOLO detection on each stream, renders a side-by-side output with
bounding boxes, and benchmarks FPS / latency.

Supports:
  - Video files (for simulation / Phase 2)
  - Live camera devices (for deployment on Jetson / Phase 3)
  - ONNX, PyTorch, and TensorRT (.engine) model backends
  - Jetson hardware simulation mode (--sim-jetson)

Usage:
    # Phase 2 simulation (video files + ONNX model)
    python dual_camera_pipeline.py \
        --thermal-source sample_data/test_thermal.mp4 \
        --visible-source sample_data/test_visible.mp4 \
        --model ../optimized_models/best_fp16_dynamic.onnx \
        --output results/dual_cam_sim.mp4

    # Headless mode (no display window)
    python dual_camera_pipeline.py \
        --thermal-source sample_data/test_thermal.mp4 \
        --visible-source sample_data/test_visible.mp4 \
        --model ../yolov8n.pt \
        --no-display --output results/dual_cam_sim.mp4

    # Phase 3 deployment (CSI cameras + TensorRT engine)
    python dual_camera_pipeline.py \
        --thermal-source /dev/video0 \
        --visible-source /dev/video1 \
        --model optimized_models/best_fp16_dynamic.engine \
        --no-display --output-dir results/

    # Jetson simulation mode (limits CPU threads, simulates thermal throttling)
    python dual_camera_pipeline.py \
        --thermal-source sample_data/test_thermal.mp4 \
        --visible-source sample_data/test_visible.mp4 \
        --model ../yolov8n.pt \
        --sim-jetson --no-display
"""
import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_engineer.camera_io import open_video_source
from pipeline_engineer.detector import PersonDetector
from pipeline_engineer.renderer import draw_detections
from jetson_sim.hardware_profile import JetsonProfile


SAMPLE_DATA_DIR = Path(__file__).resolve().parent / "sample_data"


@dataclass
class FrameResult:
    """Holds a captured frame and its index."""
    frame: np.ndarray
    frame_idx: int
    timestamp: float


@dataclass
class BenchmarkStats:
    """Aggregated benchmark statistics."""
    total_frames: int = 0
    total_time: float = 0.0
    inference_times: list = field(default_factory=list)
    capture_times: list = field(default_factory=list)
    render_times: list = field(default_factory=list)
    frame_intervals: list = field(default_factory=list)

    @property
    def avg_fps(self) -> float:
        if self.total_time <= 0:
            return 0.0
        return self.total_frames / self.total_time

    @property
    def avg_inference_ms(self) -> float:
        return np.mean(self.inference_times) * 1000 if self.inference_times else 0.0

    @property
    def avg_capture_ms(self) -> float:
        return np.mean(self.capture_times) * 1000 if self.capture_times else 0.0

    @property
    def avg_render_ms(self) -> float:
        return np.mean(self.render_times) * 1000 if self.render_times else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.inference_times:
            return 0.0
        return float(np.percentile(self.inference_times, 95)) * 1000

    def to_dict(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "total_time_s": round(self.total_time, 3),
            "avg_fps": round(self.avg_fps, 2),
            "avg_inference_ms": round(self.avg_inference_ms, 2),
            "avg_capture_ms": round(self.avg_capture_ms, 2),
            "avg_render_ms": round(self.avg_render_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p50_inference_ms": round(float(np.percentile(self.inference_times, 50)) * 1000, 2) if self.inference_times else 0,
            "p99_inference_ms": round(float(np.percentile(self.inference_times, 99)) * 1000, 2) if self.inference_times else 0,
        }


class CameraReader(threading.Thread):
    """Thread that reads frames from a video source into a bounded queue."""

    def __init__(self, source: str, name: str = "camera", max_queue: int = 4):
        super().__init__(daemon=True)
        self.source = source
        self.name = name
        self.max_queue = max_queue
        self._frames: list = []
        self._lock = threading.Lock()
        self._running = False
        self._cap = None
        self.error = None

    def run(self):
        try:
            self._cap = open_video_source(self.source)
        except Exception as e:
            self.error = e
            return

        self._running = True
        idx = 0
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                break
            with self._lock:
                if len(self._frames) >= self.max_queue:
                    self._frames.pop(0)  # Drop oldest frame
                self._frames.append(FrameResult(frame, idx, time.time()))
            idx += 1

        if self._cap is not None:
            self._cap.release()

    def get_frame(self, timeout: float = 0.5) -> FrameResult | None:
        """Pop the oldest frame from the queue. Returns None if empty."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frames:
                    return self._frames.pop(0)
            time.sleep(0.001)
        return None

    def stop(self):
        self._running = False

    def release(self):
        self.stop()
        if self._cap is not None:
            self._cap.release()


def create_dual_detector(model_path: str, conf_thres: float = 0.4) -> "DualDetector":
    """Create a dual detector (same model instance, two inference slots)."""
    return DualDetector(model_path, conf_thres)


class DualDetector:
    """
    Wraps two PersonDetector instances sharing the same model path.

    On Jetson (Phase 3) each stream can use its own TensorRT engine context.
    In simulation (Phase 2) both use ONNX Runtime / PyTorch via Ultralytics.
    """

    def __init__(self, model_path: str, conf_thres: float = 0.4):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.thermal_detector = PersonDetector(model_path, conf_thres)
        self.visible_detector = PersonDetector(model_path, conf_thres)

    def infer(self, thermal_frame, visible_frame):
        """Run detection on both frames. Returns (thermal_boxes, visible_boxes)."""
        thermal_boxes = self.thermal_detector.infer(thermal_frame)
        visible_boxes = self.visible_detector.infer(visible_frame)
        return thermal_boxes, visible_boxes


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-Camera AI Detection Pipeline")
    parser.add_argument("--thermal-source", type=str, default="0",
                        help="Thermal camera/video source (0=/dev/video0, path=mp4 file)")
    parser.add_argument("--visible-source", type=str, default="1",
                        help="Visible camera/video source (1=/dev/video1, path=mp4 file)")
    parser.add_argument("--model", type=str, default="",
                        help="Path to model (.pt / .onnx / .engine)")
    parser.add_argument("--conf-thres", type=float, default=0.4,
                        help="Confidence threshold for detections")
    parser.add_argument("--output", type=str, default="",
                        help="Output video file path")
    parser.add_argument("--output-dir", type=str, default="results/",
                        help="Directory for output video + benchmark JSON")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (no GUI window)")
    parser.add_argument("--sim-jetson", action="store_true",
                        help="Simulate Jetson Orin Nano constraints (CPU threads, thermal throttle)")
    parser.add_argument("--save-frames", type=str, default="",
                        help="Directory to save annotated frames")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames to process (0 = all)")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Jetson simulation setup ---
    profile = JetsonProfile() if args.sim_jetson else None
    if profile:
        print(f"[pipeline] Jetson simulation mode: {profile.name}")
        print(f"  {profile.summary()}")
        profile.apply_cpu_affinity()
        profile.start_inference_timer()

    # --- Model / Detector ---
    model_path = args.model
    if not model_path:
        # Default to ONNX in optimized_models, fall back to yolov8n.pt
        onnx_path = PROJECT_ROOT / "optimized_models" / "best_fp16_dynamic.onnx"
        pt_path = PROJECT_ROOT / "yolov8n.pt"
        if onnx_path.exists():
            model_path = str(onnx_path)
        elif pt_path.exists():
            model_path = str(pt_path)
        else:
            model_path = ""
            print("[pipeline] Warning: No model found. Running detection stub.")

    print(f"[pipeline] Loading model: {model_path}")
    detector = DualDetector(model_path, conf_thres=args.conf_thres)
    print(f"[pipeline] Detectors initialized (thermal + visible)")

    # --- Camera readers (threads) ---
    thermal_source = args.thermal_source
    visible_source = args.visible_source

    # If using sample_data defaults
    if thermal_source == "0" and (SAMPLE_DATA_DIR / "test_thermal.mp4").exists():
        thermal_source = str(SAMPLE_DATA_DIR / "test_thermal.mp4")
    if visible_source == "1" and (SAMPLE_DATA_DIR / "test_visible.mp4").exists():
        visible_source = str(SAMPLE_DATA_DIR / "test_visible.mp4")

    print(f"[pipeline] Thermal source: {thermal_source}")
    print(f"[pipeline] Visible source: {visible_source}")

    thermal_reader = CameraReader(thermal_source, name="thermal", max_queue=30)
    visible_reader = CameraReader(visible_source, name="visible", max_queue=30)
    thermal_reader.start()
    visible_reader.start()

    # Check for camera errors
    time.sleep(0.5)
    if thermal_reader.error:
        print(f"[pipeline] ERROR opening thermal source: {thermal_reader.error}")
        sys.exit(1)
    if visible_reader.error:
        print(f"[pipeline] ERROR opening visible source: {visible_reader.error}")
        sys.exit(1)

    # --- Output setup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_writer = None
    if args.output:
        output_path = Path(args.output)
        output_dir_for_output = output_path.parent
        output_dir_for_output.mkdir(parents=True, exist_ok=True)
    else:
        # Generate default output path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if args.sim_jetson:
            tag = "jetson_sim"
        else:
            tag = "pc_test"
        output_path = output_dir / f"dual_cam_{tag}_{timestamp}.mp4"

    # Check if both streams are alive
    thermal_w, thermal_h = 0, 0
    visible_w, visible_h = 0, 0

    # Peek at first frames to get dimensions
    thermal_frame0 = thermal_reader.get_frame(timeout=2.0)
    visible_frame0 = visible_reader.get_frame(timeout=2.0)

    if thermal_frame0 is None or thermal_frame0.frame is None:
        print("[pipeline] ERROR: No frames from thermal source")
        thermal_reader.release()
        visible_reader.release()
        sys.exit(1)
    if visible_frame0 is None or visible_frame0.frame is None:
        print("[pipeline] ERROR: No frames from visible source")
        thermal_reader.release()
        visible_reader.release()
        sys.exit(1)

    thermal_h, thermal_w = thermal_frame0.frame.shape[:2]
    visible_h, visible_w = visible_frame0.frame.shape[:2]
    print(f"[pipeline] Thermal: {thermal_w}x{thermal_h}, Visible: {visible_w}x{visible_h}")

    # Canvas: side-by-side (thermal left, visible right)
    canvas_w = thermal_w + visible_w
    canvas_h = max(thermal_h, visible_h)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(output_path), fourcc, 5.0, (canvas_w, canvas_h))

    # Save annotated frames if requested
    save_frames_dir = None
    if args.save_frames:
        save_frames_dir = Path(args.save_frames)
        save_frames_dir.mkdir(parents=True, exist_ok=True)

    # --- Main processing loop ---
    stats = BenchmarkStats()
    frame_idx = 0
    last_print = 0
    sim_start = time.time()

    print(f"[pipeline] Starting dual-camera processing...")

    # Put the peeked frames back into processing
    thermal_queue = [thermal_frame0]
    visible_queue = [visible_frame0]

    while True:
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        # Get frames
        t_cap_start = time.time()
        thermal = thermal_queue.pop(0) if thermal_queue else thermal_reader.get_frame(timeout=0.5)
        visible = visible_queue.pop(0) if visible_queue else visible_reader.get_frame(timeout=0.5)

        if thermal is None or visible is None:
            print(f"[pipeline] End of stream at frame {frame_idx}")
            break

        thermal_frame = thermal.frame
        visible_frame = visible.frame
        cap_dt = time.time() - t_cap_start

        # --- Inference ---
        t_inf_start = time.time()
        thermal_boxes, visible_boxes = detector.infer(thermal_frame, visible_frame)
        infer_dt = time.time() - t_inf_start

        # Apply thermal throttle factor to reported latency
        if profile:
            throttle = profile.thermal_factor()
            # Simulate throttled inference (add synthetic delay)
            if throttle > 1.0 and infer_dt > 0:
                time.sleep(infer_dt * (throttle - 1.0))

        # --- Render ---
        t_render_start = time.time()
        thermal_vis = draw_detections(thermal_frame, thermal_boxes)
        visible_vis = draw_detections(visible_frame, visible_boxes)

        # Side-by-side canvas
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:thermal_h, :thermal_w] = thermal_vis
        canvas[:visible_h, thermal_w:thermal_w + visible_w] = visible_vis

        # Labels
        cv2.putText(canvas, "Thermal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(canvas, "Visible", (thermal_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # FPS & latency overlay
        elapsed = time.time() - sim_start
        current_fps = stats.total_frames / elapsed if elapsed > 0 else 0.0
        cv2.putText(canvas, f"FPS: {current_fps:.1f}", (10, canvas_h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, f"Inference: {infer_dt*1000:.1f}ms", (10, canvas_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        render_dt = time.time() - t_render_start

        # Write output
        if out_writer is not None:
            out_writer.write(canvas)

        # Save frame
        if save_frames_dir is not None:
            cv2.imwrite(str(save_frames_dir / f"frame_{frame_idx:04d}.jpg"), canvas)

        # Display
        if not args.no_display:
            cv2.imshow("Drone AI — Dual Camera Detection", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Benchmark tracking
        stats.total_frames += 1
        stats.inference_times.append(infer_dt)
        stats.capture_times.append(cap_dt)
        stats.render_times.append(render_dt)
        now = time.time()
        if last_print == 0:
            last_print = now
        if now - last_print >= 2.0:
            elapsed = time.time() - sim_start
            print(f"[pipeline] Frame {frame_idx} | FPS: {stats.total_frames / elapsed if elapsed > 0 else 0:.1f} | "
                  f"inf: {infer_dt*1000:.1f}ms | "
                  f"thermal_boxes: {len(thermal_boxes)} | visible_boxes: {len(visible_boxes)}")
            last_print = now

        frame_idx += 1

    stats.total_time += time.time() - sim_start

    # --- Cleanup ---
    thermal_reader.release()
    visible_reader.release()
    if out_writer is not None:
        out_writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    # --- Save benchmark results ---
    benchmark_data = stats.to_dict()
    benchmark_data["model"] = model_path
    benchmark_data["thermal_source"] = thermal_source
    benchmark_data["visible_source"] = visible_source
    benchmark_data["sim_jetson"] = args.sim_jetson
    if profile:
        benchmark_data["jetson_profile"] = profile.name

    bench_path = output_dir / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(bench_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[pipeline] DUAL CAMERA PIPELINE — RESULTS")
    print(f"{'='*60}")
    print(f"  Frames processed:  {stats.total_frames}")
    print(f"  Total time:        {stats.total_time:.2f}s")
    print(f"  Average FPS:       {stats.avg_fps:.1f}")
    print(f"  Avg inference:     {stats.avg_inference_ms:.1f}ms")
    print(f"  Avg capture:       {stats.avg_capture_ms:.1f}ms")
    print(f"  Avg render:        {stats.avg_render_ms:.1f}ms")
    print(f"  P50 inference:     {benchmark_data['p50_inference_ms']:.1f}ms")
    print(f"  P95 latency:       {stats.p95_latency_ms:.1f}ms")
    print(f"  P99 inference:     {benchmark_data['p99_inference_ms']:.1f}ms")
    if args.sim_jetson and profile:
        target_fps = profile.sim.target_fps
        target_lat = profile.sim.target_latency_ms
        print(f"\n  Jetson target:      {target_fps} FPS, {target_lat}ms latency")
        if stats.avg_fps >= target_fps * 0.8:
            print(f"  [PASS] FPS meets target (≥{target_fps * 0.8:.0f})")
        else:
            print(f"  [WARN] FPS below target (need ≥{target_fps * 0.8:.0f}, got {stats.avg_fps:.1f})")
        if stats.avg_inference_ms <= target_lat * 1.2:
            print(f"  [PASS] Latency meets target (≤{target_lat * 1.2:.0f}ms)")
        else:
            print(f"  [WARN] Latency above target (need ≤{target_lat * 1.2:.0f}ms, got {stats.avg_inference_ms:.1f}ms)")
    print(f"\n  Output video:      {output_path}")
    print(f"  Benchmark JSON:    {bench_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
