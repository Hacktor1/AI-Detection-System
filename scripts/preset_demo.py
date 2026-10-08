#!/usr/bin/env python3
"""
Presentation-ready YOLO inference demo.

Runs ONNX model on a test video and produces annotated results
suitable for a live demo/presentation to stakeholders.

Usage:
    python3 scripts/preset_demo.py --output results/sim_detections.json
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CLASS_NAMES = ["person", "car", "bicycle", "dog", "other_vehicle"]


def run_demo(
    video_path: str,
    model_path: str,
    output_dir: str = "results",
    max_frames: int = 60,
    conf_thres: float = 0.25,
):
    """Run YOLO ONNX inference on every N-th frame and collect results."""
    output_dir = str(PROJECT_ROOT / output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load ONNX
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"[demo] Model loaded: {model_path} ({Path(model_path).stat().st_size // 1024}KB)")

    # Open video
    cap = cv2.VideoCapture(str(PROJECT_ROOT / video_path))
    if not cap.isOpened():
        print(f"[demo] ERROR: Cannot open {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[demo] Video: {total_frames} frames @ {fps:.1f} FPS")

    # Sample every 5th frame for demo
    step = max(1, total_frames // max_frames)
    results = []

    frame_idx = 0
    processed = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img_norm = img.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

        t0 = datetime.now()
        outputs = sess.run(None, {input_name: img_chw})
        latency_ms = (datetime.now() - t0).total_seconds() * 1000

        # YOLO output: [1, 9, 8400] (YOLOv8n format)
        # Post-process: filter by confidence
        pred = outputs[0][0]  # [9, 8400]
        boxes = []

        # YOLOv8n output layout: each box is [x, y, w, h, obj_conf_0, cls0, cls1, cls2, cls3]
        # 9 channels: 4 bbox + 1 obj + 4 classes
        for i in range(pred.shape[1]):
            x, y, w, h = pred[0, i], pred[1, i], pred[2, i], pred[3, i]
            obj_conf = pred[4, i]
            class_scores = pred[5:, i]  # 4 class scores

            best_cls = int(np.argmax(class_scores))
            best_score = float(class_scores[best_cls]) * float(obj_conf)

            if best_score > conf_thres:
                boxes.append({
                    "class_id": best_cls,
                    "class_name": CLASS_NAMES[best_cls] if best_cls < len(CLASS_NAMES) else "unknown",
                    "confidence": round(best_score, 4),
                    "bbox": [
                        round(float(x), 2),
                        round(float(y), 2),
                        round(float(w), 2),
                        round(float(h), 2),
                    ],
                })

        results.append({
            "frame": frame_idx,
            "timestamp": round(frame_idx / fps, 3),
            "inference_ms": round(latency_ms, 2),
            "detections": boxes,
        })

        processed += 1
        if boxes:
            print(f"  Frame {frame_idx}: {len(boxes)} detekcí ({boxes[0]['class_name']} @ {boxes[0]['confidence']:.2f})")
        elif processed % 10 == 0:
            print(f"  Frame {frame_idx}: žádné detekce (ok)")

        frame_idx += step

    cap.release()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"sim_detections_{timestamp}.json")
    summary_file = os.path.join(output_dir, f"sim_summary_{timestamp}.json")
    output_json = os.path.join(output_dir, "sim_detections_latest.json")
    summary_json = os.path.join(output_dir, "sim_summary_latest.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    total_dets = sum(len(r["detections"]) for r in results)
    class_counts = {}
    for r in results:
        for d in r["detections"]:
            name = d["class_name"]
            class_counts[name] = class_counts.get(name, 0) + 1

    avg_latency = sum(r["inference_ms"] for r in results) / len(results) if results else 0

    summary = {
        "timestamp": timestamp,
        "video": str(video_path),
        "model": str(model_path),
        "total_frames_processed": len(results),
        "total_detections": total_dets,
        "detections_per_class": class_counts,
        "avg_inference_ms": round(avg_latency, 2),
        "avg_fps": round(1000 / avg_latency, 1) if avg_latency > 0 else 0,
        "confidence_threshold": conf_thres,
        "model_size_kb": os.path.getsize(model_path) // 1024,
        "output_file": output_file,
        "sample_results": results[:3],
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Copy latest
    import shutil
    shutil.copy(output_file, output_json)
    shutil.copy(summary_file, summary_json)

    print(f"\n[DEMO RESULTS]")
    print(f"  Frames processed: {len(results)}")
    print(f"  Total detections: {total_dets}")
    print(f"  Per class: {class_counts}")
    print(f"  Avg latency: {avg_latency:.2f}ms")
    print(f"  Avg FPS: {summary['avg_fps']:.1f}")
    print(f"  Results saved: {output_file}")
    print(f"  Summary saved: {summary_file}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO ONNX presentation demo")
    parser.add_argument("--video", default="pipeline_engineer/sample_data/test_thermal.mp4",
                        help="Test video path")
    parser.add_argument("--model", default="edge_specialist/optimized_models/best_fp16_dynamic.onnx",
                        help="ONNX model path")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=60, help="Max frames to process")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    run_demo(
        video_path=args.video,
        model_path=args.model,
        output_dir=args.output,
        max_frames=args.max_frames,
        conf_thres=args.conf_thres,
    )
