#!/usr/bin/env python3
"""
Model profiler — měří FPS, latenci a GPU využití pro různé backends.

Použití:
    python3 profiler.py --onnx optimized_models/best_fp16_dynamic.onnx --frames 100
    python3 profiler.py --pt-models yolov8n.pt --onnx optimized_models/best_fp16_dynamic.onnx --frames 100
    python3 profiler.py --onnx optimized_models/best_fp16_dynamic.onnx --sim-jetson  # CPU thread limit
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Model performance profiler")
    parser.add_argument("--onnx", type=str, default="", help="ONNX model path")
    parser.add_argument("--pt-model", type=str, default="", help="PyTorch model path")
    parser.add_argument("--frames", type=int, default=100, help="Frames to benchmark")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup frames")
    parser.add_argument("--sim-jetson", action="store_true", help="Limit CPU threads (simulate Jetson)")
    parser.add_argument("--output", type=str, default="results/profile_results.json", help="Output JSON")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="Confidence threshold")
    return parser.parse_args()


def get_cpu_count():
    if os.environ.get("OMP_NUM_THREADS"):
        return int(os.environ["OMP_NUM_THREADS"])
    return os.cpu_count()


def limit_threads(jetson: bool):
    """Limit CPU threads to simulate Jetson constraints."""
    if jetson:
        n = 4  # Jetson Orin Nano has 8 cores, but use 4 for conservative sim
        os.environ["OMP_NUM_THREADS"] = str(n)
        os.environ["MKL_NUM_THREADS"] = str(n)
        try:
            import torch
            torch.set_num_threads(n)
        except ImportError:
            pass
        try:
            import onnxruntime as ort
            # Note: ort session options should be set per-session
        except ImportError:
            pass
        print(f"[profiler] Simulating Jetson: limiting to {n} CPU threads")
    else:
        print(f"[profiler] Available CPUs: {get_cpu_count()}")


def generate_test_frames(n: int, imgsz: int):
    """Generate synthetic test frames for benchmarking."""
    frames = []
    for i in range(n):
        # Random noise frame — tests pipeline speed without I/O bias
        frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def profile_pytorch(model_path, frames, imgsz, conf_thres, warmup, jetson):
    """Profile PyTorch model."""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    
    if jetson:
        import torch
        torch.set_num_threads(4)
    
    # Warmup
    for i in range(min(warmup, len(frames))):
        model.predict(frames[i], conf=conf_thres, imgsz=imgsz, verbose=False)
    
    # Benchmark
    times = []
    for frame in frames:
        t0 = time.perf_counter()
        model.predict(frame, conf=conf_thres, imgsz=imgsz, verbose=False)
        times.append(time.perf_counter() - t0)
    
    stats = np.array(times) * 1000  # ms
    
    return {
        "backend": "PyTorch",
        "model": str(model_path),
        "model_size_mb": round(Path(model_path).stat().st_size / (1024*1024), 1),
        "frames": len(frames),
        "avg_ms": round(float(np.mean(stats)), 2),
        "p50_ms": round(float(np.percentile(stats, 50)), 2),
        "p95_ms": round(float(np.percentile(stats, 95)), 2),
        "p99_ms": round(float(np.percentile(stats, 99)), 2),
        "fps": round(len(frames) / sum(t for t in times), 1),
        "sim_jetson": jetson,
        "cpu_count": get_cpu_count(),
    }


def profile_onnx(model_path, frames, imgsz, conf_thres, warmup, jetson):
    """Profile ONNX Runtime model."""
    import onnxruntime as ort
    
    # Setup session options
    sess_options = ort.SessionOptions()
    if jetson:
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2
    else:
        sess_options.intra_op_num_threads = get_cpu_count()
    
    # Try CUDA first, fall back to CPU
    try:
        sess = ort.InferenceSession(model_path, sess_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        ep = "CUDA"
    except:
        sess = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
        ep = "CPU"
    
    input_name = sess.get_inputs()[0].name
    
    # Preprocess single frame to determine output shape
    frame = frames[0]
    img = cv2.resize(frame, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    # Warmup
    for i in range(min(warmup, len(frames))):
        sess.run(None, {input_name: img})
    
    # Benchmark
    times = []
    for frame in frames:
        img = cv2.resize(frame, (imgsz, imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        t0 = time.perf_counter()
        sess.run(None, {input_name: img})
        times.append(time.perf_counter() - t0)
    
    stats = np.array(times) * 1000  # ms
    
    return {
        "backend": f"ONNX Runtime ({ep})",
        "model": str(model_path),
        "model_size_mb": round(Path(model_path).stat().st_size / (1024*1024), 1),
        "frames": len(frames),
        "avg_ms": round(float(np.mean(stats)), 2),
        "p50_ms": round(float(np.percentile(stats, 50)), 2),
        "p95_ms": round(float(np.percentile(stats, 95)), 2),
        "p99_ms": round(float(np.percentile(stats, 99)), 2),
        "fps": round(len(frames) / sum(t for t in times), 1),
        "sim_jetson": jetson,
        "cpu_count": get_cpu_count(),
    }


def main():
    args = parse_args()
    limit_threads(args.sim_jetson)
    
    # Generate test frames
    frames = generate_test_frames(args.frames, args.imgsz)
    print(f"[profiler] Generated {len(frames)} test frames at {args.imgsz}x{args.imgsz}")
    
    results = []
    
    # Profile PyTorch
    if args.pt_model and Path(args.pt_model).exists():
        print(f"\n[profiler] Profiling PyTorch: {args.pt_model}")
        r = profile_pytorch(args.pt_model, frames, args.imgsz, args.conf_thres,
                           args.warmup, args.sim_jetson)
        results.append(r)
        print(f"  -> {r['fps']} FPS, {r['avg_ms']} ms avg")
    
    # Profile ONNX
    if args.onnx and Path(args.onnx).exists():
        print(f"\n[profiler] Profiling ONNX: {args.onnx}")
        r = profile_onnx(args.onnx, frames, args.imgsz, args.conf_thres,
                        args.warmup, args.sim_jetson)
        results.append(r)
        print(f"  -> {r['fps']} FPS, {r['avg_ms']} ms avg")
    
    if not results:
        print("❌ Žádné modely k profilování")
        sys.exit(1)
    
    # Print comparison table
    print(f"\n{'='*60}")
    print(f"Profiler Results — {args.frames} synthetic frames @ {args.imgsz}x{args.imgsz}")
    print(f"{'='*60}")
    print(f"{'Backend':<20} {'Size(MB)':<10} {'Avg(ms)':<10} {'FPS':<8} {'P95(ms)':<10}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['backend']:<20} {r['model_size_mb']:<10.1f} {r['avg_ms']:<10.1f} "
              f"{r['fps']:<8.1f} {r['p95_ms']:<10.1f}")
    print(f"{'-'*60}")
    
    # Speedup
    if len(results) == 2:
        pt = results[0]
        onnx = results[1]
        if "error" not in pt and "error" not in onnx:
            speedup = pt["avg_ms"] / onnx["avg_ms"] if onnx["avg_ms"] > 0 else 0
            print(f"  ONNX speedup vs PyTorch: {speedup:.2f}x")
    
    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frames": args.frames,
        "imgsz": args.imgsz,
        "sim_jetson": args.sim_jetson,
        "results": results,
    }
    
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "edge_specialist" / "results" / "profile_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n📊 Results saved: {output_path}")


if __name__ == "__main__":
    main()
