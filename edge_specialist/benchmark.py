#!/usr/bin/env python3
"""
Benchmark YOLO model inference across different backends.

Compares PyTorch (.pt), ONNX Runtime (.onnx), and TensorRT (.engine)
performance on the same input data. Reports FPS, latency percentiles,
and model size.

Designed to run on both the development machine (Phase 2 simulation)
and on Jetson Orin Nano (Phase 3 deployment).

Usage:
    # Benchmark all available backends
    python -m edge_specialist.benchmark \
        --pt-model ../yolov8n.pt \
        --onnx-model ../optimized_models/best_fp16_dynamic.onnx \
        --video ../pipeline_engineer/sample_data/test_thermal.mp4 \
        --frames 30 \
        --output results/benchmark.json

    # Benchmark only ONNX (Phase 2)
    python -m edge_specialist.benchmark \
        --onnx-model ../optimized_models/best_fp16_dynamic.onnx \
        --video ../pipeline_engineer/sample_data/test_thermal.mp4 \
        --frames 30

    # Jetson simulation mode (limits CPU threads)
    python -m edge_specialist.benchmark \
        --pt-model ../yolov8n.pt \
        --onnx-model ../optimized_models/best_fp16_dynamic.onnx \
        --video ../pipeline_engineer/sample_data/test_thermal.mp4 \
        --frames 30 --sim-jetson
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark model inference across backends")
    parser.add_argument("--pt-model", type=str, default="",
                        help="Path to PyTorch .pt model")
    parser.add_argument("--onnx-model", type=str, default="",
                        help="Path to ONNX .onnx model")
    parser.add_argument("--engine-model", type=str, default="",
                        help="Path to TensorRT .engine model")
    parser.add_argument("--video", type=str, required=True,
                        help="Video file to run inference on")
    parser.add_argument("--frames", type=int, default=30,
                        help="Number of frames to benchmark (0 = all)")
    parser.add_argument("--conf-thres", type=float, default=0.3,
                        help="Confidence threshold for detections")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference resolution (square)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup frames (not counted in benchmark)")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSON path for results")
    parser.add_argument("--sim-jetson", action="store_true",
                        help="Limit CPU threads to simulate Jetson constraints")
    return parser.parse_args()


def load_video_frames(video_path: str, max_frames: int = 0):
    """Load frames from a video file into a list."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames > 0 and len(frames) >= max_frames + 3:  # +warmup
            break

    cap.release()
    print(f"  Loaded {len(frames)} frames from {Path(video_path).name}")
    return frames


# ------------------------------------------------------------------
# Backend 1: PyTorch (via Ultralytics)
# ------------------------------------------------------------------
def benchmark_pytorch(model_path: str, frames, imgsz: int, conf_thres: float,
                      warmup: int, sim_jetson: bool) -> dict:
    from ultralytics import YOLO

    print(f"\n[benchmark] PyTorch backend: {model_path}")
    model = YOLO(model_path)

    # Simulate Jetson constraints
    if sim_jetson:
        try:
            import torch
            torch.set_num_threads(4)  # Simulate Jetson 6-core (reduced threads)
            print(f"  Simulating Jetson: torch threads=4")
        except ImportError:
            pass

    def predict(frame):
        results = model.predict(frame, conf=conf_thres, imgsz=imgsz, verbose=False)[0]
        return len(results.boxes)

    # Warmup
    for i in range(min(warmup, len(frames))):
        predict(frames[i])
    print(f"  Warmup: {warmup} frames")

    # Benchmark
    times = []
    detections_total = 0
    for i in range(len(frames)):
        t0 = time.time()
        n = predict(frames[i])
        dt = time.time() - t0
        times.append(dt)
        detections_total += n

    # Model size
    model_mb = Path(model_path).stat().st_size / (1024 * 1024)

    return {
        "backend": "PyTorch (.pt)",
        "model_path": model_path,
        "model_size_mb": round(model_mb, 2),
        "total_inference_ms": round(sum(times) * 1000, 1),
        "avg_inference_ms": round(np.mean(times) * 1000, 2),
        "p50_inference_ms": round(float(np.percentile(times, 50)) * 1000, 2),
        "p95_inference_ms": round(float(np.percentile(times, 95)) * 1000, 2),
        "p99_inference_ms": round(float(np.percentile(times, 99)) * 1000, 2),
        "fps": round(len(times) / sum(times), 1),
        "total_detections": detections_total,
    }


# ------------------------------------------------------------------
# Backend 2: ONNX Runtime
# ------------------------------------------------------------------
def benchmark_onnx(model_path: str, frames, imgsz: int, conf_thres: float,
                   warmup: int, sim_jetson: bool) -> dict:
    import onnxruntime as ort

    print(f"\n[benchmark] ONNX Runtime backend: {model_path}")

    # Validate ONNX model
    try:
        import onnx
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        input_meta = [(i.name, [d.dim_value if d.HasField("dim_value") else d.dim_param
                                 for d in i.type.tensor_type.shape.dim])
                      for i in onnx_model.graph.input]
        output_meta = [(o.name, [d.dim_value if d.HasField("dim_value") else d.dim_param
                                  for d in o.type.tensor_type.shape.dim])
                       for o in onnx_model.graph.output]
        print(f"  ONNX inputs: {input_meta}")
        print(f"  ONNX outputs: {output_meta}")
    except Exception as e:
        print(f"  WARNING: ONNX validation failed: {e}")

    # Choose execution provider
    ep = "CUDAExecutionProvider"
    try:
        sess = ort.InferenceSession(model_path, providers=[ep, "CPUExecutionProvider"])
    except Exception:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        ep = "CPU"

    # Simulate Jetson constraints
    if sim_jetson:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4  # Simulate Jetson CPU threads
        sess_options.inter_op_num_threads = 2
        sess = ort.InferenceSession(model_path, sess_options=sess_options,
                                    providers=["CPUExecutionProvider"])
        ep = "CPU (sim Jetson 4 threads)"
        print(f"  Simulating Jetson: ort threads=4")

    input_name = sess.get_inputs()[0].name
    print(f"  Execution provider: {ep}")

    def preprocess(frame):
        img = cv2.resize(frame, (imgsz, imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def predict(frame):
        inp = preprocess(frame)
        output = sess.run(None, {input_name: inp})[0]
        # YOLOv8 output: [1, 9, 8400] — raw, needs post-processing
        # For benchmarking we count non-trivial detections
        obj_conf = output[0, 4, :]
        return int(np.sum(obj_conf > conf_thres * 0.01))  # Count anchors with any confidence

    # Warmup
    for i in range(min(warmup, len(frames))):
        predict(frames[i])
    print(f"  Warmup: {warmup} frames")

    # Benchmark
    times = []
    detections_total = 0
    for i in range(len(frames)):
        t0 = time.time()
        n = predict(frames[i])
        dt = time.time() - t0
        times.append(dt)
        detections_total += n

    model_mb = Path(model_path).stat().st_size / (1024 * 1024)

    return {
        "backend": "ONNX Runtime",
        "execution_provider": ep,
        "model_path": model_path,
        "model_size_mb": round(model_mb, 2),
        "total_inference_ms": round(sum(times) * 1000, 1),
        "avg_inference_ms": round(np.mean(times) * 1000, 2),
        "p50_inference_ms": round(float(np.percentile(times, 50)) * 1000, 2),
        "p95_inference_ms": round(float(np.percentile(times, 95)) * 1000, 2),
        "p99_inference_ms": round(float(np.percentile(times, 99)) * 1000, 2),
        "fps": round(len(times) / sum(times), 1),
        "total_detections": detections_total,
    }


# ------------------------------------------------------------------
# Backend 3: TensorRT (Jetson only — stub for reference)
# ------------------------------------------------------------------
def benchmark_tensorrt(model_path: str, frames, imgsz: int, conf_thres: float,
                       warmup: int) -> dict:
    """TensorRT engine inference — only available on Jetson."""
    try:
        import tensorrt as trt
    except ImportError:
        return {
            "backend": "TensorRT (.engine)",
            "model_path": model_path,
            "error": "tensorrt package not available (run on Jetson)",
        }

    import pycuda.driver as cuda
    import pycuda.autoinit

    print(f"\n[benchmark] TensorRT backend: {model_path}")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    input_name = engine.get_binding_name(0)
    output_name = engine.get_binding_name(1)

    # Allocate buffers
    batch_size = engine.get_binding_shape(0)[0]
    input_shape = engine.get_binding_shape(0)
    output_shape = engine.get_binding_shape(1)

    d_input = cuda.mem_alloc(np.prod(input_shape) * 4)  # float32 = 4 bytes
    d_output = cuda.mem_alloc(np.prod(output_shape) * 4)
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()

    def preprocess(frame):
        img = cv2.resize(frame, (imgsz, imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    times = []
    detections_total = 0

    for i in range(len(frames)):
        inp = preprocess(frames[i])
        cuda.memcpy_htod_async(d_input, inp, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()

        t0 = time.time()
        obj_conf = output[0, 4, :]
        n = int(np.sum(obj_conf > conf_thres * 0.01))
        dt = time.time() - t0
        times.append(dt)
        detections_total += n

    model_mb = Path(model_path).stat().st_size / (1024 * 1024)

    return {
        "backend": "TensorRT (.engine)",
        "model_path": model_path,
        "model_size_mb": round(model_mb, 2),
        "total_inference_ms": round(sum(times) * 1000, 1),
        "avg_inference_ms": round(np.mean(times) * 1000, 2),
        "p50_inference_ms": round(float(np.percentile(times, 50)) * 1000, 2),
        "p95_inference_ms": round(float(np.percentile(times, 95)) * 1000, 2),
        "p99_inference_ms": round(float(np.percentile(times, 99)) * 1000, 2),
        "fps": round(len(times) / sum(times), 1),
        "total_detections": detections_total,
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("AI-Detection-System — Model Benchmark")
    print("=" * 60)

    # Load video frames
    print(f"\n[benchmark] Loading video: {args.video}")
    frames = load_video_frames(args.video, args.frames)
    if not frames:
        print("ERROR: No frames loaded")
        sys.exit(1)

    results = []

    # Benchmark PyTorch
    if args.pt_model and Path(args.pt_model).exists():
        results.append(benchmark_pytorch(
            args.pt_model, frames, args.imgsz, args.conf_thres,
            args.warmup, args.sim_jetson
        ))

    # Benchmark ONNX
    if args.onnx_model and Path(args.onnx_model).exists():
        results.append(benchmark_onnx(
            args.onnx_model, frames, args.imgsz, args.conf_thres,
            args.warmup, args.sim_jetson
        ))

    # Benchmark TensorRT (Jetson only)
    if args.engine_model and Path(args.engine_model).exists():
        results.append(benchmark_tensorrt(
            args.engine_model, frames, args.imgsz, args.conf_thres,
            args.warmup
        ))

    if not results:
        print("ERROR: No model files found to benchmark")
        sys.exit(1)

    # Print comparison table
    print(f"\n{'=' * 60}")
    print(f"Benchmark Results — {len(frames)} frames @ {args.imgsz}x{args.imgsz}")
    print(f"{'=' * 60}")
    print(f"\n{'Backend':<25} {'Size(MB)':<10} {'Avg(ms)':<10} {'FPS':<8} {'P95(ms)':<10}")
    print(f"{'-'*60}")
    for r in results:
        if "error" in r:
            print(f"{r['backend']:<25} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<10}")
            print(f"  Error: {r['error']}")
        else:
            print(f"{r['backend']:<25} {r['model_size_mb']:<10.1f} {r['avg_inference_ms']:<10.1f} "
                  f"{r['fps']:<8.1f} {r['p95_inference_ms']:<10.1f}")
    print(f"{'-'*60}\n")

    # Speedup analysis
    if len(results) >= 2 and "error" not in results[0] and "error" not in results[1]:
        pt_ms = results[0]["avg_inference_ms"]
        onnx_ms = results[1]["avg_inference_ms"]
        speedup = pt_ms / onnx_ms if onnx_ms > 0 else 0
        print(f"  ONNX speedup vs PyTorch: {speedup:.2f}x")

    # Save results
    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "video": args.video,
        "frames": len(frames),
        "imgsz": args.imgsz,
        "sim_jetson": args.sim_jetson,
        "conf_thres": args.conf_thres,
        "results": results,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Results saved to: {out_path}")
    else:
        default_out = PROJECT_ROOT / "pipeline_engineer" / "sample_data" / "benchmark_results.json"
        with open(default_out, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Results saved to: {default_out}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
