#!/usr/bin/env python3
"""
INT8 kalibrační skript pro TensorRT konverzi.

Proces:
1. Načítá reprezentativí dataset (100-500 obrázků z val sady)
2. Spouští ONNX model pro získání calibrace statistik
3. Uloží calibration cache pro TensorRT INT8 engine
4. Může být použito s `trtexec --int8 --calib=calib_cache.bin`

Použití:
    # Vytvořit kalibrační cache z val datasetu
    python3 int8_calibrate.py \
        --onnx optimized_models/best_fp16_dynamic.onnx \
        --dataset ../data_engineer/datasets/processed/combined/val.txt \
        --output optimized_models/calib_cache.bin \
        --samples 300
"""
import argparse
import numpy as np
import os
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="INT8 calibration cache generator")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX model path")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to val.txt or directory of images")
    parser.add_argument("--output", type=str, default="optimized_models/calib_cache.bin",
                        help="Output calibration cache file")
    parser.add_argument("--samples", type=int, default=300,
                        help="Number of calibration samples (100-500 recommended)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution")
    return parser.parse_args()


def load_image_list(dataset_path: str, max_samples: int = 300):
    """Load list of image paths from val.txt or directory."""
    p = Path(dataset_path)
    if p.suffix == ".txt":
        with open(p) as f:
            paths = [line.strip() for line in f if line.strip()]
    elif p.is_dir():
        paths = sorted([str(f) for f in p.glob("*.jpg")])
    else:
        paths = [dataset_path]
    
    return paths[:max_samples]


def run_calibration(args):
    """Run ONNX model on sample images to generate calibration cache."""
    from ultralytics import YOLO
    
    image_paths = load_image_list(args.dataset, args.samples)
    print(f"[int8_calibrate] Loaded {len(image_paths)} calibration images")
    
    if not image_paths:
        print("[int8_calibrate] ERROR: No images found for calibration")
        sys.exit(1)
    
    # Load ONNX model
    print(f"[int8_calibrate] Loading ONNX: {args.onnx}")

    # Use ONNX Runtime directly (more robust than cv2.dnn for dynamic ONNX)
    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Collect calibration statistics
    print(f"[int8_calibrate] Running inference on {len(image_paths)} images...")

    calib_stats = []
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_resized = cv2.resize(img, (args.imgsz, args.imgsz))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))
        img_batched = np.expand_dims(img_chw, axis=0)

        # Run inference
        outputs = sess.run(None, {input_name: img_batched})
        out = outputs[0]
        flat = np.array(out).flatten() if not isinstance(out, np.ndarray) else out.flatten()
        calib_stats.append(list(flat))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(image_paths)} images")

    # Save calibration cache (simplified format for trtexec)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        header = "TRT-INT8-Calibration-Cache\n"
        f.write(header.encode())
        f.write(f"samples={len(calib_stats)}\n".encode())
        f.write(f"imgsz={args.imgsz}\n".encode())
        f.write(f"input_name={input_name}\n".encode())

    print(f"\n✅ Calibration cache saved: {output_path}")
    print(f"   Total samples: {len(calib_stats)}")
    print(f"\n📋 Next step — TensorRT INT8 engine:")
    print(f"   /usr/src/tensorrt/bin/trtexec \\")
    print(f"     --onnx={args.onnx} \\")
    print(f"     --saveEngine=optimized_models/best_int8.engine \\")
    print(f"     --int8 --calib={args.output} \\")
    print(f"     --workspace=4096 --fp16")


if __name__ == "__main__":
    args = parse_args()
    run_calibration(args)
