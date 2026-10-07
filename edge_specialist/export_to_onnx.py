#!/usr/bin/env python3
"""
Export natrénovaného YOLO modelu do ONNX formátu pro TensorRT inference na Jetsonu.

Výstupní formáty:
  - ONNX (univerzální, podporován všemi runtime)
  - Dynamické rozměry (pro různé velikosti snímků na dronu)

Použití:
    python -m edge_specialist.export_to_onnx --weights ../ai_ml_architect/experiments/uav_thermal_v1/weights/best.pt
    python -m edge_specialist.export_to_onnx --weights ../ai_ml_architect/experiments/uav_thermal_v1/weights/best.pt --half --dynamic

Argumenty:
    --weights    Cesta k .pt souboru (povinné)
    --output     Výstupní adresář (default: optimized_models/)
    --half       Použít FP16 kvantizaci pro menší model a rychlejší inference
    --dynamic    Podpora dynamických rozměrů (šířka/výška)
    --simplify   Použít onnx-simplifier pro optimalizaci
    --opset      ONNX opset version (default: 14)
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX for Jetson/TensorRT")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt model weights")
    parser.add_argument("--output", type=str, default="optimized_models/", help="Output directory")
    parser.add_argument("--half", action="store_true", help="Use FP16 quantization")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input dimensions")
    parser.add_argument("--simplify", action="store_true", default=True, help="Run onnx-simplifier")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    return parser.parse_args()


def export_to_onnx(args):
    """Export PyTorch model to ONNX format."""
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"❌ Weights not found: {weights_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Načítám model: {weights_path}")
    model = YOLO(str(weights_path))

    # Build output filename
    onnx_name = weights_path.stem
    if args.half:
        onnx_name += "_fp16"
    if args.dynamic:
        onnx_name += "_dynamic"
    onnx_path = output_dir / f"{onnx_name}.onnx"

    print(f"📦 Exportuji do ONNX: {onnx_path}")
    print(f"   FP16: {args.half}")
    print(f"   Dynamic: {args.dynamic}")
    print(f"   Simplify: {args.simplify}")
    print(f"   Opset: {args.opset}")

    try:
        export_result = model.export(
            format="onnx",
            imgsz=640,
            half=args.half,
            dynamic=args.dynamic,
            simplify=args.simplify,
            opset=args.opset,
        )

        # Ultralytics saves ONNX alongside the .pt in the weights directory
        # Copy the exported ONNX file to our output directory
        weights_dir = weights_path.parent
        onnx_source = list(weights_dir.glob("*.onnx"))
        if onnx_source:
            import shutil
            shutil.copy2(onnx_source[-1], onnx_path)  # Use the last one (FP16 dynamic)
            print(f"   Copied: {onnx_source[-1]} -> {onnx_path}")
        elif not onnx_path.exists():
            print(f"   ONNX file není v {onnx_path}")
            print(f"   Hledám v: {weights_dir}")
            return False

        # Print model info
        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            print(f"\n✅ ONNX export dokončen!")
            print(f"   Velikost: {size_mb:.1f} MB")
            print(f"   Cesta: {onnx_path}")

            # Print next steps for TensorRT conversion
            print(f"\n📋 Další kroky pro TensorRT:")
            engine_path = output_dir / f"{onnx_name}.engine"
            print(f"   /usr/src/tensorrt/bin/trtexec --onnx={onnx_path} --saveEngine={engine_path} --fp16 --workspace=2048")

    except Exception as e:
        print(f"❌ ONNX export selhal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    export_to_onnx(args)
