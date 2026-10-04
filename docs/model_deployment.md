# Model Deployment Guide (Jetson Orin Nano)

## Overview
This guide walks through exporting a trained YOLO model from PyTorch to a
TensorRT engine optimized for inference on the Jetson Orin Nano.

## Prerequisites
- Trained YOLOv8/v9/v11 model (`.pt` file)
- Jetson Orin Nano with JetPack installed
- ONNX opset installed (`onnx`, `onnx-simplifier`)

## Step 1: Export to ONNX

On your development machine (or Jetson):
```bash
# Using Ultralytics CLI
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=13 simplify=true

# Output: best.onnx
```

## Step 2: Convert ONNX to TensorRT Engine

On the Jetson Orin Nano:
```bash
# Create TensorRT engine
/usr/src/tensorrt/bin/trtexec \
    --onnx=best.onnx \
    --saveEngine=best.engine \
    --fp16 \
    --workspace=2048 \
    --minShapes=input0:1x3x640x640 \
    --optShapes=input0:16x3x640x640 \
    --maxShapes=input0:32x32x640x640

# Output: best.engine
```

## Step 3: Load and Run in Python

Use `pipeline_engineer/detector.py` with TensorRT backend:
```python
# Example usage in pipeline
from detector import PersonDetector

detector = PersonDetector(model_path="best.engine", conf_thres=0.4)
boxes = detector.infer(frame)
```

## Performance Notes
- **YOLOv8-nano**: ~30-60 FPS @ 640x640 on Orin Nano (FP16)
- **YOLOv11**: Better accuracy, slightly slower
- **INT8 quantization**: Can boost FPS but requires calibration dataset

## Troubleshooting
- If `trtexec` fails, check ONNX opset version (use 13 or lower for compatibility)
- If engine creation fails, reduce `--workspace` size
- For dynamic shapes, ensure input tensor names match ONNX

## Next Steps
After deploying the model, test with the [Dual Camera Pipeline](testing.md).
