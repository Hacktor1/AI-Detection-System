# Testing the Detection System

## Overview
Guides for testing the dual-camera detection pipeline on both development machines
and the Jetson Orin Nano.

## Pre-Test Checklist
- [ ] Cameras connected and detected (`/dev/video*`)
- [ ] JetPack SDK installed on Jetson
- [ ] Model exported to TensorRT `.engine` format
- [ ] `requirements.txt` installed in virtual environment

## Test 1: Camera Detection
Verify both cameras are detected:
```bash
# List all video devices
ls -la /dev/video*

# Check each camera stream
python pipeline_engineer/camera_io.py --source 0  # Camera 1
python pipeline_engineer/camera_io.py --source 1  # Camera 2
```

## Test 2: Model Inference
Run inference on a sample image to confirm model loads:
```bash
python pipeline_engineer/detector.py --image data/test/sample.jpg --model models/best.engine
```

Expected output:
```
[detector] TensorRT engine loaded: models/best.engine
[detector] Inference done: 2 people detected (conf: 0.85, 0.76)
```

## Test 3: Full Dual-Camera Pipeline
Run the complete pipeline with both streams:
```bash
python pipeline_engineer/dual_camera_pipeline.py \
    --thermal-source /dev/video0 \
    --visible-source /dev/video1 \
    --model models/best.engine \
    --output-dir results/
```

This will:
1. Capture frames from both cameras simultaneously
2. Run detection on each frame
3. Overlay bounding boxes
4. Save annotated frames + detection logs to `results/`

## Test 4: Headless Mode (Jetson)
On the Jetson without display:
```bash
# Run without GUI display
python pipeline_engineer/dual_camera_pipeline.py \
    --thermal-source /dev/video0 \
    --visible-source /dev/video1 \
    --model models/best.engine \
    --no-display \
    --output-dir /tmp/results/
```

## Performance Metrics
Monitor FPS and resource usage:
```bash
# GPU utilization
tegrastats

# Pipeline log will show:
# [pipeline] Frame 100 | inference: 0.032s | boxes: 3
```

## Debugging Tips
- If one camera fails, check `dmesg | grep -i camera` for hardware errors
- If model inference is slow, try reducing input resolution (e.g., 480x360)
- Check thermal throttling: `cat /sys/class/thermal/thermal_zone*/temp`
