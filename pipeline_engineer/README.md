# Pipeline Engineer Role

## Task Overview

Build the main video processing pipeline that loads thermal camera video, runs detection, and renders bounding boxes. This is the integration point for Data Engineer (datasets), AI/ML Architect (model), and Edge Specialist (optimized model).

## Key Activities

1. **Camera Input Handling** (`camera_io.py`)
   - Read from video files (for testing).
   - Interface for dual camera streams.
   - Later: connect to Jetson CSI/MIPI camera.

2. **Detection Inference** (`detector.py`)
   - Load YOLO model (ONNX or PyTorch).
   - Run inference on frames.
   - Return bounding boxes, scores, classes.

3. **Rendering** (`renderer.py`)
   - Overlay bounding boxes on the thermal frame.
   - Handle dual camera visualization.

4. **Main Loop** (`run_pipeline.py`)
   - Orchestrate: capture → detect → render.
   - CLI args for input file, model path, output directory.

## Directory Layout
```
pipeline_engineer/
├── camera_io.py       # Camera/stream reader
├── detector.py        # Model inference wrapper
├── renderer.py        # Bounding box drawer
├── run_pipeline.py    # Main pipeline entry point
└── sample_data/       # Small sample videos for testing (gitignored)
```

## Quick Test
```bash
cd pipeline_engineer
python run_pipeline.py --video sample_data/test_thermal.mp4 --model ../ai_ml_architect/models/best.onnx
```

## Notes
- For initial testing, any MP4 video works as input.
- Model can be `.engine` (TensorRT) when deployed on Jetson.