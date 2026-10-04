# AI-Detection-System for UAV Thermal Imagery

Real-time people detection system for drones using thermal cameras,
deployed on NVIDIA Jetson Orin Nano.

## System Overview

- **Hardware Target**: NVIDIA Jetson Orin Nano
- **Input**: Dual thermal cameras (local to drone)
- **Processing**: Python pipeline → YOLOv8/v11 → ONNX → TensorRT
- **Output**: Bounding boxes overlaid on thermal video stream
- **Goal**: Real-time human detection at long range

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Thermal    │    │  Video       │    │  TensorRT    │
│   Camera    │───▶│   Pipeline   │───▶│   Inference  │
│   (x2)      │    │    (Python)  │    │   (Jetson)   │
└─────────────┘    └──────┬───────┘    └──────┬───────┘
                          │                   │
                    ┌─────▼─────┐       ┌─────▼─────┐
                    │  YOLOv8   │       │  Bounding │
                    │ Detection │       │  Boxes    │
                    └───────────┘       └───────────┘
```

## Repository Structure

| Role | Directory | Purpose |
|------|-----------|---------|
| Data Engineer | `data_engineer/` | Dataset discovery, download, preprocessing |
| AI/ML Architect | `ai_ml_architect/` | Model selection, training, hyperparameter tuning |
| Pipeline Engineer | `pipeline_engineer/` | Video processing loop, camera I/O, rendering |
| Edge Specialist | `edge_specialist/` | Model optimization, TensorRT conversion, Jetson deployment |
| Team Lead | `team_lead/` | GitHub management, coordination docs |

## Getting Started

See [docs/getting_started.md](docs/getting_started.md) for setup instructions.

Quick start:
```bash
cd AI-Detection-System
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline_engineer/run_pipeline.py
```

## Quick Links
- [Data Engineer Guide](data_engineer/README.md)
- [AI/ML Architect Guide](ai_ml_architect/README.md)
- [Pipeline Engineer Guide](pipeline_engineer/README.md)
- [Edge Specialist Guide](edge_specialist/README.md)
- [Team Lead Guide](team_lead/README.md)
- [Getting Started](docs/getting_started.md)