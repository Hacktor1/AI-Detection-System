# AI-Detection-System for UAV Thermal Imaging

Real-time human and object detection system for drones using dual thermal
and visible cameras, deployed on NVIDIA Jetson Orin Nano.

## System Overview

| Component | Details |
|-----------|---------|
| **Hardware** | NVIDIA Jetson Orin Nano (8GB / 16GB) |
| **Cameras** | 1× Thermal (FLIR Lepton / OAK Thermal), 1× Visible (CSI / USB) |
| **Use Case** | Security surveillance, human detection, license plate detection |
| **Framework** | YOLOv8/v11 → ONNX → TensorRT |
| **Output** | Bounding boxes with class labels overlaid on camera feeds |

## Architecture

```
┌─────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│  Thermal    │    │  Video Processing   │    │  YOLOv8/v11     │
│   Camera    │───▶│  Pipeline (Python) │───▶│  Detection      │
│  (FLIR/OAK) │    │  - Dual-stream     │    │  - TensorRT      │
└─────────────┘    │  - Preprocessing    │    │  - NMS           │
                   └──────────────────────┘    └────────┬────────┘
                                                          │
┌─────────────┐    ┌──────────────┐             ┌─────────▼─────────┐
│  Visible    │    │  Camera I/O  │◀────────────│  Detection Output │
│  Camera     │───▶│  (GStreamer) │             │  - Bounding Boxes │
│  (CSI/USB)  │    └──────────────┘             │  - Class Labels   │
└─────────────┘                                 │  - Confidence     │
                                              └─────────────────────┘
                                                          │
                                                          ▼
                                                ┌─────────────────────┐
                                                │  Annotated Display  │
                                                │  (Web UI / Stream)  │
                                                └─────────────────────┘
```

## Repository Structure

| Role | Directory | Purpose |
|------|-----------|---------|
| Data Engineer | `data_engineer/` | Dataset discovery, download, preprocessing |
| AI/ML Architect | `ai_ml_architect/` | Model selection, training, hyperparameter tuning |
| Pipeline Engineer | `pipeline_engineer/` | Dual-camera video pipeline, detection, rendering |
| Edge Specialist | `edge_specialist/` | Model optimization, TensorRT conversion, Jetson deployment |
| Team Lead | `team_lead/` | GitHub management, coordination docs |

## Documentation

| Guide | Link | Description |
|-------|------|-------------|
| Getting Started | `docs/getting_started.md` | Environment setup (venv, dependencies) |
| Hardware Setup | `docs/hardware_setup.md` | Bills of materials, component list |
| Camera Wiring | `docs/camera_wiring.md` | How to connect cameras to Orin Nano |
| JetPack Install | `docs/jetpack_install.md` | Installing JetPack SDK on Orin Nano |
| Model Deployment | `docs/model_deployment.md` | Export to ONNX, convert to TensorRT |
| Testing Guide | `docs/testing.md` | How to verify the full pipeline works |

## Quick Start

```bash
cd AI-Detection-System
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline_engineer/run_pipeline.py --video data/sample.mp4
```

## Development Roadmap

### Phase 0: Skeleton & Documentation (DONE)
- [x] Repo structure created with role-based directories
- [x] README and getting started guides
- [x] Hardware/installation/testing documentation templates

### Phase 1: Static Testing on PC
- [ ] Collect/open-source datasets for thermal human detection
- [ ] Train / select YOLOv8/v11 model for people + objects
- [ ] Validate model accuracy offline on sample videos

### Phase 2: Simulation on Jetson (Dummy Environment)
- [ ] Install JetPack SDK simulator or Jetson Nano
- [ ] Simulate dual camera input with video files
- [ ] Run pipeline and benchmark FPS / latency

### Phase 3: Real Hardware Integration (Orin Nano + Cameras)
- [ ] Install JetPack 6 on Orin Nano
- [ ] Connect both cameras (thermal + visible)
- [ ] Convert model to TensorRT engine
- [ ] Deploy and run full dual-camera pipeline

### Phase 4: Optimization & Tuning
- [ ] Profile GPU/CPU utilization on Orin Nano
- [ ] Optimize model size for FP16 or INT8 inference
- [ ] Tune confidence thresholds per camera type

### Phase 5: Feature Expansion (Future)
- [ ] Add license plate detection (LPDR)
- [ ] Package/object detection
- [ ] Add web dashboard for remote monitoring
- [ ] Integrate with MAVLink for drone telemetry

## Quick Links
- [Data Engineer Guide](data_engineer/README.md)
- [AI/ML Architect Guide](ai_ml_architect/README.md)
- [Pipeline Engineer Guide](pipeline_engineer/README.md)
- [Edge Specialist Guide](edge_specialist/README.md)
- [Team Lead Guide](team_lead/README.md)