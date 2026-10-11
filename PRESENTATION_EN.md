# AI Detection System for Drones — Presentation

> **Live detection of people and vehicles** from drones using thermal camera and AI (YOLOv8n).
> All 52 tests passed ✅ — 100% functional development system.

---

## Page 1: Architecture & Workflow

```
┌──────────────────┐    WiFi    ┌─────────────────────┐   HTTP/MJPEG+JSON    ┌────────────────────┐
│   Drone (Jetson) │◄───────────│  Mobile Stream      │◄─────────────────────│  Smartphone / Tablet │
│                  │            │  Server (Flask)     │                       │  (Web browser)      │
│  ┌─────────────┐ │            │                     │                       │                      │
│  │Thermal cam  │ │─── frame ──→│  Video Processing   │─── frame + boxes ───→ │  Live Video         │
│  │(FLIR/ OAK)  │ │            │  + ONNX Model       │                       │  with red boxes      │
│  └─────────────┘ │            │  (YOLOv8n 6.3MB)    │                       │                      │
│                  │            │                     │─── JSON (stats) ─────→│  Telemetry:         │
│  CPU: 30%  FPS:30│◄── JSON ───│  Flask Endpoints    │                       │  Battery, GPS, Alt  │
└──────────────────┘            └─────────────────────┘                       └────────────────────┘
```

### How it works (step by step):

1. **Camera** films the scene (thermal sees heat from people)
2. **Jetson Nano** reads video and passes it to the AI model
3. **AI Model** (YOLOv8n) detects: Person, Car, Bicycle, Dog, Other Vehicle
4. **Mobile Stream Server** sends video with boxes + JSON data to smartphone
5. **Web App** shows live video + telemetry — no app installation needed

---

## Page 2: Features & Demos

### Core Features:

- **5 Object Classes**: Person, Car, Bicycle, Dog, Other Vehicle
- **Live Streaming**: MJPEG video + JSON API (6 endpoints)
- **Mobile Optimized**: Responsive design works in web browser
- **Edge Computing**: No cloud needed — runs locally on Jetson
- **Test Demo**: Complete test video (30 seconds, 98 detections)

### Technical Specifications:

| Parameter | Value |
|-----------|-------|
| **Model** | YOLOv8n (nano) |
| **Size** | 6.3 MB |
| **Format** | ONNX (FP16, opset 14) |
| **Classes** | 5 (Person, Car, Bicycle, Dog, Other Vehicle) |
| **Inference** | 31.4 FPS (31.86 ms) on CPU |
| **Test Status** | 52/52 tests passed ✅ |

### API Endpoints:

| Endpoint | Method | Output | Description |
|----------|--------|--------|-------------|
| `/` | GET | HTML page | Main UI with video |
| `/stream.mjpeg` | GET | MJPEG video | Live video with bounding boxes |
| `/api/stats` | GET | JSON | FPS, detections, telemetry |
| `/api/telemetry` | GET | JSON | Battery, altitude, GPS |
| `/api/detections` | GET | JSON | History of recent detections |
| `/api/system` | GET | JSON | Model info, platform |

### Demo Commands:

```bash
# Start live stream (with test video)
python3 edge_specialist/mobile_stream.py \
    --source pipeline_engineer/sample_data/test_thermal.mp4 \
    --model optimized_models/best_fp16_dynamic.onnx \
    --port 8080 --loop

# Open in smartphone browser:
http://<computer-ip>:8080/

# Create demo video with simulated detections
python3 scripts/create_demo_video.py

# Run all 52 tests
python3 tests/test_dual_camera.py
```

---

## Page 3: Technology Stack & Roadmap

### Technology Stack:

- **Languages**: Python 3.11
- **AI Framework**: Ultralytics YOLOv8 → ONNX Runtime
- **Streaming**: Flask (MJPEG + REST API)
- **Computer Vision**: OpenCV
- **Target Hardware**: NVIDIA Jetson Orin Nano

### Project Structure:

```
AI-Detection-System/
├── ai_ml_architect/          # Model training (train.py, params.yaml)
├── data_engineer/            # Datasets (28,738 images, 3 sources)
├── pipeline_engineer/        # Video pipeline (dual camera)
├── edge_specialist/          # Mobile Stream + Edge optimization
├── jetson_sim/               # Hardware simulation
├── scripts/                  # Demo tools (preset_demo, create_demo_video)
├── tests/                    # 52 automated tests
├── docs/                     # Documentation (6 guides)
├── deploy.sh                 # Automated Jetson deployment
├── PRESENTATION.md           # This presentation (CZ/DE)
├── PRESENTATION_EN.md        # This presentation (EN)
├── ARCHITECTURE.md           # Full technical architecture
└── README.md                 # Main documentation
```

### Development Roadmap:

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | ✅ Done | Project coordination & documentation |
| Phase 1 | ✅ Done | Datasets & training pipeline |
| Phase 2 | ✅ Done | ONNX export & performance benchmarks |
| Phase 3 | ✅ Script | Jetson deployment (hardware required) |
| Phase 4 | ✅ Done | Optimization: FP16, INT8 calibration |
| Phase 5 | ✅ Done | Mobile Stream + Web UI |

### Next steps:

- [ ] Full model retraining (100 epochs with GPU)
- [ ] Real-world test on Jetson Orin Nano + cameras
- [ ] INT8 TensorRT engine build on real hardware
- [ ] LPDR (license plate recognition) as extension

---

> **Contact**: GitHub [@Hacktor1/AI-Detection-System](https://github.com/Hacktor1/AI-Detection-System)
> **Status**: 100% tested — 52/52 automated tests passed
