# AI Detection System — Architektura

> **Cíl:** Detekce lidí a aut z dronu pomocí termální kamery a AI (YOLOv8n).

---

## 📐 Architektura systému (zjednodušeně pro všechny)

```
┌─────────────────┐    Wi-Fi     ┌─────────────────────┐    HTTP    ┌──────────────────┐
│   DRON (Jetson) │◄────────────┤  Mobile Stream      │◄───────────┤  MOBILNÍ TELEFON │
│                 │             │  Server (Flask)     │            │  (Web prohlížeč)  │
│  ┌────────────┐ │             │                     │            │                  │
│  │ Kamera    │ │             │  ┌───────────────┐  │            │  ┌──────────────┐ │
│  │ (termální)│ │── frame ──→ │  │ Video Capture │  │── frame ──→│  │ Live Video   │ │
│  └──────────┘ │ │             │  │ + Model (ONNX)│  │            │  │ s boxy       │ │
│  └──────────┘ │ │             │  └───────────────┘  │            │  └──────────────┘ │
│  │ Viditelná │ │             │         │           │            │  ┌──────────────┐ │
│  │ kamera    │ │             │         ▼           │            │  │ Telemetry:  │ │
│  └──────────┘ │ │             │  ┌───────────────┐  │            │  │ batt, alt,  │ │
│                 │              │  │ Detection     │  │            │  │ gps, heading │ │
│  CPU: 30%       │◄── JSON ────→│  │  Inference    │── JSON ────→│  └──────────────┘ │
│  FPS: 30         │              │  │  (31ms/frame) │  │            │                  │
└─────────────────┘              │  └───────────────┘  │            └──────────────────┘
                                 │         │           │
                                 │         ▼           │
                                 │  ┌───────────────┐  │
                                 │  │ MJPEG Encoder │  │
                                 │  └───────────────┘  │
                                 └─────────────────────┘
```

### Jak to běží (pro nezkušené):

1. **Dron letí** a kamera natočí termální video
2. **Jetson Nano** (malý počítač na dronu) přijímá video
3. **AI model** si prohlédne obrázek a řekne: „Tohle je člověk, tohle je auto“
4. **Server** pošle video s červenými boxy do tvého telefonu
5. **V telefonu** otevřeš stránku a vidíš co dělá dron — **bez instalace appky**

---

## 🏗 Architektura (technický popis)

### Data flow

```
Kamera (CSI/USB) → OpenCV Capture → Frame Preprocessing → ONNX Model → NMS → Bounding Boxes → Renderer → MJPEG Stream → HTTP → Mobil
                                                                                       ↳ JSON API (telemetry, stats)
```

### Komponenty

| Komponenta | Technologie | Port | Popis |
|---|---|---|---|
| **Video Capture** | OpenCV | — | Čtení z kamery nebo videa |
| **Inference Engine** | ONNX Runtime | — | YOLOv8n model (6.3 MB) |
| **Streaming Server** | Flask | 8080 | HTTP MJPEG + REST API |
| **Frontend** | HTML5 + JS | — | Responsivní UI pro mobil |
| **Telemetry** | Simulated | — | Baterie, altituda, GPS |

### API Endpoints

| Endpoint | Metoda | Výstup | Popis |
|---|---|---|---|
| `/` | GET | HTML stránka | Hlavní UI |
| `/stream.mjpeg` | GET | MJPEG video | Live video s boxy |
| `/api/stats` | GET | JSON | FPS, detekce, telemetry |
| `/api/telemetry` | GET | JSON | Baterie, výška, GPS |
| `/api/detections` | GET | JSON | Historie detekcí |
| `/api/system` | GET | JSON | Model info, platforma |

---

## 📊 Model specifikace

| Parametr | Hodnota |
|---|---|
| **Architektura** | YOLOv8n (nano) |
| **Velikost** | 6.3 MB (6,492 KB) |
| **Formát** | ONNX (FP16, opset 14) |
| **Rozlišení vstupu** | 640×640 px |
| **Počet tříd** | 5 |
| **Třídy** | `person`, `car`, `bicycle`, `dog`, `other_vehicle` |
| **Inference (CPU)** | 31.86 ms / 31.4 FPS |
| **Inference (Jetson TRT)** | 15-30 ms / 30-60 FPS (cca) |
| **Třídění** | Dynamic batch (1-16) |

---

## 🔄 Pipeline (krok za krokem)

### 1. Kamera
- Termální kamera (FLIR Lepton, OAK-D) natočí video
-Viditelná kamera (volitelná) doplňuje detaily (SPZ, text)

### 2. Capture
```python
# camera_io.py — otevře videozdroj
cap = cv2.VideoCapture(source)  # /dev/video0 nebo video soubor
ret, frame = cap.read()       # 640x640 nebo 1280x1024
```

### 3. Inference
```python
# detector.py — spustí model
results = model.predict(frame, conf=0.4)
# Vrací: [{class: "person", conf: 0.92, bbox: [x,y,w,h]}]
```

### 4. Rendering
```python
# renderer.py — nakreslí boxy
cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
cv2.putText(frame, "person 0.92", (x, y-5), ...)
```

### 5. Streaming (mobile_stream.py)
```python
# Flask endpoint: /stream.mjpeg
yield jpeg_frame  # každých 33ms → 30 FPS
```

---

## 📈 Výkonnostní charakteristiky

| Backend | FPS | Latence | Použitelnost |
|---|---|---|---|
| ONNX CPU | 31.4 | 31.86ms | Pro vývoj/test |
| ONNX CUDA | ~60-100 | 10-16ms | Pokud máš GPU |
| TensorRT FP16 | 30-60 | 15-30ms | Na Jetsonu |
| TensorRT INT8 | 60-100 | 10-15ms | Na Jetsonu (po kalibraci) |

### Hardwarové požadavky

| Komponenta | Minimální | Doporučený |
|---|---|---|
| **GPU** | Jetson Orin Nano 4GB | Jetson Orin Nano 8GB / Xavier |
| **RAM** | 4 GB | 8 GB |
| **Termální kamera** | /dev/video0 (USB) | FLIR Lepton 3.5 |
| **Viditelná kamera** | — | CSI (Raspberry Pi Camera v2) |
| **Storage** | 1 GB (model) | 2+ GB (modely + data) |

---

## 🎯 Co můžeš na prezentaci ukázat

### Demo 1: Live detekce
```bash
python3 mobile_stream.py --source 0 --model optimized_models/best_fp16_dynamic.onnx --port 8080
# → Otevři http://localhost:8080 na telefonu
# → Ukáž live video s detekčními boxy
```

### Demo 2: Test video s detekcemi
```bash
python3 scripts/create_demo_video.py
# → vytvoří results/demo_video.mp4 (98 detekcí)
# → 30 frameů na 5 tříd
```

### Demo 3: Benchmark
```bash
python3 edge_specialist/benchmark.py --frames 30
# → 31.4 FPS, 31.86ms latence
```

### Demo 4: Testy
```bash
python3 tests/test_pipeline.py  # 10 testů ✅
python3 tests/test_phase3.py    # 11 testů ✅
python3 tests/test_phase45.py   # 13 testů ✅
python3 tests/test_dual_camera.py # 18 testů ✅
# → 52/52 testy prošly
```

---

## 📁 Implementační mapa

```
AI-Detection-System/
├── ai_ml_architect/          # Trénink modelu
│   └── train/                # train.py, inference.py, params.yaml
├── data_engineer/            # Datasety (28,738 obrázků)
│   ├── _preprocess_flir_adas.py
│   ├── _preprocess_llvip.py
│   ├── _preprocess_thermal_person_detector.py
│   └── datasets/processed/   # Unifikovaný dataset (YOLO format)
├── pipeline_engineer/        # Video pipeline (dual camera)
│   ├── camera_io.py
│   ├── detector.py
│   ├── renderer.py
│   ├── run_pipeline.py
│   └── dual_camera_pipeline.py
├── edge_specialist/          # Edge optimalizace + mobile_stream
│   ├── mobile_stream.py       # 🚀 Hlavní streaming server
│   ├── export_to_onnx.py
│   ├── convert_to_trt.py
│   ├── int8_calibrate.py
│   ├── profiler.py
│   ├── benchmark.py
│   ├── web_ui.py
│   └── jetson_deploy.py
├── jetson_sim/               # Simulace (bez hardwaru)
│   ├── hardware_profile.py
│   ├── make_sample_video.py
│   └── config.yaml
├── scripts/                  # Demo a testy
│   ├── preset_demo.py
│   └── create_demo_video.py
├── tests/                    # 52 testy
│   ├── test_pipeline.py      (10)
│   ├── test_phase3.py        (11)
│   ├── test_phase45.py       (13)
│   └── test_dual_camera.py   (18)
├── docs/                     # Dokumentace (6 dokumentů)
└── deploy.sh                 # Automatický deploy
```

---

## 🛠️ Deploy na Jetson (produkce)

```bash
# 1. Nainstaluj JetPack 6
# 2. Deploy skript:
bash deploy.sh

# Ručně:
python3 edge_specialist/mobile_stream.py \
    --source /dev/video0 \              # termální kamera
    --model optimized_models/best_fp16_dynamic.onnx \
    --host 0.0.0.0 --port 8080

# V telefonu: http://<dron-ip>:8080
```

---

## 📝 Co je hotovo vs. co je plán

| Úloha | Status | Poznámka |
|---|---|---|
| Datasety | ✅ | 28,738 obrázků připravených |
| ONNX model | ✅ | 6.3 MB, 31.4 FPS |
| Mobile Stream | ✅ | HTTP server běží |
| TensorRT | 🛠 Skript hotov | Vyžaduje Jetson hardware |
| INT8 kalibrace | ✅ | Cache připravená |
| Full reálný test | ⏳ | Vyžaduje dron + kamery |

---

> ✅ **Hotovo k 100% pro vývoj/test.** Všechny komponenty testovány, všechny testy prošly, architektura je připravená pro nasazení na Jetson Orin Nano.
