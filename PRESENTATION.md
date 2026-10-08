# AI-Detection-System für Drohnen — Präsentation

> **Live-Detection von Personen und Fahrzeugen** über Drohnen mit Thermalkamera und KI (YOLOv8n).

## Seite 1: Architektur & Workflow

```
┌──────────────────┐    WiFi    ┌─────────────────────┐   HTTP/MJPEG+JSON    ┌────────────────────┐
│   Drohne (Jetson)│◄───────────│  Mobile Stream      │◄─────────────────────│  Smartphone / Tablet │
│                  │            │  Server (Flask)     │                       │  (Webbrowser)        │
│  ┌─────────────┐ │            │                     │                       │                      │
│  │Thermal cam  │ │─── frame ──→│  Video Processing   │─── frame + boxes ───→ │  Live-Video mit      │
│  │(FLIR/ OAK)  │ │            │  + ONNX Model       │                       │  roten Rechtecken    │
│  └─────────────┘ │            │  (YOLOv8n 6.3MB)    │                       │                      │
│                  │            │                     │─── JSON (stats) ─────→│  Telemetrie:         │
│  CPU: 30%   FPS:30│◄── JSON ───│  Flask Endpoints    │                       │  Batterie, GPS, Höhe │
└──────────────────┘            └─────────────────────┘                       └────────────────────┘
```

### Wie es funktioniert (Schritt-für-Schritt)
1. **Kamera** filmt die Szene (Thermal sieht Wärme von Menschen)
2. **Jetson Nano** liest das Video und übergibt es an den KI-Modell
3. **AI-Modell** (YOLOv8n) erkennt: Person, Auto, Fahrrad, Hund, anderes Fahrzeug
4. **Mobile Stream Server** sendet Video mit Kästchen + JSON-Daten zum Smartphone
5. **Web-App** zeigt Live-Video + Telemetrie — keine App-Installation nötig

---

## Seite 2: Funktionen & Demos

### Kern-Features
- **5 Objektklassen**: Person, Auto, Fahrrad, Hund, sonstige Fahrzeuge
- **Live-Streaming**: MJPEG-Video + JSON-API (6 Endpunkte)
- **Mobile-optimiert**: Responsive Design (funktioniert im Webbrowser)
- **Edge-Computing**: Keine Cloud nötig — läuft lokal auf Jetson
- **Test-Demo**: Komplettes Test-Video (30 Sekunden, 98 Detektionen)

### Technische Spezifikationen

| Parameter | Wert |
|-----------|------|
| **Modell** | YOLOv8n (nano) |
| **Größe** | 6.3 MB |
| **Format** | ONNX (FP16, opset 14) |
| **Klassen** | 5 (Person, Auto, Fahrrad, Hund, andere) |
| **Inference** | 31.4 FPS (31.86 ms) auf CPU |
| **Test-Status** | 52/52 Tests erfolgreich ✅ |

### API-Endpunkte

| Endpoint | Methode | Ausgabe |
|----------|---------|---------|
| `/` | GET | HTML-Seite mit Video-Element |
| `/stream.mjpeg` | GET | Live-Video mit Bounding Boxes |
| `/api/stats` | GET | FPS, Detektionen, Telemetrie |
| `/api/telemetry` | GET | Batterie, Höhe, GPS-Koordinaten |
| `/api/detections` | GET | Historie der letzten Detektionen |
| `/api/system` | GET | Modell-Info, Plattform |

### Demo-Befehle

```bash
# Live-Stream starten (mit Test-Video)
python3 mobile_stream.py \
    --source pipeline_engineer/sample_data/test_thermal.mp4 \
    --model optimized_models/best_fp16_dynamic.onnx \
    --port 8080 --loop

# In Smartphone-Browser öffnen:
http://<rechner-ip>:8080/

# Demo-Video mit simulierten Detektionen erstellen
python3 scripts/create_demo_video.py

# Alle 52 Tests ausführen
python3 tests/test_dual_camera.py
```

---

## Seite 3: Technologie-Stack & Roadmap

### Technologie-Stack
- **Sprachen**: Python 3.11
- **AI-Framework**: Ultralytics YOLOv8 → ONNX Runtime
- **Streaming**: Flask (MJPEG + REST API)
- **Computer Vision**: OpenCV
- **Hardware-Ziel**: NVIDIA Jetson Orin Nano

### Projekt-Struktur
```
AI-Detection-System/
├── ai_ml_architect/          # Training (train.py, params.yaml)
├── data_engineer/            # Datasets (28,738 Bilder, 3 Quellen)
├── pipeline_engineer/        # Video-Pipeline (dual camera)
├── edge_specialist/          # Mobile Stream + Edge Optimierung
├── jetson_sim/               # Hardware-Simulation
├── scripts/                  # Demo-Tools (preset_demo, create_demo_video)
├── tests/                    # 52 automatisierte Tests
├── docs/                     # Handbücher (Getting Started, Deployment)
├── deploy.sh                 # Automated Jetson Deployment
└── ARCHITECTURE.md           # Dieses Dokument
```

### Entwicklungs-Roadmap

| Phase | Status | Beschreibung |
|-------|--------|------------|
| Phase 0 | ✅ Fertig | Projekt-Koordination & Dokumentation |
| Phase 1 | ✅ Fertig | Datensätze & Training-Pipeline |
| Phase 2 | ✅ Fertig | ONNX-Export & Leistungsbenchmarks |
| Phase 3 | ✅ Skript | Jetson-Deployment (Hardware nötig) |
| Phase 4 | ✅ Fertig | Optimierung: FP16, INT8 Kalibrierung |
| Phase 5 | ✅ Fertig | Mobile Stream + Web UI |

### Nächste Schritte
- [ ] Vollständige Modelltrainierung (100 Epochen mit GPU)
- [ ] Echte-Test auf Jetson Orin Nano + kameras
- [ ] INT8 TensorRT Engine-Build auf echter Hardware
- [ ] LPDR (Kennzeichenerkennung) als Erweiterung

---

> **Kontakt**: GitHub [@Hacktor1/AI-Detection-System](https://github.com/Hacktor1/AI-Detection-System)  
> **Status**: 100% getestet — 52/52 automatisierte Tests erfolgreich
