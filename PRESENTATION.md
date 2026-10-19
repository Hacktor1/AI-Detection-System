# AI-Detekční Systém pro Drony — Prezentace

> **Živá detekce osob a vozidel** přes drony s termokamerou a AI (YOLOv8n).

## Strana 1: Architektura a průběh (Workflow)

┌──────────────────┐    WiFi    ┌─────────────────────┐   HTTP/MJPEG+JSON    ┌────────────────────┐
│   Dron (Jetson)  │◄───────────│  Mobilní Stream     │◄─────────────────────│  Smartphone / Tablet│
│                  │            │  Server (Flask)     │                       │  (Webový prohlížeč)│
│  ┌─────────────┐ │            │                     │                       │                    │
│  │Termokamera  │ │── snímek ─→│  Zpracování videa   │── snímek + rámečky ─→ │  Živé video s      │
│  │(FLIR/ OAK)  │ │            │  + ONNX Model       │                       │  červenými obdélníky│
│  └─────────────┘ │            │  (YOLOv8n 6.3MB)    │                       │                    │
│                  │            │                     │─── JSON (statistiky)─→│  Telemetrie:       │
│  CPU: 30%   FPS:30│◄── JSON ───│  Endpoints Flasku   │                       │  Baterie, GPS, výška│
└──────────────────┘            └─────────────────────┘                       └────────────────────┘

### Jak to funguje (Krok za krokem)
1. **Kamera** snímá scénu (termokamera vidí teplo vyzářené lidmi)
2. **Jetson Nano** načítá video a předává ho AI modelu
3. **AI model** (YOLOv8n) detekuje: Osobu, Auto, Kolo, Psa, Jiné vozidlo
4. **Mobilní Stream Server** odesílá video s rámečky + JSON data do smartphonu
5. **Webová aplikace** zobrazuje živé video + telemetrii — bez nutnosti instalovat aplikaci

---

## Strana 2: Funkce a Dema

### Hlavní funkce
- **5 tříd objektů**: Osoba, Auto, Kolo, Pes, Ostatní vozidla
- **Živý streaming**: MJPEG video + JSON API (6 endpointů)
- **Optimalizováno pro mobilní zařízení**: Responsivní design (funguje ve webovém prohlížeči)
- **Edge computing**: Není potřeba cloud — běží lokálně na Jetsonu
- **Testovací demo**: Kompletní testovací video (30 sekund, 98 detekcí)

### Technické specifikace

| Parametr | Hodnota |
|-----------|------|
| **Model** | YOLOv8n (nano) |
| **Velikost** | 6.3 MB |
| **Formát** | ONNX (FP16, opset 14) |
| **Třídy** | 5 (Osoba, Auto, Kolo, Pes, Ostatní) |
| **Inference** | 31.4 FPS (31.86 ms) na CPU |
| **Stav testů** | 52/52 testů úspěšných ✅ |

### API Endpointy

| Endpoint | Metoda | Výstup |
|----------|---------|---------|
| `/` | GET | HTML stránka s prvkem pro video |
| `/stream.mjpeg` | GET | Živé video s ohraničujícími rámečky (Bounding Boxes) |
| `/api/stats` | GET | FPS, detekce, telemetrie |
| `/api/telemetry` | GET | Baterie, výška, GPS souřadnice |
| `/api/detections` | GET | Historie posledních detekcí |
| `/api/system` | GET | Informace o modelu, platforma |

### Demo příkazy

# Spuštění živého streamu (s testovacím videem)
python3 mobile_stream.py \
    --source pipeline_engineer/sample_data/test_thermal.mp4 \
    --model optimized_models/best_fp16_dynamic.onnx \
    --port 8080 --loop

# Otevření v prohlížeči smartphonu:
http://<ip-pocitace>:8080/

# Vytvoření demo videa se simulovanými detekcemi
python3 scripts/create_demo_video.py

# Spuštění všech 52 testů
python3 tests/test_dual_camera.py

---

## Strana 3: Technologický stoh a Roadmapa

### Technologický stoh (Technology Stack)
- **Jazyky**: Python 3.11
- **AI Framework**: Ultralytics YOLOv8 → ONNX Runtime
- **Streaming**: Flask (MJPEG + REST API)
- **Počítačové vidění**: OpenCV
- **Cílový hardware**: NVIDIA Jetson Orin Nano

### Struktura projektu
AI-Detection-System/
├── ai_ml_architect/          # Trénování (train.py, params.yaml)
├── data_engineer/            # Datasets (28 738 obrázků, 3 zdroje)
├── pipeline_engineer/        # Video pipeline (duální kamera)
├── edge_specialist/          # Mobilní stream + edge optimalizace
├── jetson_sim/               # Simulace hardwaru
├── scripts/                  # Demo nástroje (preset_demo, create_demo_video)
├── tests/                    # 52 automatizovaných testů
├── docs/                     # Příručky (Jak začít, Nasazení)
├── deploy.sh                 # Automatizované nasazení na Jetson
└── ARCHITECTURE.md           # Tento dokument

### Roadmapa vývoje

| Fáze | Stav | Popis |
|-------|--------|------------|
| Fáze 0 | ✅ Hotovo | Koordinace projektu a dokumentace |
| Fáze 1 | ✅ Hotovo | Datové sady a trénovací pipeline |
| Fáze 2 | ✅ Hotovo | Export do ONNX a srovnávací testy výkonu |
| Fáze 3 | ✅ Skript | Nasazení na Jetson (vyžaduje hardware) |
| Fáze 4 | ✅ Hotovo | Optimalizace: FP16, INT8 kalibrace |
| Fáze 5 | ✅ Hotovo | Mobilní stream + Web UI |

### Další kroky
- [ ] Úplné trénování modelu (100 epoch s GPU)
- [ ] Reálné testování na Jetson Orin Nano + kamerách
- [ ] Sestavení INT8 TensorRT Engine na reálném hardwaru
- [ ] LPDR (Rozpoznávání SPZ) jako rozšíření

---