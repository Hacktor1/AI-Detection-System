# AI-Detection-System pro UAV Termální Snímání

Reálný systém pro detekci lidí a objektů pro drony pomocí termální a
viditelné kamery, nasazený na NVIDIA Jetson Orin Nano.

## Přehled systému

| Komponenta | Detaily |
|-----------|---------|
| **Hardwar** | NVIDIA Jetson Orin Nano (8GB / 16GB) |
| **Kamery** | 1× Termální (FLIR Lepton / OAK Thermal), 1× Viditelné světlo (CSI / USB) |
| **Použití** | Bezpečnostní dohled, detekce lidí, detekce SPZ |
| **Framework** | YOLOv8/v11 → ONNX → TensorRT |
| **Výstup** | Bounding boxy s štítky přidané do kamerových streamů |
| **Simulace** | Jetson sim (`jetson_sim/`): CPU thread limiting, thermal throttle sim |

## Architektura

```
┌─────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│  Termální  │    │  Video Processing  │    │  YOLOv8/v11     │
│   Kamera   │───▶│  Pipeline (Python)  │───▶│  Detekce       │
│  (FLIR/OAK) │    │  - Dual-stream     │    │  - TensorRT     │
└─────────────┘    │  - Preprocessing    │    │  - NMS         │
                   └──────────────────────┘    └────────┬────────┘
                                                          │
┌─────────────┐    ┌──────────────┐             ┌─────────▼─────────┐
│  Viditelná │    │  Camera I/O │◀────────────│  Detekční Výstup │
│  Kamera    │───▶│  (GStreamer) │             │  - Bounding Boxy │
│  (CSI/USB) │    └──────────────┘             │  - Štítky       │
└─────────────┘                                 │  - Confidence   │
                                              └─────────────────────┘
                                                          │
                                                          ▼
                                                ┌─────────────────────┐
                                                │  Annotated Display │
                                                │  (Web UI / Stream) │
                                                └─────────────────────┘
```

## Struktura repozitáře

| Role | Adresář | Účel |
|------|---------|------|
| Datový inženýr | `data_engineer/` | Objevování datasetů, stahování, předzpracování |
| AI/ML architekt | `ai_ml_architect/` | Výběr modelu, trénink, ladění hyperparametrů |
| Pipeline inženýr | `pipeline_engineer/` | Video pipeline pro dual kamery, detekce, vykreslování |
| Edge specialista | `edge_specialist/` | Optimalizace modelu, TensorRT konverze, nasazení na Jetson |
| Team lead | `team_lead/` | GitHub správa, dokumentace koordinace |

## Dokumentace

| Průvodce | Odkaz | Popis |
|----------|-------|-------|
| Začáteční průvodce | `docs/getting_started.md` | Nastavení prostředí (venv, závislosti) |
|| Hardware nastavení | `docs/hardware_setup.md` | Seznam komponent, nákupní seznam |
|| Připojení kamer | `docs/camera_wiring.md` | Jak připojit kamery k Orin Nano |
|| Instalace JetPacku | `docs/jetpack_install.md` | Jak nainstalovat JetPack SDK na Orin Nano |
|| Nasazení modelu | `docs/model_deployment.md` | Export do ONNX, konverze na TensorRT |
|| Testovací průvodce | `docs/testing.md` | Jak ověřit, že dvojité kamerové pipeline funguje |

## Rychlý start

```bash
cd AI-Detection-System
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline_engineer/run_pipeline.py --video data/sample.mp4
```

## Plán vývoje

### Fáze 0: Kostra & Dokumentace (HOTOVÉ)
- [x] Struktura repozitáře vytvořena s rolemi
- [x] Hlavní README a začátečnický průvodce
- [x] Šablony dokumentace pro hardware/instalaci/testování

### Fáze 1: Statické testování na PC (HOTOVÁ)
|- [x] Shromažďování veřejných datasetů pro termální detekci lidí
  - FLIR ADAS thermal dataset (166k anotací, 5 tříd)
  - LLVIP infrared pedestrian dataset (15k+57k obrázků, 1 třída)
  - Thermal Person Detector z HuggingFace (8,110 obrázků, 1 třída)
  - Celkem: **28,738 obrázků** s YOLO anotacemi (22,521 trénink + 6,217 validace)
|- [x] Trénink / výběr YOLOv8/v11 modelu pro lidi + objekty
  - `ai_ml_architect/train/train.py` — trénink pipeline s Ultralytics API
  - `ai_ml_architect/train/params.yaml` — konfigurace (YOLOv8n default, 100 epoch)
  - `ai_ml_architect/train/inference.py` — inference na obrázcích/video
  - Validováno: 3-epochový test trénink úspěšný, inference funguje (1.2ms/img)
  - ONNX export připraven v `edge_specialist/` pro TensorRT
|- [x] Validace přesnosti modelu na ukázkových datech
  - `data_engineer/_validate_datasets.py` — validace formátu YOLO labelů
  - `tests/test_pipeline.py` — 10 integračních testů (všechny prošly)
  - mAP50=0.274 na 3-epochovém testu (očekává se ~0.7+ po 100 epochách)

### Fáze 2: Simulace na Jetson (HOTOVÁ)
|- [x] ONNX export pipeline (`edge_specialist/export_to_onnx.py`) — FP16 dynamic opset 14
|- [x] Simulace dual camera vstupu — syntetické viditelné video z termálního
|- [x] Spuštění pipeline a benchmark FPS / latence — výsledky viz `edge_specialist/results/benchmark_results.json`:
  - ONNX Runtime (CPU): **39.7 FPS**, 25.2ms avg latency (model: 6.3MB FP16)
  - ONNX Runtime (CUDA): selhání kvůli chybějícím CUDA knihovnám, CPU fall-back aktivní
  - PyTorch inference: 0.004s/img na CPU (3-epoch model)
|- [x] TensorRT engine konverce skript (`edge_specialist/convert_to_trt.py`)
|- [x] Benchmark skript (`edge_specialist/benchmark.py`)
|- [x] INT8 kalibrační skript (`edge_specialist/int8_calibrate.py`) — calib_cache.bin
|- [x] Profiling skript (`edge_specialist/profiler.py`) — ONNX 1.48x rychlejší než PyTorch
|- [x] 11 integračních testů pro Phase 3 (všechny prošly)

### Fáze 3: Skutečná integrace hardwaru (Rozpracovaná)
|- [x] Instalace JetPack SDK dokumentace (`docs/jetpack_setup.md`)
|- [x] Model nasazení dokumentace (`docs/model_deployment.md`)
|- [x] Jetson deploy skript (`deploy.sh`) — environment check → TRT convert → pipeline
|- [x] Jetson env check (`edge_specialist/jetson_deploy.py --check`)
|- [ ] Připojení obou kamer (termální + viditelné světlo) — vyžaduje hardware
|- [x] TensorRT engine konverze (skript v `edge_specialist/convert_to_trt.py`, `jettson_deploy.py`)
|- [ ] Nasazení a spuštění plné dual-camera pipeline — vyžaduje hardware
### Fáze 4: Optimalizace & Ladění (HOTOVÁ)
|- [x] Profilování (PyTorch vs ONNX vs TensorRT) — `edge_specialist/profiler.py`
  - ONNX 382.9 FPS vs PyTorch 259.3 FPS (1.48x speedup)
|- [x] INT8 kalibrace — `edge_specialist/int8_calibrate.py` (calib_cache.bin vytvořen)
|- [x] FP16 optimalizace — 6.3MB ONNX model
|- [x] Confidence threshold tuning — lze nastavit v pipeline (`--conf-thres`)
### Fáze 5: Rozšíření funkcemi (Rozpracovaná)
|- [ ] Přidání detekce SPZ (LPDR) — plán
|- [ ] Detekce balíčků / objektů
|- [x] Web UI pro vzdálený monitoring (`edge_specialist/web_ui.py` — Flask + MJPEG)
|- [x] MAVLink bridge pro telemetrii dronu (`edge_specialist/mavlink_bridge.py`)
  - telemetry, sender, a sim režimy

## Rychlé odkazy
- [Průvodce datovým inženýrem](data_engineer/README.md)
|- [Průvodce AI/ML architektem](ai_ml_architect/README.md)
|- [Průvodce pipeline inženýrem](pipeline_engineer/README.md)
|- [Průvodce edge specialistem](edge_specialist/README.md)
|- [Průvodce team leadem](team_lead/README.md)