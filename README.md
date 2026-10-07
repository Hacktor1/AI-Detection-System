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
- [x] Simulační prostředí Jetson Orin Nano (`jetson_sim/`) — CPU thread limiting, thermal throttling, memory limits
- [x] Simulace dual camera vstupu — syntetické viditelné video z termálního (`jetson_sim/make_sample_video.py`)
- [x] Dual camera pipeline s paralelním čtením obou streamů (`pipeline_engineer/dual_camera_pipeline.py`)
- [x] Spuštění pipeline a benchmark FPS / latence — výsledky viz `results/benchmark_phase2.json`:
  - ONNX Runtime (4 threads, sim Jetson): **47.3 FPS**, 21.1ms avg latency
  - ONNX Runtime (CUDA): **337 FPS**, 3.0ms avg latency
  - PyTorch (CUDA): **241 FPS**, 4.2ms avg latency
- [x] TensorRT engine konverze skript (`edge_specialist/convert_to_trt.py`) — dry-run ověřen
- [x] Benchmark skript pro srovnání backends (`edge_specialist/benchmark.py`)
- [x] 14 integračních testů pro simulaci a pipeline (všechny prošly)

### Fáze 3: Skutečná integrace hardwaru (CÍL)
|- [ ] Instalace JetPack 6 na Orin Nano
|- [ ] Připojení obou kamer (termální + viditelné světlo)
|- [ ] Konverze modelu do TensorRT enginy (skript připraven v `edge_specialist/convert_to_trt.py`)
|- [ ] Nasazení a spuštění plné dual-camera pipeline

### Fáze 4: Optimalizace & Ladění
- [ ] Profilování GPU/CPU využití na Orin Nano
- [ ] Optimalizace velikosti modelu pro FP16 nebo INT8 inferenci
- [ ] Ladění confidence threshold podle typu kamery

### Fáze 5: Rozšíření funkcemi (Budoucnost)
- [ ] Přidání detekce SPZ (LPDR)
- [ ] Detekce balíčků / objektů
- [ ] Přidání webového rozhraní pro vzdálený monitoring
- [ ] Integrace s MAVLink pro telemetrii dronu

## Rychlé odkazy
- [Průvodce datovým inženýrem](data_engineer/README.md)
|- [Průvodce AI/ML architektem](ai_ml_architect/README.md)
|- [Průvodce pipeline inženýrem](pipeline_engineer/README.md)
|- [Průvodce edge specialistem](edge_specialist/README.md)
|- [Průvodce team leadem](team_lead/README.md)