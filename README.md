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
| **Výstup** | Bounding boxy s štítky přidanými do kamerových streamů |

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

### Fáze 1: Statické testování na PC
- [ ] Shromažďování veřejných datasetů pro termální detekci lidí
- [ ] Trénink / výběr YOLOv8/v11 modelu pro lidi + objekty
- [ ] Validace přesnosti modelu na ukázkových videích

### Fáze 2: Simulace na Jetson (Testovací prostředí)
- [ ] Instalace JetPack SDK simulátor nebo Jetson Nano
- [ ] Simulace dual camera vstupu pomocí video souborů
- [ ] Spuštění pipeline a benchmark FPS / latence

### Fáze 3: Skutečná integrace hardwaru (Orin Nano + Kamery)
- [ ] Instalace JetPack 6 na Orin Nano
- [ ] Připojení obou kamer (termální + viditelné světlo)
- [ ] Konverze modelu do TensorRT enginy
- [ ] Nasazení a spuštění plné dual-camera pipeline

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