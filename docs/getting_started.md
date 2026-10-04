# Začátečnický průvodce

## Požadavky

- Python 3.10+
- pip nebo uv (pro správu prostředí)
- Git

## Nastavení prostředí

### Pomocí standardní venv + pip
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Pomocí uv (doporučeno)
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Ověření instalace
```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import ultralytics; print('YOLO:', ultralytics.__version__)"
```

## Rychlý start (Pipeline)

```bash
python pipeline_engineer/run_pipeline.py --video data/sample.mp4
```

## Struktura projektu

```
AI-Detection-System/
├── docs/
│   └── getting_started.md          # Tento soubor
├── data_engineer/
│   └── datasets/                   # Stažené datasety (gitignored)
├── ai_ml_architect/
│   ├── train/                      # Tréninkové skripty
│   └── models/                     # Váhy modelu (gitignored)
├── pipeline_engineer/
│   ├── camera_io.py                # Obsluha kamer
│   ├── detector.py                 # Inferenční engine detekce
│   ├── renderer.py                 # Overlay bounding boxů
│   └── run_pipeline.py             # Hlavní pipeline
├── edge_specialist/
│   └── tensorrt/                   # Konverzní skripty
├── team_lead/
│   └── coordination.md             # Poznámky koordinace týmu
├── requirements.txt
├── .gitignore
└── README.md
```

## Poznámky
- Slovníky `data/` a `models/` jsou úmyslně gitignored (velké binární soubory).
- Pattern `_*` v `.gitignore` vylučuje místní nástrojové skripty.