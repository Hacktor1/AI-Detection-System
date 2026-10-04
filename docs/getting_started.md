# Getting Started

## Prerequisites

- Python 3.10+
- pip or uv (for environment management)
- Git

## Environment Setup

### Using standard venv + pip
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Using uv (recommended)
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Verify installation
```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import ultralytics; print('YOLO:', ultralytics.__version__)"
```

## Quick Start (Pipeline)

```bash
python pipeline_engineer/run_pipeline.py --video data/sample.mp4
```

## Project Layout

```
AI-Detection-System/
├── docs/
│   └── getting_started.md          # This file
├── data_engineer/
│   └── datasets/                   # Downloaded datasets (gitignored)
├── ai_ml_architect/
│   ├── train/                      # Training scripts
│   └── models/                     # Model weights (gitignored)
├── pipeline_engineer/
│   ├── camera_io.py                # Camera input handling
│   ├── detector.py                 # Detection inference
│   ├── renderer.py                 # Bounding box overlay
│   └── run_pipeline.py             # Main pipeline
├── edge_specialist/
│   └── tensorrt/                   # Conversion scripts
├── team_lead/
│   └── coordination.md             # Team coordination notes
├── requirements.txt
├── .gitignore
└── README.md
```

## Notes
- The `data/` and `models/` directories are intentionally gitignored (large binaries).
- The `_*` pattern in `.gitignore` excludes local utility scripts.