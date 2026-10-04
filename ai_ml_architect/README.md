# Rolle: AI/ML Architekt

## Přehled úlohy

Navrhnout, trénovat a vyhodnotit model pro detekci lidí. Hlavní kandidáti: **YOLOv8** nebo **YOLOv11**.

## Klíčové aktivity

1. **Výběr modelu**
   - Porovnat YOLOv8-nano, YOLOv8-small, YOLOv11-nano pro nasazení na hraní.
   - Zhodnotit kompromisy: přesnost vs. FPS na Jetson Nano/Orin.

2. **Trénink**
   - Použít dataset připravený Data Engineerem.
   - Nastavit hyperparametry v `train/params.yaml`.
   - Trénovat pomocí Ultralytics YOLO API.

3. **Ladění hyperparametrů**
   - Spouštět experimenty s různými nastaveními mozaiky, měřítka a augmentace.
   - Logovat výsledky do adresáře `experiments/` (nebo externího MLflow).

4. **Vhodnocení**
   - Měřit mAP, FPS a termálníspecifické metriky.
   - Exportovat nejlepší model do ONNX pro Edge Specialisty.

## Struktura adresářů
```
ai_ml_architect/
├── train/
│   ├── params.yaml      # Konfigurace tréninku
│   └── train.py         # Vstupní bod pro trénink
├── models/              # Váhy modelu (gitignored)
└── experiments/         # Logy experimentů
```

## Příkaz pro trénink
```bash
cd ai_ml_architect
python -m train.train  # Po napsání train.py
# NEBO použij Ultralytics CLI:
yolo detect train data=datasets.yaml model=yolov8n.pt epochs=50
```

## Export do ONNX (pro Edge Specialista)
```bash
yolo export model=best.pt format=onnx
```
Předávej `.onnx` soubor Edge Specialistaovi.