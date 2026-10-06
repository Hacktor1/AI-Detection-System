# Unifikovaný dataset pro UAV termální detekci lidí a aut

> Tento dokument popisuje strukturu a metadata pro trénink YOLO modelu na kombinaci všech stažených termálních datasetů.

## Cíl

Entrvírovat jeden unifikovaný dataset z více zdrojů pro detekci **lidí** a **aut** v termálním světle pro použití na dronu (UAV) v režimu hledání a záchranky (SAR).

## Zdroje datasetů

Kombinace všech tří stažených datasetů vede k této celkové částce:

| Dataset | Train | Val/Test | Třídy |
|---|---|---|---|
| FLIR ADAS (aligned) | 4,129 | 1,013 | person, car, bicycle, dog, other_vehicle |
| LLVIP (infrared) | 12,023 | 3,463 | person |
| Thermal Person Detector | 6,369 | 1,741 | person |
| **Celkem** | **22,521** | **6,217** | person (0), car (1), bicycle (2), dog (3), other_vehicle (4) |

## Unifikovaný formát

Všechny datasety byly převedeny do **YOLO formátu** (xywh, normalizované 0–1):

```
datasets/processed/
├── flir_adas/
│   ├── images/
│   │   ├── train/     # 4,129 JPG (640x512)
│   │   └── val/       # 1,013 JPG
│   ├── labels/
│   │   ├── train/     # YOLO .txt soubory
│   │   └── val/
│   └── dataset.yaml
├── llvip/
│   ├── images/
│   │   ├── train/     # 12,023 IR JPG (1280x1024)
│   │   └── test/      # 3,463 IR JPG
│   ├── labels/
│   │   ├── train/     # YOLO .txt soubory
│   │   └── test/
│   └── dataset.yaml
└── thermal_person_detector/
    ├── images/
    │   ├── train/     # 6,369 JPG
    │   └── test/      # 1,741 JPG
    ├── labels/
    │   ├── train/     # YOLO .txt soubory
    │   └── test/
    └── dataset.yaml
```

## Unifikovaná třídění

Všechny datasety používají stejné ID pro třídy:

| ID | Třída |
|---|---|
| 0 | person |
| 1 | car |
| 2 | bicycle |
| 3 | dog |
| 4 | other_vehicle |

## Unifikovaný dataset.yaml

Pro trénink na kombinovaném datasetu vytvořte soubor `datasets/processed/combined/dataset.yaml`:

```yaml
path: /home/hacktor/Projects/GitHub-Hacktor1/AI-Detection-System/data_engineer/datasets/processed
train: ../combined/train.txt
val:   ../combined/val.txt

nc: 5
names: ['person', 'car', 'bicycle', 'dog', 'other_vehicle']
```

Soubory `train.txt` a `val.txt` budou obsahovat úplné cesty k obrázkům ze všech tří datasetů.

## Jak použít pro trénink

```bash
# Aktivujte virtuální prostředí
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Trénink s Ultralytics YOLOv8
yolo task=detect mode=train \
  data=data_engineer/datasets/processed/combined/dataset.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16
```

## Poznámky pro UAV nasazení

- **Termální imagery** je citlivní na teplotu podlohy — model je trénován i na data s různými podmínkami.
- **Detekce lidí** je primární — `person` třída má nejvyšší priorítů v SAR režimu.
- **Detekce aut** je sekundární — užitečná pro identifikaci vozidel v havariích.
- Všechny datasety obsahují snímky zvýšené výšky (drony), které odpovídají reálné nasazení.
