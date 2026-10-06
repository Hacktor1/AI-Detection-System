# Registr termálních datasetů pro detekci lidí a aut

> Shromažďováno a udržováno týmem data_engineer. Každý řádek představuje jeden veřejný zdroj vhodný pro trénink YOLO/lidské detekce na termálním imagery pro UAV.

## Stažené datasety (in-scope pro UAV termální detekci lidí + aut)

| # | Název | Umístění | Licence | Velikost | Počet anotací | Třídy | Formát | Poznámka |
|---|---|---|---|---|---|---|---|---|
| 1 | **FLIR ADAS Thermal (aligned)** | `datasets/raw/flir_adas/align/` | Bezplatná komersní | 1.3 GB | 4,129 train + 1,013 val = 5,142 XML | `person`, `car`, `bicycle`, `dog`, `other vehicle` | PASCAL VOC XML | Párové RGB-T snímky, ručně zarovnané. Obsahuje detekci lidí i aut. Stáhnuto z Google Drive (https://drive.google.com/file/d/1xHDMGl6HJZwtarNWkEV3T4O9X4ZQYz2Y). |
| 2 | **LLVIP** | `datasets/raw/llvip/LLVIP/` | Apache 2.0 | 3.9 GB | 15,488 XML | `person` | PASCAL VOC XML | 30,976 snímků (15,488 párových RGB-T). Zaměřeno na detekci lidí ve tmavosti. Obsahuje `infrared/train/`, `infrared/test/`, `visible/train/`, `visible/test/`. Stáhnuto z Google Drive. |
| 3 | **Thermal Person Detector (HF)** | `datasets/raw/thermal_person_detector/` | CC-BY-4.0 | 254 MB | 8,778 JPG | `person` | Embedded (FiftyOne) | Jednoduchý thermal dataset pouze s lidmi. Obsahuje pouze obrázky (anotace embedded v FiftyOne formátu). Staženo z HuggingFace (Voxel51/Thermal-Person-Detector). |

## Další k dispozici (nejsou ještě staženy)

| Název | Zdroj | Licence | Formát | Poznámka |
|---|---|---|---|---|
| **HIT-UAV** | https://github.com/HIT-GIDS-HUST/HIT-UVRD | Apache 2.0 | YOLO/COCO/DOTA | Termální + viditelné snímky z dronů. |
| **RGBTDronePerson** | https://github.com/mil-tv/RGBTDronePerson | Apache 2.0 | YOLO/COCO | Drony, viditelné + termální paire. |
| **Roboflow People Detection - Thermal** | https://universe.roboflow.com/... | Otvřená | YOLO/COCO | 26,014 obrázků. Vyžaduje Roboflow API klíč. |

## Shrnutí pro detekci lidí a aut (Cíl projektu)

- **Detekce lidí**: FLIR ADAS, LLVIP, Thermal Person Detector → 3 zdroje
- **Detekce aut**: FLIR ADAS (obsahuje `car` třídu) → hlavní zdroj
- **UAV kontext**: LLVIP (drony), FLIR ADAS (automobilový kontext, lze upravit)

## Struktura adresářů

```
data_engineer/datasets/
├── raw/
│   ├── flir_adas/
│   │   └── align/
│   │       ├── Annotations/       # PASCAL VOC XML
│   │       ├── JPEGImages/        # Termální + RGB JPG
│   │       └── AnnotatedImages/   # Vizualizace anotací
│   ├── llvip/
│   │   └── LLVIP/
│   │       ├── Annotations/       # PASCAL VOC XML (15,488)
│   │       ├── infrared/
│   │       │   ├── train/         # 12,025 obrázků
│   │       │   └── test/          # 3,463 obrázků
│   │       └── visible/
│   │           ├── train/         # 12,025 obrázků
│   │           └── test/          # 3,463 obrázků
│   └── thermal_person_detector/
│       └── data/                  # 8,778 JPG obrázků
├── _download_llvip.py             # Skript pro stažení LLVIP (manifest)
└── dataset_registry.md            # Tento soubor
```

## Licence / etické požadavky

- Všechny datasety jsou určeny výhradně pro výzkumné a vývojové účely.
- FLIR ADAS podléhá registraci před stažením — licencováno jako "free for research".
- LLVIP: Apache 2.0 — volně použitelné pro komerční i akademické účely.
- Thermal Person Detector: CC-BY-4.0 — vyžaduje citaci původní práci.
- Nikdy nezveřejňovat ani nesdílet originální data mimo rámec školení modelu.

---

> Generováno dle [`data_engineer/README.md`](README.md) sekce *Objevování datasetů*.
