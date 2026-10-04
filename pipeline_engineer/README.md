# Rolle: Pipeline Inženýr

## Přehled úlohy

Vytvořit hlavní video pipeline, která načítá termální video z kamer, spouští detekci a vykresluje bounding boxy. Je to integrační bod pro Data Engineera (datasety), AI/ML Architekta (model) a Edge Specialista (optimalizovaný model).

## Klíčové aktivity

1. **Obsluha vstupu z kamer** (`camera_io.py`)
   - Číst z video souborů (pro testování).
   - Rozhraní pro dual camera streamy.
   - Později: připojení k Jetson CSI/MIPI kameře.

2. **Inferenční detekce** (`detector.py`)
   - Načíst YOLO model (ONNX nebo PyTorch).
   - Spouštět inferenci na snímcích.
   - Vracet bounding boxy, skóre, třídy.

3. **Vykreslování** (`renderer.py`)
   - Overlay bounding boxů na termální snímek.
   - Zpracování vizualizace dvou kamer.

4. **Hlavní smyčka** (`run_pipeline.py`)
   - Orchestrace: zachytřit → detekovat → vykreslit.
   - CLI argumenty pro vstupní soubor, cestu k modelu, výstupní adresář.

## Struktura adresářů
```
pipeline_engineer/
├── camera_io.py       # Čtečka kamer/streamerů
├── detector.py        # Obalový interface pro inferenci modelu
├── renderer.py        # Kreslič bounding boxů
├── run_pipeline.py    # Hlavní vstupní bod pipeline
└── sample_data/       # Malé ukázkové videa pro testování (gitignored)
```

## Rychlý test
```bash
cd pipeline_engineer
python run_pipeline.py --video sample_data/test_thermal.mp4 --model ../ai_ml_architect/models/best.onnx
```

## Poznámky
- Pro počáteční testování funguje jakýkoli MP4 jako vstup.
- Model může být `.engine` (TensorRT) při nasazení na Jetsonu.