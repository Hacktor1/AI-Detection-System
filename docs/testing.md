# Testovací průvodce

## Přehled testů

Projekt obsahuje **33 integračních testů** rozdělených do 3 souborů.

### Spuštění všech testů

```bash
# Bez pytest (použije vestavěný test runner)
python3 tests/test_pipeline.py     # Phase 1-2: 10 testů
python3 tests/test_phase3.py       # Phase 3: 11 testů
python3 tests/test_phase45.py      # Phase 4-5: 12 testů

# S pytest (instalace: pip install pytest)
python3 -m pytest tests/ -v        # Všechny 33 testy
```

### Co testy ověřují

| Test soubor | Fáze | Obsah |
|---|---|---|
| `test_pipeline.py` | 1-2 | Dataset YAML, label formát, importy, validace |
| `test_phase3.py` | 3 | TRT command, JetsonProfile, env check, ONNX model |
| `test_phase45.py` | 4-5 | Profiler, INT8 kalib, web UI, MAVLink, dual cam |

## Testovací dataset

Pro rychlé testy je připravený malý test dataset:

```bash
# Struktura test datasetu
data_engineer/datasets/processed/combined_test/
├── images/train/   # 100 obrázků (ze všech 3 datasetů)
├── images/val/     # 60 obrázků
├── labels/train/   # YOLO .txt anotace
├── labels/val/
└── dataset.yaml
```

## Pipeline test

```bash
# Headless test (bez GUI)
cd pipeline_engineer
python3 run_pipeline.py \
    --video sample_data/test_thermal.mp4 \
    --model ../optimized_models/best_fp16_dynamic.onnx \
    --no-display \
    --conf-thres 0.1

# Dual camera test
python3 dual_camera_pipeline.py \
    --thermal-source sample_data/test_thermal.mp4 \
    --visible-source sample_data/test_visible.mp4 \
    --model ../edge_specialist/optimized_models/best_fp16_dynamic.onnx \
    --no-display \
    --max-frames 10
```

## Benchmark test

```bash
cd edge_specialist
python3 benchmark.py \
    --onnx optimized_models/best_fp16_dynamic.onnx \
    --video ../pipeline_engineer/sample_data/test_thermal.mp4 \
    --frames 30
```

## Validation test

```bash
cd data_engineer
python3 _validate_datasets.py
```

## Očekávané výsledky

- **10/10 Phase 1-2 testy** prošly
- **11/11 Phase 3 testy** prošly
- **12/12 Phase 4-5 testy** prošely
- **Data validation**: 0 errors across 28,738 images
- **Inference speed**: 39.7 FPS (ONNX CPU), 0.004s/img (PyTorch CPU)
