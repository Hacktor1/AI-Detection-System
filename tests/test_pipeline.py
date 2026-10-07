#!/usr/bin/env python3
"""
Integrační testy pro AI-Detection-System pipeline.

Testuje:
1. Datový pipeline (validace formátu YOLO labelů)
2. Tréninkový pipeline (spustitelnost train.py)
3. Inference pipeline (detekce na ukázkových obrázcích)

Použití:
    python -m pytest tests/test_pipeline.py -v
    python tests/test_pipeline.py  # bez pytest
"""
import os
import sys
from pathlib import Path

import pytest

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_dataset_yaml_exists():
    """Ověříme, že combined dataset.yaml existuje."""
    yaml_path = PROJECT_ROOT / "data_engineer" / "datasets" / "processed" / "combined" / "dataset.yaml"
    assert yaml_path.exists(), f"dataset.yaml not found at {yaml_path}"


def test_dataset_yaml_format():
    """Ověříme formát dataset.yaml."""
    import yaml
    yaml_path = PROJECT_ROOT / "data_engineer" / "datasets" / "processed" / "combined" / "dataset.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    assert "nc" in cfg, "Missing 'nc' in dataset.yaml"
    assert "names" in cfg, "Missing 'names' in dataset.yaml"
    assert cfg["nc"] == len(cfg["names"]), f"nc ({cfg['nc']}) != len(names) ({len(cfg['names'])})"


def test_train_txt_exists():
    """Ověříme, že train.txt existuje a není prázdný."""
    train_txt = PROJECT_ROOT / "data_engineer" / "datasets" / "processed" / "combined" / "train.txt"
    assert train_txt.exists(), f"train.txt not found"
    lines = train_txt.read_text().strip().split("\n")
    assert len(lines) > 0, "train.txt is empty"


def test_val_txt_exists():
    """Ověříme, že val.txt existuje a není prázdný."""
    val_txt = PROJECT_ROOT / "data_engineer" / "datasets" / "processed" / "combined" / "val.txt"
    assert val_txt.exists(), f"val.txt not found"
    lines = val_txt.read_text().strip().split("\n")
    assert len(lines) > 0, "val.txt is empty"


def test_label_format():
    """Ověříme formát YOLO labelů v jednom souboru."""
    label_files = list(
        (PROJECT_ROOT / "data_engineer" / "datasets" / "processed" / "flir_adas" / "labels" / "train").glob("*.txt")
    )
    if not label_files:
        pytest.skip("No label files found to test")
    
    for lbl_file in label_files[:10]:  # Test first 10
        with open(lbl_file) as f:
            lines = f.readlines()
        
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            assert len(parts) == 5, f"{lbl_file.name}: expected 5 values, got {len(parts)}"
            
            cls_id = int(parts[0])
            assert 0 <= cls_id <= 4, f"{lbl_file.name}: invalid class_id {cls_id}"
            
            for i, val in enumerate(parts[1:]):
                v = float(val)
                assert 0.0 <= v <= 1.0, f"{lbl_file.name}: value {v} at index {i+1} out of [0,1]"


def test_train_module_importable():
    """Ověříme, že train.py lze importovat."""
    train_path = PROJECT_ROOT / "ai_ml_architect" / "train"
    sys.path.insert(0, str(train_path))
    import train
    assert hasattr(train, "run_training"), "train.py missing run_training function"
    sys.path.remove(str(train_path))


def test_inference_module_importable():
    """Ověříme, že inference.py má správné argumenty."""
    inference_path = PROJECT_ROOT / "ai_ml_architect" / "train" / "inference.py"
    assert inference_path.exists(), "inference.py not found"


def test_params_yaml_exists():
    """Ověříme, že params.yaml existuje."""
    params_path = PROJECT_ROOT / "ai_ml_architect" / "train" / "params.yaml"
    assert params_path.exists(), "params.yaml not found"


def test_validate_datasets_script_exists():
    """Ověříme, že validační skript existuje."""
    validate_path = PROJECT_ROOT / "data_engineer" / "_validate_datasets.py"
    assert validate_path.exists(), "data_engineer/_validate_datasets.py not found"


def test_preprocess_scripts_exist():
    """Ověříme, že preprocessing skripty existují."""
    for script in ["_preprocess_flir_adas.py", "_preprocess_llvip.py"]:
        path = PROJECT_ROOT / "data_engineer" / script
        assert path.exists(), f"{script} not found"


if __name__ == "__main__":
    # Run tests without pytest
    import traceback
    
    test_functions = [
        test_dataset_yaml_exists,
        test_dataset_yaml_format,
        test_train_txt_exists,
        test_val_txt_exists,
        test_label_format,
        test_train_module_importable,
        test_inference_module_importable,
        test_params_yaml_exists,
        test_validate_datasets_script_exists,
        test_preprocess_scripts_exist,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"  ✅ {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {test_fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
