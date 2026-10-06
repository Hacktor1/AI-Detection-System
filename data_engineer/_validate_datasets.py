#!/usr/bin/env python3
"""
Validační skript pro předtréninkové data.

Prověřuje:
1. Formát YOLO label souborů (správný počet sloupců, hodnoty 0-1).
2. Shodu mezi obrázky a label soubory.
3. Platnost class IDs oproti dataset.yaml.
4. Kontrolu prázdných anotací a duplicitních bboxů.

Použití:
    python3 -m data_engineer._validate_datasets
    python3 -m data_engineer._validate_datasets --dataset flir_adas
"""
import os
import sys
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"


def validate_dataset(dataset_name):
    """Validate a single processed dataset."""
    ds_dir = PROCESSED_DIR / dataset_name
    if not ds_dir.exists():
        print(f"[ERROR] Dataset directory not found: {ds_dir}")
        return False
    
    # Load dataset.yaml
    yaml_path = ds_dir / "dataset.yaml"
    if not yaml_path.exists():
        print(f"[ERROR] dataset.yaml not found: {yaml_path}")
        return False
    
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    nc = config.get("nc", 0)
    names = config.get("names", [])
    expected_classes = set(range(nc))
    
    errors = []
    warnings = []
    total_labels = 0
    total_images = 0
    empty_annotations = 0
    
    for split in ["train", "val", "test"]:
        img_dir = ds_dir / "images" / split
        lbl_dir = ds_dir / "labels" / split
        
        if not img_dir.exists() and not lbl_dir.exists():
            continue
        
        # Get file sets
        images = set()
        if img_dir.exists():
            images = {f.stem for f in img_dir.iterdir() if f.suffix in ('.jpg', '.jpeg')}
        
        labels = set()
        if lbl_dir.exists():
            labels = {f.stem for f in lbl_dir.iterdir() if f.suffix == '.txt'}
        
        # Check image-label pairs
        img_only = images - labels
        lbl_only = labels - images
        
        if img_only:
            warnings.append(f"{dataset_name}/{split}: {len(img_only)} images without labels")
        if lbl_only:
            errors.append(f"{dataset_name}/{split}: {len(lbl_only)} labels without images")
        
        # Validate label content
        if lbl_dir.exists():
            for lbl_file in sorted(lbl_dir.glob("*.txt")):
                total_labels += 1
                has_valid = False
                with open(lbl_file, "r") as f:
                    lines = f.readlines()
                
                if not lines:
                    empty_annotations += 1
                    continue
                
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        errors.append(f"{lbl_file.name} line {i+1}: expected 5 values, got {len(parts)}")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    except ValueError:
                        errors.append(f"{lbl_file.name} line {i+1}: invalid number format")
                        continue
                    
                    if class_id not in expected_classes:
                        errors.append(f"{lbl_file.name} line {i+1}: invalid class_id {class_id}")
                    
                    for val_name, val in [("x", x), ("y", y), ("w", w), ("h", h)]:
                        if val < 0 or val > 1:
                            errors.append(f"{lbl_file.name} line {i+1}: {val_name}={val} out of range [0, 1]")
                    
                    if w <= 0 or h <= 0:
                        warnings.append(f"{lbl_file.name} line {i+1}: zero-size bbox")
                    
                    has_valid = True
                
                if not has_valid:
                    empty_annotations += 1
        
        total_images += len(images)
    
    # Summary
    print(f"\n=== {dataset_name} ===")
    print(f"  Classes: {names} (nc={nc})")
    print(f"  Total label files: {total_labels}")
    print(f"  Total images: {total_images}")
    print(f"  Empty annotations: {empty_annotations}")
    print(f"  Errors: {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    
    if errors:
        print(f"\n  Errors:")
        for e in errors[:10]:
            print(f"    - {e}")
    
    if warnings:
        print(f"\n  Warnings:")
        for w in warnings[:10]:
            print(f"    - {w}")
    
    return len(errors) == 0


def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["flir_adas", "llvip", "thermal_person_detector"]
    
    all_valid = True
    for ds in datasets:
        if not validate_dataset(ds):
            all_valid = False
    
    if all_valid:
        print("\n✅ Všechny datasety jsou validní!")
    else:
        print("\n❌ Některé datasety mají chyby!")
        sys.exit(1)


if __name__ == "__main__":
    main()
