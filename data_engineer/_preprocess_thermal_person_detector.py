#!/usr/bin/env python3
"""
Předzpracování Thermal Person Detector dataset z HuggingFace.

Čte anotace z `samples.json` (FiftyOne formát) a převádí je do formátu YOLO.
Rozděluje data na tréninkovou a testovací sadu podle tagů v JSON.
Uložené do `datasets/processed/thermal_person_detector/`.

Zdroj: https://huggingface.co/datasets/Voxel51/Thermal-Person-Detector
Licence: CC-BY-4.0
"""
import os
import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "datasets" / "raw" / "thermal_person_detector"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed" / "thermal_person_detector"

CLASS_MAP = {"person": 0}


def samples_json_to_yolo(json_path, images_dir, out_dir):
    """Convert samples.json (FiftyOne format) to YOLO format."""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    samples = data.get("samples", [])
    if not samples:
        print(f"[WARN] No samples found in {json_path}")
        return 0, 0
    
    train_count = 0
    test_count = 0
    
    for sample in samples:
        filepath = sample.get("filepath", "")
        if not filepath:
            continue
        
        # Determine split from tags
        tags = sample.get("tags", [])
        split = "train" if "train" in tags else "test"
        
        # Full path to image
        img_path = RAW_DIR / filepath
        if not img_path.exists():
            # Try relative to data dir
            img_path = RAW_DIR / "data" / os.path.basename(filepath)
            if not img_path.exists():
                continue
        
        # Get image dimensions
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                width, height = img.size
        except ImportError:
            # Fallback without PIL
            width, height = 1280, 720  # Common thermal resolution
        except Exception:
            continue
        
        # Get annotations
        ground_truth = sample.get("ground_truth", {})
        detections = ground_truth.get("detections", [])
        
        if not detections:
            continue  # Skip images without annotations
        
        # Create output directories
        label_dir = out_dir / "labels" / split
        img_out_dir = out_dir / "images" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        img_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Write YOLO label file
        img_name = os.path.basename(filepath)
        label_name = img_name.replace(".jpg", ".txt")
        label_path = label_dir / label_name
        
        with open(label_path, "w") as f:
            for det in detections:
                label = det.get("label", "")
                bbox = det.get("bounding_box", [])
                # FiftyOne format: [x, y, w, h] already normalized 0-1 (xywh)
                if label in CLASS_MAP and len(bbox) == 4:
                    class_id = CLASS_MAP[label]
                    x, y, w, h = bbox
                    f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        # Copy image
        img_out = img_out_dir / img_name
        if not img_out.exists():
            shutil.copy2(img_path, img_out)
        
        if split == "train":
            train_count += 1
        else:
            test_count += 1
    
    return train_count, test_count


def main():
    json_path = RAW_DIR / "samples.json"
    if not json_path.exists():
        print(f"[ERROR] {json_path} not found. Download it manually:")
        print("  curl -L https://huggingface.co/datasets/Voxel51/Thermal-Person-Detector/resolve/main/samples.json \\")
        print(f"    -o {json_path}")
        return
    
    print(f"[INFO] Thermal Person Detector předzpracování")
    print(f"[INFO] Raw dir: {RAW_DIR}")
    print(f"[INFO] Output dir: {PROCESSED_DIR}")
    
    images_dir = RAW_DIR / "data"
    train_count, test_count = samples_json_to_yolo(json_path, images_dir, PROCESSED_DIR)
    
    print(f"[INFO] train: {train_count} images with annotations")
    print(f"[INFO] test: {test_count} images with annotations")
    
    # Write dataset YAML
    yaml_content = f"""# Thermal Person Detector dataset (YOLO format)
path: .
train: images/train
val: images/test

# Number of classes
nc: {len(CLASS_MAP)}

# Class names
names: {list(CLASS_MAP.keys())}
"""
    yaml_path = PROCESSED_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\n[INFO] dataset.yaml written to {yaml_path}")
    print(f"[OK] Celkem zpracováno: {train_count + test_count} obrázků s anotacemi")


if __name__ == "__main__":
    main()
