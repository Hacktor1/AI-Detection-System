#!/usr/bin/env python3
"""
Předzpracování FLIR ADAS Aligned dataset.

Převádí PASCAL VOC XML anotace do formátu YOLO (xywh, normalizované 0-1).
Rozděluje data na tréninkovou a validační sadu podle `align_train.txt` a `align_validation.txt`.
Uložené do `datasets/processed/flir_adas/`.

Zdroj: https://drive.google.com/file/d/1xHDMGl6HJZwtarNWkEV3T4O9X4ZQYz2Y
Licence: Bezplatná pro výzkum
"""
import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "datasets" / "raw" / "flir_adas" / "align"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed" / "flir_adas"

# Class mapping - keep all classes FLIR ADAS provides
CLASS_MAP = {
    "person": 0,
    "car": 1,
    "bicycle": 2,
    "dog": 3,
    "other_vehicle": 4,
}

def parse_xml(xml_path):
    """Parse a PASCAL VOC XML file and return list of objects."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find(".//size")
    if size is not None:
        width = int(size.find("width").text)
        height = int(size.find("height").text)
    else:
        # Fallback - FLIR images are 640x512
        width = 640
        height = 512
    
    objects = []
    for obj in root.findall(".//object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        if bndbox is not None:
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            
            objects.append({
                "class": name,
                "bbox": [xmin, ymin, xmax, ymax],
                "width": width,
                "height": height
            })
    
    return objects, width, height


def voc_to_yolo(bbox, width, height):
    """
    Convert bounding box from VOC format (xmin, ymin, xmax, ymax)
    to YOLO format (x_center, y_center, width, height) normalized to 0-1.
    """
    xmin, ymin, xmax, ymax = bbox
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return [x_center, y_center, w, h]


def process_split(split_name, split_file, annotations_dir, images_dir, output_dir):
    """Process a train/val/test split."""
    split_path = RAW_DIR / split_file
    
    if not split_path.exists():
        print(f"[WARN] Split file {split_file} not found, skipping.")
        return 0
    
    with open(split_path, "r") as f:
        image_names = [line.strip() for line in f if line.strip()]
    
    processed = 0
    for name in image_names:
        # Split file entries are like "FLIR_00258_PreviewData" — match without adding _PreviewData again
        xml_path = annotations_dir / f"{name}.xml"
        if not xml_path.exists():
            continue
        
        # Find corresponding image
        img_path = images_dir / f"{name}.jpeg"
        if not img_path.exists():
            continue
        
        # Parse and convert
        objects, w, h = parse_xml(xml_path)
        if not objects:
            continue
        
        # Write YOLO label file
        label_dir = output_dir / "labels" / split_name
        label_dir.mkdir(parents=True, exist_ok=True)
        
        label_path = label_dir / f"{name}.txt"
        with open(label_path, "w") as f:
            for obj in objects:
                class_id = CLASS_MAP.get(obj["class"], -1)
                if class_id == -1:
                    # Skip unknown classes
                    continue
                yolo_bbox = voc_to_yolo(obj["bbox"], w, h)
                f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                        f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
        
        # Copy image to processed directory
        img_out = output_dir / "images" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        img_out_path = img_out / f"{name}.jpeg"
        if not img_out_path.exists():
            shutil.copy2(img_path, img_out_path)
        
        processed += 1
    
    return processed


def main():
    annotations_dir = RAW_DIR / "Annotations"
    images_dir = RAW_DIR / "JPEGImages"
    
    # Also check AnnotatedImages as fallback
    annotated_dir = RAW_DIR / "AnnotatedImages"
    
    splits = [
        ("train", "align_train.txt"),
        ("val", "align_validation.txt"),
    ]
    
    print(f"[INFO] FLIR ADAS předzpracování")
    print(f"[INFO] Raw dir: {RAW_DIR}")
    print(f"[INFO] Output dir: {PROCESSED_DIR}")
    
    total = 0
    for split_name, split_file in splits:
        count = process_split(split_name, split_file, annotations_dir, images_dir, PROCESSED_DIR)
        print(f"[INFO] {split_name}: processed {count} images")
        total += count
    
    # Write dataset YAML for YOLO
    yaml_content = f"""# FLIR ADAS Thermal dataset (YOLO format)
path: .
train: images/train
val: images/val

# Number of classes
nc: {len(CLASS_MAP)}

# Class names
names: {list(CLASS_MAP.keys())}
"""
    yaml_path = PROCESSED_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\n[INFO] dataset.yaml written to {yaml_path}")
    print(f"[OK] Celkem zpracováno: {total} obrázků")


if __name__ == "__main__":
    main()
