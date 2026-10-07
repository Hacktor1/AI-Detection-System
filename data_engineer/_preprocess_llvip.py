#!/usr/bin/env python3
"""
Předzpracování LLVIP dataset.

Převádí PASCAL VOC XML anotace do formátu YOLO (xywh, normalizované 0-1).
Rozděluje data na tréninkovou a testovací sadu dle adresářové struktury.
Uložené do `datasets/processed/llvip/`.

Zdroj: https://github.com/bupt-ai-cz/LLVIP
Licence: Apache 2.0
"""
import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "datasets" / "raw" / "llvip" / "LLVIP"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed" / "llvip"

CLASS_MAP = {"person": 0}


def parse_xml(xml_path):
    """Parse a PASCAL VOC XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find(".//size")
    if size is not None:
        width = int(size.find("width").text)
        height = int(size.find("height").text)
    else:
        width = 1280
        height = 1024

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
    """Convert VOC (xmin, ymin, xmax, ymax) to YOLO (xc, yc, w, h) normalized."""
    xmin, ymin, xmax, ymax = bbox
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return [x_center, y_center, w, h]


def convert_split(split_name, ir_dir, vis_dir, annot_dir, out_dir):
    """Convert a split (train/test) from VOC to YOLO format."""
    images_dir = ir_dir  # Use infrared images as primary
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])

    label_dir = out_dir / "labels" / split_name
    img_out_dir = out_dir / "images" / split_name
    label_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for img_name in image_files:
        # Corresponding XML has same name
        xml_name = img_name.replace(".jpg", ".xml")
        xml_path = annot_dir / xml_name
        if not xml_path.exists():
            continue

        img_path = images_dir / img_name
        if not img_path.exists():
            continue

        objects, w, h = parse_xml(xml_path)
        if not objects:
            continue

        # Write YOLO label
        label_path = label_dir / img_name.replace(".jpg", ".txt")
        with open(label_path, "w") as f:
            for obj in objects:
                class_id = CLASS_MAP.get(obj["class"], -1)
                if class_id == -1:
                    continue
                yolo_bbox = voc_to_yolo(obj["bbox"], w, h)
                f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                        f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

        # Copy image
        img_out = img_out_dir / img_name
        if not img_out.exists():
            shutil.copy2(img_path, img_out)

        processed += 1

    return processed


def main():
    annot_dir = RAW_DIR / "Annotations"
    ir_train = RAW_DIR / "infrared" / "train"
    ir_test = RAW_DIR / "infrared" / "test"

    print(f"[INFO] LLVIP předzpracování")
    print(f"[INFO] Raw dir: {RAW_DIR}")
    print(f"[INFO] Output dir: {PROCESSED_DIR}")

    train_count = convert_split("train", ir_train, None, annot_dir, PROCESSED_DIR)
    print(f"[INFO] train: processed {train_count} images")

    test_count = convert_split("test", ir_test, None, annot_dir, PROCESSED_DIR)
    print(f"[INFO] test: processed {test_count} images")

    # Write dataset YAML
    yaml_content = f"""# LLVIP Thermal dataset (YOLO format)
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
    print(f"[OK] Celkem zpracováno: {train_count + test_count} obrázků")


if __name__ == "__main__":
    main()
