#!/usr/bin/env python3
"""
Inference skript pro UAV termální detekci lidí a aut.

Načítá natrénovaný YOLO model a provádí detekci na:
- Jednotlivých obrázcích
- Adresářích obrázků
- MP4 video souborech (pro drone footage)

Použití:
    python -m train.inference --weights ../models/best.pt --source ../data/sample.jpg
    python -m train.inference --weights ../models/best.pt --source ../data/images/
    python -m train.inference --weights ../models/best.pt --source ../data/video.mp4 --save-txt

Argumenty:
    --weights   Cesta k .pt souboru s natrénovanými váhami
    --source    Cesta k obrázku, adresáři, nebo video souboru
    --conf      Confidence threshold (default: 0.25)
    --iou       NMS IoU threshold (default: 0.5)
    --save-txt  Uložit YOLO txt detekce
    --save-img  Uložit obrázky s nakreslenými bounding boxy
    --classes   Filtrovat podle class ID (prolid: 0)
"""
import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="UAV Thermal Detection — Inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image, directory, or video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO txt detections")
    parser.add_argument("--save-img", action="store_true", help="Save images with bounding boxes")
    parser.add_argument("--classes", type=int, nargs="+",
                        help="Filter by class IDs (0=person, 1=car)")
    parser.add_argument("--device", type=str, default="",
                        help="GPU ID (e.g. 0) or leave empty for auto")
    return parser.parse_args()


def main():
    args = parse_args()

    # Načtení modelu
    print(f"Načítám model: {args.weights}")
    model = YOLO(args.weights)

    # Spuštění detekce
    print(f"Zdroj: {args.source}")
    print(f"Conf: {args.conf}, IoU: {args.iou}")

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=args.save_img,
        save_txt=args.save_txt,
        classes=args.classes,
        device=args.device or None,
    )

    # Shrnutí výsledků
    total_detections = sum(len(r.boxes) for r in results)
    print(f"\n✅ Dokončeno — celkem detekováno: {total_detections} objektů")

    if args.classes:
        class_names = model.names
        for cls_id in args.classes:
            cls_count = sum(
                sum(1 for box in r.boxes if int(box.cls[0]) == cls_id)
                for r in results
            )
            print(f"  - {class_names.get(cls_id, f'class_{cls_id}')}: {cls_count}")
    else:
        class_names = model.names
        for cls_id in range(len(class_names)):
            cls_count = sum(
                sum(1 for box in r.boxes if int(box.cls[0]) == cls_id)
                for r in results
            )
            print(f"  - {class_names.get(cls_id, f'class_{cls_id}')}: {cls_count}")


if __name__ == "__main__":
    main()
