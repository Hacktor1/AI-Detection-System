import os
import sys
from pathlib import Path

from ultralytics import YOLO


def get_project_root():
    """Return project root (repo root) — three levels up from train/."""
    return Path(__file__).resolve().parent.parent.parent


def run_training():
    """
    Hlavní tréninková funkce — používá Ultralytics YOLO Python API.

    Načítá konfiguraci z params.yaml a spouští trénink na unifikovaném datasetu.
    Výsledky se ukládají do ai_ml_architect/experiments/ jako Ultralytics experiment.
    """
    import yaml

    project_root = get_project_root()
    params_path = Path(__file__).resolve().parent / "params.yaml"

    with open(params_path) as f:
        params = yaml.safe_load(f)

    # Resolve dataset path relative to project root
    data_rel = params.get("data", "data_engineer/datasets/processed/combined/dataset.yaml")
    data_path = str(project_root / data_rel)

    if not Path(data_path).exists():
        print(f"❌ Dataset path does not exist: {data_path}")
        print("   Run data preprocessing first:")
        print("   cd data_engineer && python3 -m _preprocess_flir_adas && python3 -m _preprocess_llvip")
        sys.exit(1)

    model_name = params.get("model", "yolov8n.pt")
    epochs = params.get("epochs", 100)
    imgsz = params.get("imgsz", 640)
    batch = params.get("batch", 16)
    project_dir = str(Path(__file__).resolve().parent.parent / "experiments")
    name = params.get("name", "uav_thermal_v1")
    device = params.get("device", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

    # Load model
    print(f"🔄 Načítám model: {model_name}")
    model = YOLO(model_name)

    # Run training
    print("Spouštím trénink...")
    print(f"  Dataset: {data_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch: {batch}")
    print(f"  Device: {device}")
    print()

    model.train(
        data=data_path,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project_dir,
        name=name,
        patience=params.get("patience", 20),
        lr0=params.get("lr0", 0.01),
        lrf=params.get("lrf", 0.1),
        mosaic=params.get("mosaic", 1.0),
        mixup=params.get("mixup", 0.0),
        degrees=params.get("degrees", 0.0),
        fliplr=params.get("fliplr", 0.5),
        hsv_v=params.get("hsv_v", 0.4),
        optimizer=params.get("optimizer", "auto"),
    )

    # Summary
    print(f"\n✅ Trénnování dokončeno!")
    print(f"   Výsledky uloženy v: {Path(project_dir) / name}")
    print(f"   Nejlepší váhy: {Path(project_dir) / name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    run_training()
