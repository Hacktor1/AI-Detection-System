import os
import shutil
import subprocess
import sys
from pathlib import Path
import yaml

def ensure_dirs():
    """Vytvoří potřebné adresáře pro trénink."""
    for d in ["experiments", "models"]:
        Path(d).mkdir(exist_ok=True)

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def run_training():
    """Hlavní tréninková funkce — volá Ultralytics YOLO API."""
    params = load_params()
    ensure_dirs()

    model_name = params.get("model", "yolov8n.pt")
    data_path = params.get("data", "../data_engineer/datasets/processed/combined/dataset.yaml")
    epochs = params.get("epochs", 50)
    imgsz = params.get("imgsz", 640)
    batch = params.get("batch", 16)
    project = params.get("project", "experiments")
    name = params.get("name", "uav_thermal_v1")
    device = params.get("device", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    name = params.get("name", "uav_thermal_v1")
    device = params.get("device", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    name = params.get("name", "uav_thermal_v1")
    device = params.get("device", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

    cmd = [
        "python", "-m", "yolo", "detect", "train",
        f"data={data_path}",
        f"model={model_name}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"project={project}",
        f"name={name}",
        f"device={device}",
        f"patience={params.get('patience', 10)}",
        f"lr0={params.get('lr0', 0.01)}",
        f"lrf={params.get('lrf', 0.1)}",
        f"mosaic={params.get('mosaic', 1.0)}",
        f"mixup={params.get('mixup', 0.0)}",
        f"degrees={params.get('degrees', 0.0)}",
    ]

    print("Spouštím trénink:")
    print("  " + " ".join(cmd))
    print()

    try:
        subprocess.run(cmd, check=True, cwd=str(Path(__file__).parent))
        print("\n✅ Trénnování dokončeno úspěšně!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Trénnování selhalo s chybou: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_training()
