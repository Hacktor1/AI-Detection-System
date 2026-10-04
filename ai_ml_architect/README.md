# AI/ML Architect Role

## Task Overview

Design, train, and evaluate the human detection model. Primary candidate: **YOLOv8** or **YOLOv11**.

## Key Activities

1. **Model Selection**
   - Compare YOLOv8-nano, YOLOv8-small, YOLOv11-nano for edge deployment.
   - Evaluate tradeoffs: accuracy vs. FPS on Jetson Nano/Orin.

2. **Training**
   - Use dataset prepared by Data Engineer.
   - Configure hyperparameters in `train/params.yaml`.
   - Train with Ultralytics YOLO API.

3. **Hyperparameter Tuning**
   - Run experiments with different mosaic, scale, and augmentation settings.
   - Log results to `experiments/` directory (or external MLflow).

4. **Evaluation**
   - Measure mAP, FPS, and thermal-specific metrics.
   - Export best model to ONNX for Edge Specialist.

## Directory Layout
```
ai_ml_architect/
├── train/
│   ├── params.yaml      # Training config
│   └── train.py         # Entry point for training
├── models/              # Model weights (gitignored)
└── experiments/         # Experiment logs
```

## Training Command
```bash
cd ai_ml_architect
python -m train.train  # After writing train.py
# OR use ultralytics CLI:
yolo detect train data=datasets.yaml model=yolov8n.pt epochs=50
```

## Export to ONNX (for Edge Specialist)
```bash
yolo export model=best.pt format=onnx
```
Pass the `.onnx` file to Edge Specialist.