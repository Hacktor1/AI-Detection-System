#!/usr/bin/env python3
"""
Detection inference wrapper.

Wraps a YOLO model (Ultralytics) and runs inference on a single frame.
Returns detections in a standardized format for the renderer.

Model can be a .pt (PyTorch), .onnx, or .engine (TensorRT) file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Detector results (shared structure)
@dataclass
class Detection:
    """A single detected object."""
    x: int       # top-left x
    y: int       # top-left y
    w: int       # width
    h: int       # height
    conf: float  # confidence score
    class_id: int = 0  # default: "person"
    label: str = "person"


class PersonDetector:
    """YOLO-based person detector."""

    def __init__(self, model_path: str = "", conf_thres: float = 0.4):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.model = None
        self._load_model()

    def _load_model(self):
        if not self.model_path or not os.path.isfile(self.model_path):
            print("[detector] Warning: No model file found. Detection stub active.")
            return

        ext = os.path.splitext(self.model_path)[1].lower()

        if ext == ".pt":
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                print(f"[detector] Loaded PyTorch model: {self.model_path}")
            except ImportError:
                raise ImportError("ultralytics not installed. Run: pip install -r requirements.txt")

        elif ext == ".onnx":
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                print(f"[detector] Loaded ONNX model: {self.model_path}")
            except ImportError:
                raise ImportError("ultralytics not installed. Run: pip install -r requirements.txt")

        elif ext == ".engine":
            self._load_tensorrt_engine()

        else:
            raise ValueError(f"Unsupported model format: {ext}")

    def _load_tensorrt_engine(self):
        """Stub: load TensorRT engine on Jetson."""
        try:
            import tensorrt as trt
            self.model = "tensorrt"
            print(f"[detector] TensorRT engine (stub): {self.model_path}")
        except ImportError:
            print("[detector] Warning: tensorrt not available. Engine inference disabled.")
            self.model = None

    def infer(self, frame) -> list[Detection]:
        """Run inference on a frame. Returns list of Detection objects."""
        if frame is None:
            return []

        if self.model is None:
            # Fallback: no model — return empty list (visualization still works)
            return []

        # YOLO models handle internally
        if hasattr(self.model, "predict"):
            results = self.model.predict(frame, conf=self.conf_thres, verbose=False)[0]
            boxes = []
            for box in results.boxes:
                cls = int(box.cls[0])
                label = self.model.names.get(cls, "unknown")
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes.append(Detection(
                    x=int(x1), y=int(y1),
                    w=int(x2 - x1), h=int(y2 - y1),
                    conf=conf, class_id=cls, label=label
                ))
            return boxes

        return []
