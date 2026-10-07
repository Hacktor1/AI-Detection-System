#!/usr/bin/env python3
"""
Renderer module.

Draws bounding boxes and labels on frames for visualization.
"""
import cv2

try:
    from .detector import Detection
except ImportError:
    from detector import Detection


# Thermal-friendly colors (BGR)
_COLOR_MAP = {
    "person": (0, 255, 0),    # green (visible in thermal)
    "car":   (0, 165, 255),   # orange
    "dog":   (255, 0, 0),     # red
}


def _color_for(label: str) -> tuple:
    return _COLOR_MAP.get(label.lower(), (0, 255, 255))


def draw_detections(frame, detections: list[Detection]):
    """Draw bounding boxes + labels onto a copy of the frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    for det in detections:
        color = _color_for(det.label)

        # Bounding box
        x2 = min(det.x + det.w, w)
        y2 = min(det.y + det.h, h)
        cv2.rectangle(vis, (det.x, det.y), (x2, y2), color, 2)

        # Label background + text
        label_text = f"{det.label} {det.conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (det.x, det.y - th - 4), (det.x + tw, det.y), color, -1)
        cv2.putText(vis, label_text, (det.x, det.y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # FPS counter placeholder (set externally)
    cv2.putText(vis, "Drone AI — Detection Active", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis
