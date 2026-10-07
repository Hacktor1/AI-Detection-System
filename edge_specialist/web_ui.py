#!/usr/bin/env python3
"""
Web UI pro vzdálený monitoring AI detekčního systému na dronu.

Poskytuje:
- Live video stream s bounding boxy (MJPEG)
- JSON API pro detekční výsledky
- Dashboard s FPS, GPU využitím a zprávami o detekcích
- Statistiky dronu (baterie, altituda, pozice)

Použití:
    python3 -m edge_specialist.web_ui --host 0.0.0.0 --port 8080
    python3 -m edge_specialist.web_ui --host 0.0.0.0 --port 8080 --source /dev/video0
"""
import argparse
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, Response, jsonify, render_template_string

app = Flask(__name__)

# Global state — shared between capture thread and web requests
class AppState:
    def __init__(self):
        self.frame = None
        self.boxes = []
        self.fps = 0.0
        self.frame_count = 0
        self.last_update = time.time()
        self.running = True
        self.model = None
        self.drone_state = {
            "battery_pct": 0,
            "altitude_m": 0.0,
            "latitude": 0.0,
            "longitude": 0.0,
            "heading_deg": 0,
        }

state = AppState()


def parse_args():
    parser = argparse.ArgumentParser(description="Drone AI Detection Web UI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Listen address")
    parser.add_argument("--port", type=int, default=8080, help="Listen port")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source (0=camera, path=video file)")
    parser.add_argument("--model", type=str, default="",
                        help="Model path (.pt/.onnx/.engine)")
    parser.add_argument("--conf-thres", type=float, default=0.4, help="Confidence threshold")
    return parser.parse_args()


def load_model(model_path):
    """Load detection model."""
    if not model_path or not os.path.isfile(model_path):
        print("[web_ui] No model — detections disabled (video only)")
        return None

    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".pt":
        from ultralytics import YOLO
        return YOLO(model_path)
    elif ext == ".onnx":
        from ultralytics import YOLO
        return YOLO(model_path)
    elif ext == ".engine":
        print("[web_ui] TensorRT engine — requires Jetson deployment")
        return None
    else:
        print(f"[web_ui] Unknown model format: {ext}")
        return None


def capture_loop(args):
    """Background thread: capture frames, run detection, update state."""
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        print(f"[web_ui] ERROR: Cannot open video source {args.source}")
        state.running = False
        return

    model = load_model(args.model)
    frame_times = []
    last_time = time.time()

    while state.running:
        ret, frame = cap.read()
        if not ret:
            print("[web_ui] End of stream, looping...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Run detection
        boxes = []
        if model is not None and hasattr(model, "predict"):
            try:
                results = model.predict(frame, conf=args.conf_thres, verbose=False)[0]
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_name = model.names.get(cls_id, "unknown")
                    boxes.append({
                        "class": cls_name,
                        "class_id": cls_id,
                        "confidence": round(conf, 3),
                        "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    })
            except Exception as e:
                print(f"[web_ui] Detection error: {e}")
                boxes = []

        # Update state
        state.frame = frame
        state.boxes = boxes
        state.frame_count += 1

        now = time.time()
        dt = now - last_time
        if dt > 0:
            state.fps = 1.0 / dt
        last_time = now

        # Simulate drone telemetry (would come from MAVLink in production)
        if state.drone_state["battery_pct"] > 0:
            state.drone_state["battery_pct"] = max(0, state.drone_state["battery_pct"] - 0.01)

    cap.release()


# ------------------------------------------------------------------
# Web routes
# ------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Drone AI — Thermal Detection</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Segoe UI', sans-serif; margin: 0; background: #1a1a2e; color: #eee; }
        .header { background: #16213e; padding: 1rem; text-align: center; }
        .header h1 { margin: 0; color: #00d4ff; }
        .dashboard { display: grid; grid-template-columns: 2fr 1fr; gap: 1rem; padding: 1rem; }
        .video-panel { text-align: center; }
        .video-panel img { max-width: 100%; border: 2px solid #0f3460; border-radius: 8px; }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
        .stat-card { background: #16213e; padding: 1rem; border-radius: 8px; text-align: center; }
        .stat-card .label { font-size: 0.8rem; color: #888; }
        .stat-card .value { font-size: 1.8rem; font-weight: bold; color: #00d4ff; }
        .detections { list-style: none; padding: 0; }
        .detections li { background: #0f3460; margin: 0.3rem 0; padding: 0.5rem;
                        border-radius: 4px; border-left: 3px solid #e94560; }
        .detections .cls { color: #e94560; font-weight: bold; }
        .detections .conf { float: right; color: #00d4ff; }
        .telemetry { background: #16213e; padding: 1rem; border-radius: 8px; margin-top: 1rem; }
        .telemetry .row { display: flex; justify-content: space-between; padding: 0.3rem 0; }
        .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px;
                font-size: 0.7rem; font-weight: bold; }
        .badge-person { background: #00d4ff; color: #000; }
        .badge-car { background: #e94560; color: #fff; }
        .badge-other { background: #888; color: #fff; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛸 Drone AI — Thermal Detection</h1>
        <div>Status: <span id="status">Connecting...</span></div>
    </div>
    <div class="dashboard">
        <div class="video-panel">
            <img src="/stream.mjpeg" alt="Live Feed" id="video-feed"
                 style="width: 100%; height: auto;">
            <div style="margin-top: 0.5rem;">
                <span class="badge-person">person</span>
                <span class="badge-car">car</span>
                <span class="badge-other">other</span>
            </div>
        </div>
        <div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="label">FPS</div>
                    <div class="value" id="fps">0.0</div>
                </div>
                <div class="stat-card">
                    <div class="label">Detections</div>
                    <div class="value" id="detections">0</div>
                </div>
                <div class="stat-card">
                    <div class="label">Frames Processed</div>
                    <div class="value" id="frames">0</div>
                </div>
                <div class="stat-card">
                    <div class="label">Battery</div>
                    <div class="value" id="battery">--%</div>
                </div>
            </div>
            <div class="telemetry">
                <div class="row"><span>Altitude:</span> <span id="alt">-- m</span></div>
                <div class="row"><span>Position:</span> <span id="pos">--, --</span></div>
                <div class="row"><span>Heading:</span> <span id="heading">--°</span></div>
                <div class="row"><span>Last Detection:</span> <span id="last-det">--</span></div>
            </div>
            <ul class="detections" id="det-list">
                <li style="color: #888;">Waiting for detections...</li>
            </ul>
        </div>
    </div>

    <script>
        async function updateStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();

                document.getElementById('fps').textContent = data.fps.toFixed(1);
                document.getElementById('detections').textContent = data.total_detections;
                document.getElementById('frames').textContent = data.frame_count;

                // Drone telemetry
                const dt = data.drone_state;
                document.getElementById('battery').textContent =
                    dt.battery_pct > 0 ? dt.battery_pct.toFixed(0) + '%' : '--%';
                document.getElementById('alt').textContent = dt.altitude_m.toFixed(1) + ' m';
                document.getElementById('pos').textContent =
                    dt.latitude.toFixed(6) + ', ' + dt.longitude.toFixed(6);
                document.getElementById('heading').textContent = dt.heading_deg.toFixed(0) + '°';
                document.getElementById('last-det').textContent = data.last_detection || '--';

                // Detection list
                const list = document.getElementById('det-list');
                if (data.boxes && data.boxes.length > 0) {
                    list.innerHTML = data.boxes.map(b =>
                        `<li><span class="cls">${b.class}</span>
                         <span class="conf">${b.confidence.toFixed(2)}</span></li>`
                    ).join('');
                } else {
                    list.innerHTML = '<li style="color:#888;">No detections</li>';
                }
            } catch(e) {
                document.getElementById('status').textContent = 'Error: ' + e.message;
            }
        }
        setInterval(updateStats, 200);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/stream.mjpeg")
def stream():
    """MJPEG video stream with bounding boxes."""
    def gen():
        while state.running:
            if state.frame is not None:
                # Draw boxes on frame
                vis = state.frame.copy()
                for box in state.boxes:
                    x, y, w, h = box["bbox"]
                    label = f"{box['class']} {box['confidence']:.2f}"
                    color = (0, 255, 0) if box["class"] == "person" else (0, 165, 255)
                    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(vis, f"FPS: {state.fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                ret, jpeg = cv2.imencode(".jpg", vis)
                if ret:
                    buf = io.BytesIO(jpeg.tobytes())
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + buf.getvalue() + b"\r\n")
                time.sleep(0.03)
            else:
                time.sleep(0.1)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def stats():
    """JSON stats endpoint."""
    total_det = len(state.boxes)
    last_det = ""
    if state.boxes:
        last = state.boxes[0]
        last_det = f"{last['class']} ({last['confidence']:.2f})"

    return jsonify({
        "fps": round(state.fps, 1) if state.fps else 0.0,
        "frame_count": state.frame_count,
        "total_detections": total_det,
        "boxes": state.boxes,
        "last_detection": last_det,
        "drone_state": state.drone_state,
    })


@app.route("/api/telemetry")
def telemetry():
    """Simulate MAVLink telemetry data."""
    # In production, this would read from a MAVLink connection
    return jsonify(state.drone_state)


def main():
    args = parse_args()

    # Start capture thread
    t = threading.Thread(target=capture_loop, args=(args,), daemon=True)
    t.start()

    print(f"\n🌐 Web UI: http://{args.host}:{args.port}")
    print(f"   Video: {args.model or 'no model (video only)'}")
    print(f"   Source: {args.source}")
    print(f"   Press Ctrl+C to stop\n")

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        state.running = False


if __name__ == "__main__":
    main()
