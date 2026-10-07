#!/usr/bin/env python3
"""
Mobile Stream Bridge pro AI-Detection-System.

Nahrazuje MAVLink bridge novým přístupem:
- Detekční systém na dronu (Raspberry Pi / Jetson Nano)
- Streamuje video + detekce přímo přes HTTP/WebSocket do mobilu
- Nenahrazuje dron ovladač — jen poskytuje vizualizaci detekcí

Architektura:
┌─────────┐      Wi-Fi      ┌──────────────┐    HTTP/WebSocket    ┌─────────┐
│  Dron   │◄────────────────┤  Mobile      │◄─────────────────────┤  Mobil  │
│ (Pi/Jetson) │               │  Stream      │                        │         │
│ - kamera   │   RTSP/MJPEG    │  Server      │  (Flask + WebRTC)     │  app    │
│ - ONNX    │◄──detekce───►   │  - přijímá   │◄──video+boxes───►     │  (Web)   │
│  model    │                  │  video       │                        │         │
│           │                  │  - spouští   │                        │         │
│           │                  │  inference   │                        │         │
│           │                  │  - streamuje │                        │         │

Použití:
    # Na dronu (Jetson/PI) — streamuje video
    python3 mobile_stream.py \
        --camera /dev/video0 \
        --model optimized_models/best_fp16_dynamic.onnx \
        --port 8080 \
        --host 0.0.0.0

    # Výstup na mobil (libovolný web prohlížeč):
    http://<drone-ip>:8080/

    # Debug/test na PC
    python3 mobile_stream.py \
        --video pipeline_engineer/sample_data/test_thermal.mp4 \
        --model optimized_models/best_fp16_dynamic.onnx \
        --no-telemetry
"""
import argparse
import base64
import io
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, Response, jsonify, render_template_string, request

app = Flask(__name__)

# ------------------------------------------------------------------
# Globální stav serveru
# ------------------------------------------------------------------
class ServerState:
    """Stav streamovacího serveru — sdílený mezi capture threadem a web requesty."""
    def __init__(self):
        self.frame = None                  # Poslední zpracovaný frame
        self.frame_raw = None               # Surový frame (pro nízkou latenci)
        self.boxes = []                    # Detekce na aktuálním frame
        self.fps = 0.0                     # Aktuální FPS
        self.frame_count = 0
        self.total_detections = 0
        self.last_detection_ts = None
        self.running = True
        self.model = None
        self.detection_history = []          # Posledních 50 detekcí
        self.telemetry = {
            "battery_pct": 98.0,
            "altitude_m": 12.5,
            "latitude": 50.0755,           # Praha
            "longitude": 14.4378,
            "heading_deg": 45.0,
        }
        self.model_fps = 0.0
        self.model_latency_ms = 0.0


state = ServerState()


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
def load_model(model_path: str):
    """Načte ONNX/TensorRT/PyTorch model pro inference."""
    if not model_path or not os.path.isfile(model_path):
        print("[mobile_stream] Žádný model — video only (bez detekce)")
        return None

    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".engine":
        print("[mobile_stream] TensorRT engine — vyžaduje Jetson nasazení")
        return None
    elif ext in (".onnx", ".pt"):
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            print(f"[mobile_stream] Model loaded: {model_path} ({os.path.getsize(model_path)//1024}KB)")
            return model
        except Exception as e:
            print(f"[mobile_stream] Model load error: {e}")
            return None
    else:
        print(f"[mobile_stream] Neznámý formát: {ext}")
        return None


# ------------------------------------------------------------------
# Kamera / Video input
# ------------------------------------------------------------------
def open_capture(source: str):
    """Otevře video zdroj (kamera nebo soubor)."""
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[mobile_stream] ERROR: Nemohu otevřít zdroj {source}")
        return None

    # Pro kamery nastav frame size
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"[mobile_stream] Video zdroj otevřen: {source}")
    return cap


# ------------------------------------------------------------------
# Capture + inference thread
# ------------------------------------------------------------------
def capture_loop(args):
    """Background thread: capture → detect → update state."""
    cap = open_capture(args.source)
    if not cap:
        state.running = False
        return

    model = load_model(args.model)
    frame_times = []
    last_time = time.time()
    last_detections = []

    while state.running:
        ret, frame = cap.read()
        if not ret:
            if args.loop and isinstance(args.source, str) and args.source.rsplit(".", 1)[-1] in ("mp4", "avi", "mkv"):
                print("[mobile_stream] Konec videa, loop zpět na začátek")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[mobile_stream] Konec streamu")
                break

        # Resize pro kontrolu velikosti (YOLO akceptuje libovolné)
        display_frame = frame.copy()

        # Inference
        boxes = []
        if model is not None and hasattr(model, "predict"):
            t0 = time.time()
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
                state.model_latency_ms = round((time.time() - t0) * 1000, 1)
            except Exception as e:
                print(f"[mobile_stream] Inference error: {e}")

        # Update state
        state.frame = display_frame
        state.boxes = boxes
        state.frame_count += 1
        state.total_detections += len(boxes)

        now = time.time()
        dt = now - last_time
        if dt > 0:
            state.fps = 1.0 / dt
        last_time = now

        # Telemetry sim (pouze pokud není reálný drone)
        if not args.no_telemetry:
            _update_telemetry(dt)

        # Detection history (posledních 50)
        for b in boxes:
            entry = {
                "class": b["class"],
                "confidence": b["confidence"],
                "timestamp": datetime.now().isoformat(),
            }
            state.detection_history.append(entry)
            state.last_detection_ts = entry["timestamp"]
            if len(state.detection_history) > 50:
                state.detection_history.pop(0)

        last_detections = boxes

        # Throttle: ~30 FPS
        time.sleep(max(0, 1/30 - (time.time() - now)))

    cap.release()
    print("[mobile_stream] Capture thread ukončen")


def _update_telemetry(dt: float):
    """Aktualizuje telemetry pro simulaci (na skutečném dronu by to přišlo přes MAVLink)."""
    state.telemetry["battery_pct"] = max(0, state.telemetry["battery_pct"] - 0.01 * dt)
    state.telemetry["altitude_m"] = round(state.telemetry["altitude_m"] + 0.01 * dt, 1)
    # Simulace pohybu
    state.telemetry["latitude"] += 0.0001 * math.sin(time.time() * 0.1)
    state.telemetry["longitude"] += 0.0001 * math.cos(time.time() * 0.1)
    state.telemetry["heading_deg"] = (state.telemetry["heading_deg"] + 0.1 * dt) % 360


# ------------------------------------------------------------------
# Web routes
# ------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="cs">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    <title>Drone AI — Detekční systém</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #0f0f23;
            color: #e0e0e0;
            overflow-x: hidden;
        }
        .status-bar {
            background: #1a1a2e;
            padding: 0.5rem 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
        }
        .status-dot { height: 10px; width: 10px; border-radius: 50%; background: #00ff88; display: inline-block; margin-right: 5px; }
        .header {
            background: #16213e;
            padding: 0.75rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { color: #00d4ff; font-size: 1.3rem; }
        .header .subtitle { color: #888; font-size: 0.8rem; }
        .video-container {
            position: relative;
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
            background: #000;
        }
        #video-feed {
            width: 100%;
            height: auto;
            display: block;
        }
        .stats-overlay {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(26, 31, 48, 0.8);
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            font-size: 0.8rem;
        }
        .stats-overlay .stat { display: flex; justify-content: space-between; gap: 10px; }
        .stats-overlay .stat .label { color: #888; }
        .stats-overlay .stat .value { color: #00d4ff; font-weight: bold; }
        .bottom-panel {
            padding: 1rem;
            max-width: 800px;
            margin: 0 auto;
        }
        .telemetry-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        .telemetry-card {
            background: #16213e;
            border-radius: 8px;
            padding: 0.75rem;
            text-align: center;
        }
        .telemetry-card .label { font-size: 0.7rem; color: #888; text-transform: uppercase; }
        .telemetry-card .value { font-size: 1.2rem; font-weight: bold; color: #00d4ff; }
        .detections {
            background: #16213e;
            border-radius: 8px;
            padding: 0.75rem;
            list-style: none;
            max-height: 200px;
            overflow-y: auto;
        }
        .detections li {
            padding: 0.4rem 0;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
        }
        .detections .cls { color: #e94560; font-weight: bold; text-transform: capitalize; }
        .detections .conf { color: #00d4ff; }
        .detections .ts { color: #888; font-size: 0.7rem; }
        .empty-msg { color: #555; text-align: center; padding: 1rem; }
    </style>
</head>
<body>
    <div class="status-bar">
        <span><span class="status-dot"></span>Online</span>
        <span id="connection-status">Spouštím...</span>
    </div>
    <div class="header">
        <div>
            <h1>🛸 Drone AI Detekční systém</h1>
            <div class="subtitle">AI detekce lidí a aut (termální)</div>
        </div>
        <div style="text-align: right; font-size: 0.8rem; color: #888;">
            <div id="connect-info">---.---.---.---</div>
        </div>
    </div>
    <div class="video-container">
        <img src="/stream.mjpeg" alt="Live Feed" id="video-feed">
        <div class="stats-overlay">
            <div class="stat"><span class="label">Detekce:</span> <span class="value" id="det-count">0</span></div>
            <div class="stat"><span class="label">FPS:</span> <span class="value" id="live-fps">0.0</span></div>
            <div class="stat"><span class="label">Latence:</span> <span class="value" id="latency">0.0 ms</span></div>
        </div>
    </div>
    <div class="bottom-panel">
        <div class="telemetry-grid">
            <div class="telemetry-card"><div class="label">Baterie</div><div class="value" id="bat">--%</div></div>
            <div class="telemetry-card"><div class="label">Altituda</div><div class="value" id="alt">-- m</div></div>
            <div class="telemetry-card"><div class="label">Pozice</div><div class="value" id="pos">--, --</div></div>
            <div class="telemetry-card"><div class="label">Nálož</div><div class="value" id="heading">--°</div></div>
        </div>
        <ul class="detections" id="det-list">
            <li class="empty-msg">Čekání na detekce...</li>
        </ul>
    </div>
<script>
let reconnectAttempts = 0;
const maxReconnects = 10;

function updateStats() {
    fetch('/api/stats')
        .then(r => r.json())
        .then(data => {
            document.getElementById('det-count').textContent = data.total_detections;
            document.getElementById('live-fps').textContent = data.fps.toFixed(1);
            document.getElementById('latency').textContent = data.model_latency_ms || 0;
            document.getElementById('bat').textContent = data.drone_state.battery_pct.toFixed(0) + '%';
            document.getElementById('alt').textContent = data.drone_state.altitude_m.toFixed(1) + ' m';
            document.getElementById('pos').textContent = data.drone_state.latitude.toFixed(5) + ', ' + data.drone_state.longitude.toFixed(5);
            document.getElementById('heading').textContent = data.drone_state.heading_deg.toFixed(0) + '°';

            // Detekce
            const list = document.getElementById('det-list');
            if (data.boxes && data.boxes.length > 0) {
                list.innerHTML = data.boxes.map(b =>
                    '<li><span class="cls">' + b.class + '</span>' +
                    '<span class="conf">' + b.confidence.toFixed(2) + '</span>' +
                    '<span class="ts">' + b.timestamp || '' + '</span></li>'
                ).join('');
            } else {
                list.innerHTML = '<li class="empty-msg">Žádné detekce</li>';
            }
        })
        .catch(err => {
            document.getElementById('connection-status').textContent = 'Offline - retrying...';
            reconnectAttempts++;
        });
}

// Poll stats
setInterval(updateStats, 300);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/stream.mjpeg")
def stream():
    """MJPEG video stream s bounding boxy."""
    def gen():
        while state.running:
            if state.frame is not None:
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
    last_det = ""
    if state.boxes:
        last = state.boxes[0]
        last_det = f"{last['class']} ({last['confidence']:.2f})"

    return jsonify({
        "fps": round(state.fps, 1) if state.fps else 0.0,
        "frame_count": state.frame_count,
        "total_detections": state.total_detections,
        "boxes": state.boxes,
        "last_detection": last_det,
        "drone_state": state.telemetry,
        "model_latency_ms": state.model_latency_ms,
        "detection_history": state.detection_history[-20:],
    })


@app.route("/api/telemetry")
def telemetry():
    """Telemetry endpoint."""
    return jsonify(state.telemetry)


@app.route("/api/detections")
def detections():
    """Detection history endpoint."""
    return jsonify({
        "recent": state.detection_history[-20:],
        "current": state.boxes,
    })


@app.route("/api/system")
def system():
    """System info endpoint."""
    return jsonify({
        "model_path": str(state.model),
        "model_size_kb": os.path.getsize(args.model) // 1024 if args.model and os.path.isfile(args.model) else 0,
        "hostname": os.uname().nodename,
        "platform": sys.platform,
    })


def signal_handler(signum, frame):
    print("\n[mobile_stream] Shutting down...")
    state.running = False
    sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Mobile Stream Bridge for drone AI detection")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Listen address")
    parser.add_argument("--port", type=int, default=8080, help="Listen port")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source (0=camera, path=video file)")
    parser.add_argument("--model", type=str, default="",
                        help="Model path (.pt/.onnx/.engine)")
    parser.add_argument("--conf-thres", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--loop", action="store_true", help="Loop video file")
    parser.add_argument("--no-telemetry", action="store_true",
                        help="Disable simulated telemetry (real drone provides its own)")
    return parser.parse_args()


args = None


def main():
    global args
    args = parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start capture thread
    t = threading.Thread(target=capture_loop, args=(args,), daemon=True)
    t.start()

    print(f"\n🌐 Mobile Stream: http://{args.host}:{args.port}")
    print(f"   Model: {args.model or 'none (video only)'}")
    print(f"   Source: {args.source}")
    print(f"   Telemetry: {'disabled' if args.no_telemetry else 'simulated'}")
    print(f"   Press Ctrl+C to stop\n")

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True,
                use_reloader=False)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"[mobile_stream] Port {args.port} is already in use. Try --port 8081")
        else:
            raise
    except KeyboardInterrupt:
        state.running = False


if __name__ == "__main__":
    main()
