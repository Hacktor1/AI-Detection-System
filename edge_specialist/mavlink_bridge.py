#!/usr/bin/env python3
"""
MAVLink bridge pro AI-Detection-System.

Poskytuje:
- Telemetry feed (altituda, pozice, baterie, heading) do Web UI
- Detekční výsledky jako MAVLink messages pro ground station
- Heartbeat a status reporty

Funguje v dvou režimech:
1. --mode telemetry: čte MAVLink z dronu (USB/UDP) a předává do Web UI
2. --mode sender: odesílá detekce jako MAVLink zprávy
3. --mode sim: simuluje dron (pro testování bez hardwaru)

Použití:
    # Simulace (pro vývoj/debug)
    python3 mavlink_bridge.py --mode sim --port 14550

    # Skutečný dron (USB connection)
    python3 mavlink_bridge.py --mode telemetry --connection /dev/ttyACM0

    # Odesílání detekcí jako MAVLink
    python3 mavlink_bridge.py --mode sender --port 14550 --detections-file output.txt
"""
import argparse
import json
import math
import os
import socket
import sys
import threading
import time
from pathlib import Path

try:
    import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False
    print("[mavlink_bridge] Warning: pymavlink not installed. Using simulation mode.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DroneState:
    """Simulated or telemetry drone state."""
    def __init__(self):
        self.battery_pct = 100.0
        self.altitude_m = 0.0
        self.latitude = 48.8566  # Paris
        self.longitude = 2.3522
        self.heading_deg = 0.0
        self.airspeed_ms = 0.0
        self.last_update = time.time()

    def to_dict(self):
        return {
            "battery_pct": round(self.battery_pct, 1),
            "altitude_m": round(self.altitude_m, 1),
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "heading_deg": round(self.heading_deg, 1),
            "airspeed_ms": round(self.airspeed_ms, 1),
        }

    def update(self):
        """Simulate drone state changes."""
        dt = time.time() - self.last_update
        self.last_update = time.time()

        # Simulated flight path: climb + circle
        self.altitude_m = min(100.0, self.altitude_m + 0.1 * dt)
        self.heading_deg = (self.heading_deg + 0.5 * dt) % 360
        self.airspeed_ms = 5.0 + 2.0 * math.sin(time.time() * 0.1)
        self.battery_pct = max(0, self.battery_pct - 0.5 * dt)  # 0.5%/sec simulated drain

        # Simple circular path
        radius = 0.001
        self.latitude += radius * math.cos(math.radians(self.heading_deg)) * dt * 0.01
        self.longitude += radius * math.sin(math.radians(self.heading_deg)) * dt * 0.01


class MAVLinkBridge:
    """Bridge between drone telemetry and AI detection system."""

    def __init__(self, mode: str, connection: str = "", udp_port: int = 14550):
        self.mode = mode
        self.connection = connection
        self.udp_port = udp_port
        self.drone = DroneState()
        self.detections = []
        self.running = True
        self.vehicle = None

        if MAVLINK_AVAILABLE and mode != "sim":
            try:
                self.vehicle = mavutil.mavlink_connection(
                    connection if connection else f"udp:localhost:{udp_port}"
                )
                print(f"[mavlink_bridge] Connected: {connection or f'udp:localhost:{udp_port}'}")
            except Exception as e:
                print(f"[mavlink_bridge] Connection failed: {e}")
                self.vehicle = None

    def telemetry_loop(self):
        """Continuously read telemetry from drone."""
        if self.vehicle:
            print("[mavlink_bridge] Telemetry loop started")
            while self.running:
                msg = self.vehicle.recv_match(blocking=True, timeout=0.5)
                if msg:
                    self._process_telemetry(msg)
        else:
            print("[mavlink_bridge] Simulation mode — updating drone state")
            while self.running:
                self.drone.update()
                time.sleep(0.2)

    def _process_telemetry(self, msg):
        """Process incoming MAVLink message."""
        msg_type = msg.get_type()
        if msg_type == "HEARTBEAT":
            # System status
            pass
        elif msg_type == "GPS_RAW_INT":
            self.drone.latitude = msg.lat / 1e7
            self.drone.longitude = msg.lon / 1e7
            self.drone.altitude_m = msg.relative_alt / 1000.0
            self.drone.heading_deg = msg.cog / 100.0
        elif msg_type == "BATTERY_STATUS":
            try:
                self.drone.battery_pct = msg.battery_remaining * 100
            except:
                pass
        elif msg_type == "VFR_HUD":
            self.drone.airspeed_ms = msg.airspeed
            self.drone.altitude_m = msg.alt

    def send_detection(self, detection):
        """Send detection results via MAVLink (as STATUSTEXT messages)."""
        if not self.vehicle or not MAVLINK_AVAILABLE:
            return

        text = f"DETECT:{detection['class']}:{detection['confidence']:.2f}:{detection['bbox']}"
        try:
            self.vehicle.mav.statustext_send(
                6,  # MAV_SEVERITY_NOTICE
                text.encode()
            )
        except Exception as e:
            print(f"[mavlink_bridge] Send error: {e}")

    def start_udp_broadcast(self, port: int = 5005):
        """Broadcast drone state via UDP for Web UI."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sock.settimeout(1.0)

        print(f"[mavlink_bridge] Broadcasting drone state on UDP port {port}")

        while self.running:
            try:
                data, addr = sock.recvfrom(4096)
                # Respond with current state
                response = json.dumps({
                    "timestamp": time.time(),
                    "drone_state": self.drone.to_dict(),
                    "detections": self.detections[-10:],  # Last 10
                })
                sock.sendto(response.encode(), addr)
            except socket.timeout:
                pass

        sock.close()

    def run(self):
        """Main entry point."""
        print(f"\n[mavlink_bridge] Mode: {self.mode}")

        if self.mode == "telemetry":
            # Read telemetry from drone
            t = threading.Thread(target=self.telemetry_loop, daemon=True)
            t.start()

            # Broadcast state via UDP
            self.start_udp_broadcast()

        elif self.mode == "sender":
            print("[mavlink_bridge] Sender mode — awaiting detections via stdin")
            print("  Format: class confidence x y w h")
            print("  Example: person 0.95 100 150 50 80")
            while self.running:
                line = sys.stdin.readline()
                if not line:
                    break
                parts = line.strip().split()
                if len(parts) >= 6:
                    detection = {
                        "class": parts[0],
                        "confidence": float(parts[1]),
                        "bbox": [int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])],
                    }
                    self.detections.append(detection)
                    self.send_detection(detection)

        elif self.mode == "sim":
            # Simulation mode
            while self.running:
                self.drone.update()
                print(f"\r[battery={self.drone.battery_pct:.0f}% "
                      f"alt={self.drone.altitude_m:.1f}m "
                      f"heading={self.drone.heading_deg:.0f}°", end="")
                time.sleep(1)


def parse_args():
    parser = argparse.ArgumentParser(description="MAVLink bridge for drone AI detection")
    parser.add_argument("--mode", choices=["telemetry", "sender", "sim"],
                        default="sim", help="Operation mode")
    parser.add_argument("--connection", type=str, default="",
                        help="MAVLink connection string (e.g. /dev/ttyACM0)")
    parser.add_argument("--port", type=int, default=14550,
                        help="UDP port for MAVLink telemetry")
    parser.add_argument("--broadcast-port", type=int, default=5005,
                        help="UDP port for broadcasting state to Web UI")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bridge = MAVLinkBridge(
        mode=args.mode,
        connection=args.connection,
        udp_port=args.port
    )

    try:
        bridge.run()
    except KeyboardInterrupt:
        print("\n[mavlink_bridge] Shutting down...")
        bridge.running = False
