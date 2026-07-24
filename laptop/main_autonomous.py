#!/usr/bin/env python3
"""
Main autonomous loop (laptop side).

Loop:
    1. Capture webcam frame
    2. Run YOLO object detection (background thread)
    3. Receive sensor + GPS status from Pi
    4. PEDESTRIAN HARD-STOP: if person detected close + center → force STOP
    5. Run Predictor (ONNX) -> action_id, confidence (gets YOLO features)
    6. If confidence < threshold -> send STOP
    7. Convert action_id -> drive/steer/speed command, send to Pi
    8. Log everything for debugging

Usage:
    python laptop/main_autonomous.py --pi 192.168.1.100 --model laptop/models/model.onnx
"""

import argparse
import csv
import json
import os
import select
import socket
import sys
import termios
import threading
import time
import tty

import cv2

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from vision.camera import Camera
from ml.actions import (ACTION_NAMES, STOP, action_to_pi_command)
from ml.inference import Predictor

# YOLO is optional — if not available, autonomous mode runs without it
try:
    from vision.object_detector import ObjectDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# Hard-stop thresholds for pedestrian safety
PEDESTRIAN_AREA_THRESHOLD = 0.08   # area_ratio > 8% of frame = "close"
PEDESTRIAN_CENTER_ONLY = True       # only stop for center-positioned people


class PiClient:
    """TCP client: sends commands, receives status."""
    def __init__(self, ip: str, port: int = 5555):
        self.ip = ip; self.port = port
        self.sock = None
        self.connected = False
        self.running = True
        self.status = None
        self.status_lock = threading.Lock()

    def connect(self) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((self.ip, self.port))
            s.settimeout(0.5)
            self.sock = s; self.connected = True
            return True
        except Exception as e:
            print(f"Connect failed: {e}")
            return False

    def send(self, cmd: dict):
        if not self.connected: return
        try:
            self.sock.sendall(json.dumps(cmd).encode('utf-8'))
        except Exception:
            self.connected = False

    def _receiver(self):
        buf = ""
        while self.running and self.connected:
            try:
                data = self.sock.recv(8192)
                if not data:
                    self.connected = False; return
                buf += data.decode('utf-8')
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    try:
                        with self.status_lock:
                            self.status = json.loads(line)
                    except json.JSONDecodeError:
                        pass
            except socket.timeout:
                continue
            except Exception:
                self.connected = False; return

    def start(self):
        threading.Thread(target=self._receiver, daemon=True).start()

    def get_status(self):
        with self.status_lock:
            return dict(self.status) if self.status else None

    def close(self):
        self.running = False
        if self.sock:
            try: self.sock.close()
            except Exception: pass


class AutonomousDriver:
    def __init__(self, pi_ip: str, model_path: str, camera_id: int,
                 conf_threshold: float = 0.55, loop_hz: float = 10.0,
                 use_yolo: bool = True, log_dir: str = None):
        self.client = PiClient(pi_ip)
        self.camera = Camera(device=camera_id)
        self.predictor = Predictor(model_path)
        self.conf_threshold = conf_threshold
        self.loop_interval = 1.0 / loop_hz
        self.running = True
        self.paused = True
        self.prev_action = STOP

        # YOLO setup
        self.yolo = None
        self.yolo_lock = threading.Lock()
        self.latest_yolo = {
            'person_detected': 0, 'object_detected': 0,
            'nearest_area_ratio': 0.0, 'nearest_position': 1, 'num_objects': 0,
        }
        if use_yolo and YOLO_AVAILABLE:
            try:
                print("[YOLO] Loading...")
                self.yolo = ObjectDetector(conf_threshold=0.4, device='cpu')
                print("[YOLO] Ready")
            except Exception as e:
                print(f"[YOLO] Failed: {e}")

        # Logging setup
        self.log_dir = log_dir
        self.log_file = None
        self.log_writer = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir,
                                    f"autonomous_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            self.log_file = open(log_path, 'w', newline='')
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow([
                'timestamp', 'paused',
                'FL', 'FR', 'FW', 'BC', 'LS', 'RS',
                'min_front', 'min_back',
                'gps_valid', 'gps_speed', 'gps_heading',
                'yolo_person', 'yolo_object', 'yolo_area', 'yolo_pos', 'yolo_count',
                'ml_action', 'ml_confidence',
                'final_action', 'override_reason',
                'sent_drive', 'sent_steer', 'sent_speed',
                'safety_violation',
            ])
            print(f"[Log] Writing to {log_path}")

    def _yolo_loop(self):
        """Background thread: runs YOLO ~5Hz on latest camera frame."""
        while self.running:
            if self.yolo is None:
                time.sleep(1.0)
                continue
            frame, _ = self.camera.read()
            if frame is None:
                time.sleep(0.1)
                continue
            try:
                feats = self.yolo.extract_features(frame)
                with self.yolo_lock:
                    self.latest_yolo = {
                        'person_detected': feats.person_detected,
                        'object_detected': feats.object_detected,
                        'nearest_area_ratio': feats.nearest_area_ratio,
                        'nearest_position': feats.nearest_position,
                        'num_objects': feats.num_objects,
                    }
            except Exception:
                pass
            time.sleep(0.2)

    def _get_yolo(self):
        with self.yolo_lock:
            return dict(self.latest_yolo)

    def _check_pedestrian_hardstop(self, yolo: dict) -> str:
        """Returns reason string if pedestrian hard-stop triggered, else empty string."""
        if not yolo.get('person_detected'):
            return ''
        area = yolo.get('nearest_area_ratio', 0.0)
        pos = yolo.get('nearest_position', 1)
        if area < PEDESTRIAN_AREA_THRESHOLD:
            return ''
        if PEDESTRIAN_CENTER_ONLY and pos != 1:  # 0=left, 1=center, 2=right
            return ''
        return f"PEDESTRIAN_CLOSE area={area:.2f} pos={pos}"

    def run(self):
        print(f"Connecting to Pi @ {self.client.ip}...")
        if not self.client.connect():
            return
        self.client.start()
        if not self.camera.start():
            return

        # Start YOLO thread
        if self.yolo:
            threading.Thread(target=self._yolo_loop, daemon=True).start()

        print("=" * 55)
        print(" AUTONOMOUS ML DRIVER")
        print("=" * 55)
        print(f" YOLO: {'ON' if self.yolo else 'OFF'}")
        print(f" Pedestrian hard-stop: ON (area>{PEDESTRIAN_AREA_THRESHOLD})")
        print(f" Logging: {'ON' if self.log_file else 'OFF'}")
        print(" G      = GO (start autonomous driving)")
        print(" SPACE  = PAUSE / STOP")
        print(" Q/ESC  = Quit")
        print("=" * 55)

        old = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            last_pred_log = 0.0
            while self.running:
                loop_start = time.time()

                if select.select([sys.stdin], [], [], 0.0)[0]:
                    ch = sys.stdin.read(1)
                    if ch in ('q', 'Q', '\x03'):
                        break
                    if ch in ('g', 'G'):
                        self.paused = False
                    if ch == ' ':
                        self.paused = True
                        self.client.send({'command': 'STOP', 'steer': 'STEER_STOP', 'speed': 0})

                frame, _ = self.camera.read()
                status = self.client.get_status()
                yolo = self._get_yolo()

                if self.paused:
                    self._log_row(status, yolo, None, 0.0, STOP, 'PAUSED', None)
                    time.sleep(self.loop_interval)
                    continue

                if frame is None or status is None:
                    time.sleep(self.loop_interval)
                    continue

                dists = status.get('distances', {})
                sensors = {k: float(dists.get(k, 0)) for k in ['FL','FR','FW','BC','LS','RS']}
                gps = status.get('gps') or {}
                gps_valid = int(gps.get('valid', 0))
                gps_speed = float(gps.get('speed_mps', 0.0))
                gps_heading = float(gps.get('heading_deg', 0.0))

                # Inference (with YOLO features)
                pred = self.predictor.predict(
                    frame, sensors, gps_valid, gps_speed, gps_heading,
                    self.prev_action, yolo)

                ml_action = pred['action_id']
                conf = pred['confidence']

                # Decide final action
                final_action = ml_action
                override_reason = ''

                # 1. Pedestrian hard-stop (highest priority on laptop side)
                pedestrian_reason = self._check_pedestrian_hardstop(yolo)
                if pedestrian_reason:
                    final_action = STOP
                    override_reason = pedestrian_reason

                # 2. Low-confidence fallback
                elif conf < self.conf_threshold:
                    final_action = STOP
                    override_reason = f"LOW_CONFIDENCE {conf:.2f}"

                # Convert + send
                cmd = action_to_pi_command(final_action)
                self.client.send(cmd)
                self.prev_action = final_action

                # Log
                self._log_row(status, yolo, ml_action, conf, final_action, override_reason, cmd)

                # Console log (throttled)
                now = time.time()
                if now - last_pred_log > 0.5:
                    last_pred_log = now
                    flag = ''
                    if override_reason:
                        flag = f' ⚠{override_reason}'
                    print(f"\r[ML] {ACTION_NAMES[ml_action]:13s} c={conf*100:5.1f}% "
                          f"→ {ACTION_NAMES[final_action]:13s} | "
                          f"yolo:p={yolo.get('person_detected',0)} o={yolo.get('num_objects',0)} "
                          f"a={yolo.get('nearest_area_ratio',0):.2f} | "
                          f"F={min([v for v in [sensors['FL'],sensors['FR'],sensors['FW']] if v>2] or [0]):.0f}cm"
                          f"{flag}", end='', flush=True)

                elapsed = time.time() - loop_start
                time.sleep(max(0, self.loop_interval - elapsed))

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            self.client.send({'command': 'STOP', 'steer': 'STEER_STOP', 'speed': 0})
            time.sleep(0.3)
            self.client.close()
            self.camera.stop()
            if self.log_file:
                self.log_file.close()
            cv2.destroyAllWindows()
            print("\nStopped.")

    def _log_row(self, status, yolo, ml_action, conf, final_action, override_reason, cmd):
        if not self.log_writer:
            return
        if status is None:
            status = {}
        dists = status.get('distances', {})
        gps = status.get('gps', {})
        cmd = cmd or {}
        try:
            self.log_writer.writerow([
                time.time(), self.paused,
                dists.get('FL', 0), dists.get('FR', 0), dists.get('FW', 0),
                dists.get('BC', 0), dists.get('LS', 0), dists.get('RS', 0),
                status.get('min_distance_front', 0), status.get('min_distance_back', 0),
                int(gps.get('valid', 0)), gps.get('speed_mps', 0.0), gps.get('heading_deg', 0.0),
                yolo.get('person_detected', 0), yolo.get('object_detected', 0),
                yolo.get('nearest_area_ratio', 0.0), yolo.get('nearest_position', 1),
                yolo.get('num_objects', 0),
                ACTION_NAMES[ml_action] if ml_action is not None else '',
                conf,
                ACTION_NAMES[final_action] if final_action is not None else '',
                override_reason,
                cmd.get('command', ''), cmd.get('steer', ''), cmd.get('speed', 0),
                status.get('safety_violation', ''),
            ])
            self.log_file.flush()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pi', required=True, help='Pi IP address')
    p.add_argument('--model', required=True, help='Path to .onnx or .pt model')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--conf', type=float, default=0.55, help='Min confidence to act')
    p.add_argument('--hz', type=float, default=10.0)
    p.add_argument('--no-yolo', action='store_true', help='Disable YOLO')
    p.add_argument('--log-dir', default=os.path.join(os.path.dirname(__file__), 'autonomous_logs'),
                   help='Directory for autonomous run logs')
    args = p.parse_args()

    AutonomousDriver(
        args.pi, args.model, args.camera,
        conf_threshold=args.conf, loop_hz=args.hz,
        use_yolo=not args.no_yolo, log_dir=args.log_dir,
    ).run()


if __name__ == "__main__":
    main()
