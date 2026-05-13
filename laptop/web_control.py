#!/usr/bin/env python3
"""
Web-based vehicle remote control + recording + autonomous A->B navigation.

  Manual:        tap d-pad / WASD; sensor + camera display; recorder toggle.
  Autonomous:    draw a path on the map (tap A, then tap waypoints) -> GO.
                 Pipeline per tick (~5 Hz):
                   GPS Pure-Pursuit controller (15 m lookahead) ->
                     ML model (refinement, gated) ->
                       obstacle override (sensors + YOLO) ->
                         Pi motor command (with wheel-position integrator).

Start:
    python web_control.py --pi 192.168.1.100 --camera 4 --https
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import socket
import sys
import threading
import time
import uuid
from typing import Optional

import cv2
import yaml
from flask import Flask, Response, jsonify, render_template_string, request

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from vision.camera import Camera
from ml.actions import (
    manual_to_action, ACTION_NAMES, STOP, action_to_pi_command,
)
from nav.controller import NavController
from autonomous_hybrid import decide as obstacle_decide

try:
    from vision.object_detector import ObjectDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from ml.inference import Predictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

app = Flask(__name__)

# ---- globals ------------------------------------------------------------
pi_client: Optional["PiClient"] = None
camera: Optional[Camera] = None
recorder: Optional["DataRecorder"] = None
yolo = None
predictor: Optional["Predictor"] = None

yolo_lock = threading.Lock()
latest_yolo = {'person_detected': 0, 'object_detected': 0,
               'nearest_area_ratio': 0.0, 'nearest_position': 1,
               'num_objects': 0}

nav = NavController()
nav_lock = threading.Lock()
nav_active = False
nav_prev_action = STOP

_maps_key_cache: Optional[str] = None


# ========================================================================
# Helpers
# ========================================================================
def _load_maps_key() -> str:
    global _maps_key_cache
    if _maps_key_cache is not None:
        return _maps_key_cache
    path = os.path.join(os.path.dirname(__file__), 'secrets.yaml')
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        _maps_key_cache = (data.get('google_maps_api_key') or '').strip()
    except FileNotFoundError:
        _maps_key_cache = ''
    return _maps_key_cache


def get_yolo_snapshot() -> dict:
    with yolo_lock:
        return dict(latest_yolo)


def yolo_loop():
    global latest_yolo
    while True:
        if camera is None or yolo is None:
            time.sleep(0.5); continue
        frame, _ = camera.read()
        if frame is None:
            time.sleep(0.1); continue
        try:
            f = yolo.extract_features(frame)
            with yolo_lock:
                latest_yolo = {
                    'person_detected':    f.person_detected,
                    'object_detected':    f.object_detected,
                    'nearest_area_ratio': f.nearest_area_ratio,
                    'nearest_position':   f.nearest_position,
                    'num_objects':        f.num_objects,
                }
        except Exception:
            pass
        time.sleep(0.2)


def heading_usable_for_log(speed_mps: Optional[float]) -> bool:
    return speed_mps is not None and speed_mps >= 0.35


# ========================================================================
# PiClient (TCP to raspberry_pi/main.py)
# ========================================================================
class PiClient:
    def __init__(self, ip: str, port: int = 5555):
        self.ip = ip
        self.port = port
        self.sock = None
        self.connected = False
        self.running = True
        self.status: Optional[dict] = None
        self.status_lock = threading.Lock()
        self.drive = 'STOP'
        self.steer = 'STEER_STOP'
        self.speed = 50
        self.cmd_lock = threading.Lock()

    def connect(self) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((self.ip, self.port))
            s.settimeout(0.5)
            self.sock = s
            self.connected = True
            log.info(f"[PiClient] connected to {self.ip}:{self.port}")
            return True
        except Exception as e:
            log.error(f"[PiClient] connect failed: {e}")
            return False

    def _sender(self):
        while self.running and self.connected:
            with self.cmd_lock:
                d, s, sp = self.drive, self.steer, self.speed
            try:
                self.sock.sendall(json.dumps({'command': d, 'steer': s, 'speed': sp}).encode())
            except Exception:
                self.connected = False
                return
            time.sleep(0.2)

    def _receiver(self):
        buf = ""
        while self.running and self.connected:
            try:
                data = self.sock.recv(8192)
                if not data:
                    self.connected = False
                    return
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
                self.connected = False
                return

    def start(self):
        threading.Thread(target=self._sender, daemon=True).start()
        threading.Thread(target=self._receiver, daemon=True).start()

    def set_command(self, drive=None, steer=None, speed=None):
        with self.cmd_lock:
            if drive is not None: self.drive = drive
            if steer is not None: self.steer = steer
            if speed is not None: self.speed = max(0, min(100, int(speed)))

    def get_state_snapshot(self):
        with self.cmd_lock:
            return self.drive, self.steer, self.speed

    def get_status(self):
        with self.status_lock:
            return dict(self.status) if self.status else None

    def safe_stop(self):
        self.set_command(drive='STOP', steer='STEER_STOP', speed=0)

    def close(self):
        self.running = False
        self.safe_stop()
        time.sleep(0.2)
        if self.sock:
            try: self.sock.close()
            except Exception: pass


# ========================================================================
# DataRecorder
# ========================================================================
CSV_HEADER = [
    'session', 'timestamp', 'frame_path',
    'FL', 'FR', 'FW', 'BC', 'LS', 'RS',
    'gps_valid', 'gps_speed', 'gps_heading',
    'drive', 'steer', 'speed',
    'prev_action', 'action_label', 'action_name',
    'yolo_person', 'yolo_object', 'yolo_area', 'yolo_pos', 'yolo_count',
]


class DataRecorder:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.images_dir = os.path.join(out_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, 'dataset.csv')
        self.recording = False
        self.running = True
        self.samples_written = 0
        self.session_id = time.strftime('%Y%m%d_%H%M%S')
        self.prev_action = STOP
        self.lock = threading.Lock()
        self._init_csv()

    def _init_csv(self):
        is_new = (not os.path.exists(self.csv_path)) or os.path.getsize(self.csv_path) == 0
        self.csv_file = open(self.csv_path, 'a', newline='', buffering=1)
        self.csv_writer = csv.writer(self.csv_file)
        if is_new:
            self.csv_writer.writerow(CSV_HEADER)
        try:
            with open(self.csv_path, 'r') as f:
                self.samples_written = max(sum(1 for _ in f) - 1, 0)
        except Exception:
            pass

    def toggle(self):
        with self.lock:
            self.recording = not self.recording
            state = self.recording
        log.info(f"[Recorder] {'RECORDING' if state else 'PAUSED'} (samples: {self.samples_written})")
        return state

    def is_recording(self):
        with self.lock:
            return self.recording

    def record_loop(self):
        while self.running:
            if not self.is_recording():
                time.sleep(0.1); continue
            if camera is None or pi_client is None:
                time.sleep(0.1); continue
            frame, _ = camera.read()
            status = pi_client.get_status()
            if frame is None or status is None:
                time.sleep(0.1); continue
            drive, steer, speed = pi_client.get_state_snapshot()
            dists = status.get('distances', {})
            gps   = status.get('gps', {}) or {}
            action_id = manual_to_action(drive, steer, speed)

            frame_name = f"{self.session_id}_{uuid.uuid4().hex[:8]}.jpg"
            frame_path = os.path.join(self.images_dir, frame_name)
            tmp_path = os.path.join(self.images_dir, f".tmp_{frame_name}")
            if not cv2.imwrite(tmp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                continue
            os.rename(tmp_path, frame_path)

            y = get_yolo_snapshot()
            self.csv_writer.writerow([
                self.session_id, time.time(), frame_path,
                dists.get('FL', 0), dists.get('FR', 0), dists.get('FW', 0),
                dists.get('BC', 0), dists.get('LS', 0), dists.get('RS', 0),
                int(gps.get('valid', 0)), gps.get('speed_mps', 0.0), gps.get('heading_deg', 0.0),
                drive, steer, speed,
                self.prev_action, action_id, ACTION_NAMES[action_id],
                y['person_detected'], y['object_detected'], y['nearest_area_ratio'],
                y['nearest_position'], y['num_objects'],
            ])
            self.csv_file.flush()
            self.prev_action = action_id
            self.samples_written += 1
            time.sleep(0.1)

    def close(self):
        self.running = False
        self.recording = False
        if self.csv_file:
            self.csv_file.close()
        log.info(f"[Recorder] saved {self.samples_written} samples")


# ========================================================================
# Nav loop (autonomous mode)
# ========================================================================
def _set_nav_active(value: bool):
    global nav_active, nav_prev_action
    nav_active = value
    if not value:
        nav_prev_action = STOP
        if pi_client:
            pi_client.safe_stop()


def _apply_obstacle_override(wanted: str, sensors: dict, yolo_snap: dict):
    return obstacle_decide(sensors, yolo_snap, wanted_action=wanted)


def nav_loop():
    """GPS Pure-Pursuit -> ML model (gated) -> obstacle override -> Pi.

    Steering wheel position is tracked open-loop because the Pi's steering
    controller is an electric motor (no center sensor). Without this, every
    TURN_* would drive the wheels to the lock stop and they'd stay there.
    """
    global nav_prev_action
    log.info("[nav] loop started")

    # --- steering integrator ---
    steer_pos = 0.0                # -1 = full left, 0 = straight, +1 = full right
    last_steer_cmd = 'STEER_STOP'
    last_steer_time = time.time()
    STEER_RATE_PER_S = 1.6         # full sweep in ~0.6 s of continuous PWM
    STEER_DEADBAND = 0.10

    while True:
        if not nav_active or pi_client is None:
            time.sleep(0.2); continue

        status = pi_client.get_status()
        if status is None:
            time.sleep(0.1); continue

        gps = status.get('gps', {}) or {}
        valid = bool(gps.get('valid', 0))
        lat = gps.get('lat') if valid else None
        lon = gps.get('lon') if valid else None
        heading = gps.get('heading_deg', 0.0)
        speed_mps = gps.get('speed_mps', 0.0)
        sensors = status.get('distances', {}) or {}
        yolo_snap = get_yolo_snapshot()

        with nav_lock:
            wanted = nav.tick(lat, lon, heading, speed_mps)

        snap = nav.snapshot()
        if snap.state == "ARRIVED":
            log.info("[nav] ARRIVED — stopping autonomous mode")
            _set_nav_active(False)
            continue

        # ---- ML model usage (gated) ----
        # Model is behaviour-cloned -- it doesn't know where B is, so it
        # cannot pick direction toward the goal. But it can:
        #   1. Refine FORWARD -> SLOW_DOWN in cluttered scenes
        #   2. Pick a TURN side when GPS heading is unavailable
        #   3. Suggest REVERSE / REVERSE_LEFT/RIGHT when stuck
        # Cannot escalate to STOP (avoids the GO-time deadlock).
        ENABLE_MODEL = True
        model_action = None
        model_conf = 0.0
        if ENABLE_MODEL and predictor is not None and camera is not None and wanted != "STOP":
            frame, _ = camera.read()
            if frame is not None:
                try:
                    out = predictor.predict(
                        frame, sensors,
                        gps_valid=int(valid),
                        gps_speed=float(speed_mps or 0.0),
                        gps_heading_deg=float(heading or 0.0),
                        prev_action=int(nav_prev_action),
                        yolo=yolo_snap,
                    )
                    model_action = out['action_name']
                    model_conf = float(out.get('confidence', 0.0))
                    if wanted == "FORWARD" and model_action == "SLOW_DOWN" and model_conf > 0.55:
                        wanted = "SLOW_DOWN"
                    elif (not heading_usable_for_log(speed_mps)
                          and wanted == "FORWARD"
                          and model_action in ("TURN_LEFT", "TURN_RIGHT")
                          and model_conf > 0.50):
                        wanted = model_action
                    elif (wanted in ("FORWARD", "SLOW_DOWN")
                          and model_action in ("REVERSE", "REVERSE_LEFT", "REVERSE_RIGHT")
                          and model_conf > 0.70):
                        wanted = model_action
                except Exception as e:
                    log.warning(f"[nav] model predict failed: {e}")

        # ---- obstacle override (safety) ----
        final, reason = _apply_obstacle_override(wanted, sensors, yolo_snap)

        # ---- commitment window: prevent flapping ----
        action_id = ACTION_NAMES.index(final) if final in ACTION_NAMES else STOP
        now = time.time()
        last_action = getattr(nav_loop, '_last_action', None)
        last_action_t = getattr(nav_loop, '_last_action_t', 0.0)
        prev_was_emergency = last_action in ("REVERSE", "REVERSE_LEFT", "REVERSE_RIGHT")
        new_is_emergency = final in ("REVERSE", "REVERSE_LEFT", "REVERSE_RIGHT")
        prev_was_turn = last_action in ("TURN_LEFT", "TURN_RIGHT")

        if last_action is None or final == last_action:
            commit_ok = True
        elif prev_was_emergency:
            commit_ok = (now - last_action_t) >= 1.5
        elif new_is_emergency:
            commit_ok = True
        elif last_action == "FORWARD" and final in ("TURN_LEFT", "TURN_RIGHT"):
            commit_ok = (now - last_action_t) >= 2.0
        elif prev_was_turn and final == "FORWARD":
            commit_ok = True
        else:
            commit_ok = (now - last_action_t) >= 1.0

        if commit_ok:
            final_for_steer = final
            nav_loop._last_action = final
            nav_loop._last_action_t = now
            nav_prev_action = action_id
        else:
            final_for_steer = last_action or final

        # ---- steering integrator (re-center wheels actively) ----
        dt = now - last_steer_time
        if last_steer_cmd == 'LEFT':
            steer_pos -= STEER_RATE_PER_S * dt
        elif last_steer_cmd == 'RIGHT':
            steer_pos += STEER_RATE_PER_S * dt
        steer_pos = max(-1.0, min(1.0, steer_pos))
        last_steer_time = now

        if final_for_steer in ('TURN_LEFT', 'REVERSE_LEFT'):
            target_pos = -1.0
        elif final_for_steer in ('TURN_RIGHT', 'REVERSE_RIGHT'):
            target_pos = +1.0
        else:
            target_pos = 0.0

        err = target_pos - steer_pos
        if abs(err) < STEER_DEADBAND:
            new_steer = 'STEER_STOP'
        elif err > 0:
            new_steer = 'RIGHT'
        else:
            new_steer = 'LEFT'
        last_steer_cmd = new_steer

        send_id = action_id if commit_ok else ACTION_NAMES.index(final_for_steer)
        cmd = action_to_pi_command(send_id)
        pi_client.set_command(drive=cmd['command'], steer=new_steer, speed=cmd['speed'])

        # ---- per-second log ----
        if now - getattr(nav_loop, '_last_log', 0) > 1.0:
            nav_loop._last_log = now
            log.info(
                "[nav] state=%s want=%s final=%s reason=%s "
                "lat=%.6f lon=%.6f hd=%.1f spd=%.2f "
                "dist=%s err=%s xt=%s wheel=%+0.2f steer=%s model=%s/%.2f",
                snap.state, snap.wanted_action, final, reason or '-',
                lat or 0.0, lon or 0.0, heading or 0.0, speed_mps or 0.0,
                f"{snap.distance_m:.1f}" if snap.distance_m is not None else '-',
                f"{snap.heading_err_deg:+.1f}" if snap.heading_err_deg is not None else '-',
                f"{snap.cross_track_m:+.1f}" if snap.cross_track_m is not None else '-',
                steer_pos, new_steer,
                model_action or '-', model_conf,
            )

        time.sleep(0.2)


# ========================================================================
# HTML page (Maps + draw-path UI + manual controls + recorder)
# ========================================================================
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,user-scalable=no">
<title>Vehicle Control</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#1a1a2e; color:#eee; font-family:system-ui,sans-serif;
       touch-action:manipulation; user-select:none; -webkit-user-select:none; }
.header { background:#16213e; padding:10px 16px; display:flex;
          justify-content:space-between; align-items:center; }
.header h1 { font-size:18px; color:#0f0; }
.status-dot { width:12px; height:12px; border-radius:50%; display:inline-block; }
.status-dot.on { background:#0f0; }
.status-dot.off { background:#f00; }

#map { width:100%; max-width:640px; height:55vh; min-height:360px;
       margin:8px auto; border-radius:8px; }
.map-toolbar { max-width:640px; margin:0 auto 8px; padding:0 8px;
               display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:6px; }
.map-toolbar .tbtn { background:#16213e; border:1px solid #1e90ff; border-radius:6px;
                     padding:8px 6px; text-align:center; font-size:12px; color:#fff; cursor:pointer; }
.map-toolbar .tbtn.active { background:#1e90ff; }
#searchInput { width:calc(100% - 16px); margin:0 8px 8px; padding:8px;
               background:#16213e; border:1px solid #333; color:#fff; border-radius:6px; }
.nav-controls { max-width:640px; margin:0 auto 8px; padding:0 8px;
                display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; }
.btn-nav { padding:14px; border-radius:8px; font-weight:bold; font-size:14px;
           text-align:center; cursor:pointer; color:#fff; }
.btn-nav.go    { background:#0a3d62; border:2px solid #1e90ff; }
.btn-nav.stop  { background:#3a0000; border:2px solid #f44; }
.btn-nav.clear { background:#222; border:2px solid #555; }
.nav-status { max-width:640px; margin:0 auto; padding:4px 12px;
              text-align:center; font-size:13px; color:#0cf; }

.camera-container { max-width:640px; margin:8px auto; position:relative;
                    background:#000; border-radius:8px; overflow:hidden; }
.camera-container img { width:100%; display:block; }
.rec-overlay { position:absolute; top:10px; left:10px; padding:4px 10px;
               border-radius:4px; font-weight:bold; font-size:14px; }
.rec-overlay.on { background:rgba(255,0,0,0.85); color:#fff; }
.rec-overlay.off { background:rgba(0,0,0,0.5); color:#888; }
.sensor-bar { display:grid; grid-template-columns:repeat(3,1fr); gap:6px;
              padding:8px 12px; max-width:640px; margin:0 auto; }
.sensor { background:#16213e; border-radius:6px; padding:6px; text-align:center; }
.sensor .label { font-size:10px; color:#888; }
.sensor .value { font-size:16px; font-weight:bold; }
.sensor .value.danger { color:#f44; }
.sensor .value.warn { color:#fa0; }
.sensor .value.safe { color:#0f0; }
.controls { max-width:400px; margin:12px auto; padding:0 12px; }
.dpad { display:grid; grid-template-columns:1fr 1fr 1fr;
        grid-template-rows:1fr 1fr 1fr; gap:8px; width:260px; margin:0 auto; }
.btn { background:#0a3d62; border:2px solid #1e90ff; border-radius:12px;
       color:#fff; font-size:18px; font-weight:bold; cursor:pointer;
       display:flex; align-items:center; justify-content:center; min-height:60px; }
.btn:active, .btn.active { background:#1e90ff; }
.btn.stop-btn { background:#8b0000; border-color:#f44; font-size:14px; }
.rec-btn { display:block; max-width:260px; margin:12px auto; padding:12px;
           border:2px solid #f44; border-radius:12px; background:#3a0000;
           color:#fff; font-size:16px; font-weight:bold; text-align:center; cursor:pointer; }
.rec-btn.recording { background:#f44; }
.speed-section { max-width:400px; margin:12px auto; padding:0 24px; text-align:center; }
.speed-section input[type=range] { width:100%; }
.cmd-display { max-width:640px; margin:4px auto; text-align:center;
               font-size:14px; color:#1e90ff; }
.info-bar { max-width:640px; margin:8px auto; padding:4px 12px;
            display:flex; justify-content:space-between; font-size:11px; color:#666; }
</style>
</head>
<body>

<div class="header">
    <h1>VEHICLE CONTROL</h1>
    <div><span class="status-dot" id="connDot"></span>
         <span id="connText">---</span></div>
</div>

<div id="map"></div>
<input type="text" id="searchInput" placeholder="Search a place or address...">
<div class="map-toolbar">
    <div class="tbtn" id="btnDraw">✏️ Draw Path</div>
    <div class="tbtn" id="btnUseCar">🚗 Start at Car</div>
    <div class="tbtn" id="btnLocate">📍 Me</div>
    <div class="tbtn" id="btnFindCar">Find Car</div>
</div>
<div class="map-toolbar" style="grid-template-columns:1fr 1fr;">
    <div class="tbtn" id="btnUndo">↶ Undo last point</div>
    <div class="tbtn" id="btnClearAB">Clear Path</div>
</div>
<div class="nav-controls">
    <div class="btn-nav go"    id="btnNavGo">GO (autonomous)</div>
    <div class="btn-nav stop"  id="btnNavStop">STOP</div>
    <div class="btn-nav clear" id="btnNavClear">Clear Nav</div>
</div>
<div class="nav-status" id="navStatus">Nav: IDLE</div>

<div class="camera-container">
    <img id="camFeed" src="/video_feed" alt="Camera">
    <div class="rec-overlay off" id="recOverlay">REC OFF</div>
</div>

<div class="cmd-display">
    <span id="cmdDrive">STOP</span> | <span id="cmdSteer">STRAIGHT</span>
    | YOLO: <span id="yoloStatus">--</span>
    | GPS: <span id="gpsStatus">--</span>
</div>

<div class="sensor-bar">
    <div class="sensor"><div class="label">FL</div><div class="value" id="sFL">--</div></div>
    <div class="sensor"><div class="label">FW</div><div class="value" id="sFW">--</div></div>
    <div class="sensor"><div class="label">FR</div><div class="value" id="sFR">--</div></div>
    <div class="sensor"><div class="label">LS</div><div class="value" id="sLS">--</div></div>
    <div class="sensor"><div class="label">BC</div><div class="value" id="sBC">--</div></div>
    <div class="sensor"><div class="label">RS</div><div class="value" id="sRS">--</div></div>
</div>

<div class="controls">
    <div class="dpad">
        <div></div>
        <div class="btn" id="btnW" data-drive="FORWARD">FWD</div>
        <div class="btn" id="btnX" data-steer="STEER_STOP">STR</div>
        <div class="btn" id="btnA" data-steer="LEFT">LEFT</div>
        <div class="btn stop-btn" id="btnStop" data-drive="STOP" data-steer="STEER_STOP">STOP</div>
        <div class="btn" id="btnD" data-steer="RIGHT">RIGHT</div>
        <div></div>
        <div class="btn" id="btnS" data-drive="BACKWARD">REV</div>
        <div></div>
    </div>
</div>

<div class="speed-section">
    <label>SPEED <span id="speedVal">50%</span></label>
    <input type="range" id="speedSlider" min="0" max="100" value="50" step="5">
</div>

<div class="rec-btn" id="recBtn" onclick="toggleRec()">START RECORDING</div>
<div class="info-bar">
    <span>Samples: <span id="sampleCount">0</span></span>
    <span>Speed: <span id="actualSpeed">0</span>%</span>
    <span id="autoState">--</span>
</div>

<script>
// ============== Maps + path drawing ==============
let map = null, geocoder = null;
let drawMode = false;
let pathPoints = [];
let pathMarkers = [];
let pathLine = null;
let carMarker = null;
let trailPoints = [];
let trail = null;
let lastTrailTime = 0;

function setDrawMode(on) {
    drawMode = on;
    document.getElementById('btnDraw').classList.toggle('active', on);
    document.getElementById('btnDraw').textContent =
        on ? '✏️ Tap to add (ON)' : '✏️ Draw Path';
}
function redrawPath() {
    if (!map) return;
    if (pathLine) pathLine.setMap(null);
    pathLine = new google.maps.Polyline({
        path: pathPoints, geodesic: false,
        strokeColor: '#1e90ff', strokeOpacity: 0.9, strokeWeight: 4, map: map,
    });
    pathMarkers.forEach(m => m.setMap(null));
    pathMarkers = pathPoints.map((p, i) => {
        const label = (i === 0) ? 'A' :
                      (i === pathPoints.length - 1) ? 'B' :
                      String.fromCharCode(65 + i);
        const icon = (i === 0) ? 'http://maps.google.com/mapfiles/ms/icons/green-dot.png' :
                     (i === pathPoints.length - 1) ? 'http://maps.google.com/mapfiles/ms/icons/red-dot.png' :
                     'http://maps.google.com/mapfiles/ms/icons/blue-dot.png';
        return new google.maps.Marker({position: p, map: map, label: label, icon: icon});
    });
}
function addPathPoint(latLng) { pathPoints.push(latLng); redrawPath(); postWaypoints(); }
function undoLastPoint() { if (!pathPoints.length) return; pathPoints.pop(); redrawPath(); postWaypoints(); }
function clearPath() { pathPoints = []; redrawPath(); fetch('/api/stop', {method:'POST'}); }
function postWaypoints() {
    const wps = pathPoints.map(p => ({lat: p.lat(), lon: p.lng()}));
    if (!wps.length) return Promise.resolve();
    return fetch('/api/destination', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({waypoints: wps})
    });
}

function initMap() {
    map = new google.maps.Map(document.getElementById('map'), {
        center: {lat: 33.6520, lng: 73.1613}, zoom: 18, mapTypeId: 'hybrid',
        mapTypeControl: true,
        mapTypeControlOptions: {
            style: google.maps.MapTypeControlStyle.HORIZONTAL_BAR,
            position: google.maps.ControlPosition.TOP_LEFT,
        },
        zoomControl: true, streetViewControl: false,
        fullscreenControl: true, gestureHandling: 'greedy',
    });
    geocoder = new google.maps.Geocoder();
    map.addListener('click', (e) => { if (drawMode) addPathPoint(e.latLng); });
    const input = document.getElementById('searchInput');
    if (google.maps.places) {
        const ac = new google.maps.places.Autocomplete(input, {fields:['geometry']});
        ac.bindTo('bounds', map);
        ac.addListener('place_changed', () => {
            const p = ac.getPlace();
            if (p.geometry) { map.panTo(p.geometry.location); map.setZoom(18); }
        });
    } else {
        input.addEventListener('keypress', (e) => {
            if (e.key !== 'Enter') return;
            geocoder.geocode({address: input.value}, (res, st) => {
                if (st === 'OK' && res[0]) { map.panTo(res[0].geometry.location); map.setZoom(18); }
            });
        });
    }
    document.getElementById('btnDraw').addEventListener('click', () => setDrawMode(!drawMode));
    document.getElementById('btnUndo').addEventListener('click', undoLastPoint);
    document.getElementById('btnClearAB').addEventListener('click', clearPath);
    document.getElementById('btnLocate').addEventListener('click', () => {
        if (!navigator.geolocation) return alert('No geolocation');
        navigator.geolocation.getCurrentPosition(
            pos => { const ll = new google.maps.LatLng(pos.coords.latitude, pos.coords.longitude);
                     map.panTo(ll); map.setZoom(18); },
            err => alert('Geo error: ' + err.message),
            {enableHighAccuracy: true, timeout: 7000});
    });
    document.getElementById('btnUseCar').addEventListener('click', () => {
        fetch('/api/status').then(r=>r.json()).then(s => {
            if (s.gps && s.gps.valid) {
                const ll = new google.maps.LatLng(s.gps.lat, s.gps.lon);
                addPathPoint(ll); map.panTo(ll);
            } else { alert('Car GPS not valid'); }
        });
    });
    document.getElementById('btnFindCar').addEventListener('click', () => {
        fetch('/api/status').then(r=>r.json()).then(s => {
            if (s.gps && s.gps.valid) { map.panTo({lat: s.gps.lat, lng: s.gps.lon}); map.setZoom(19); }
            else { alert('Car GPS not valid'); }
        });
    });
    document.getElementById('btnNavGo').addEventListener('click', () => {
        fetch('/api/go', {method:'POST'}).then(r=>r.json()).then(d => {
            if (!d.ok) alert(d.error || 'GO failed');
        });
    });
    document.getElementById('btnNavStop').addEventListener('click', () => fetch('/api/stop', {method:'POST'}));
    document.getElementById('btnNavClear').addEventListener('click', () => {
        fetch('/api/stop', {method:'POST'}).then(() => clearPath());
    });
}
fetch('/api/maps_key').then(r=>r.json()).then(d => {
    if (!d.key) {
        document.getElementById('map').innerHTML =
            '<div style="padding:30px;text-align:center;color:#888;">No Google Maps API key. Add google_maps_api_key to secrets.yaml.</div>';
        return;
    }
    const s = document.createElement('script');
    s.src = 'https://maps.googleapis.com/maps/api/js?key=' + encodeURIComponent(d.key)
            + '&libraries=places&callback=initMap';
    s.async = true; document.head.appendChild(s);
});
window.initMap = initMap;

// ============== Manual d-pad ==============
function sendCmd(drive, steer, speed) {
    const params = new URLSearchParams();
    if (drive) params.set('drive', drive);
    if (steer) params.set('steer', steer);
    if (speed !== undefined) params.set('speed', speed);
    fetch('/cmd?' + params.toString()).catch(()=>{});
}
const STEER_PULSE_MS = 200;
let steerTimer = null;
document.querySelectorAll('.btn').forEach(btn => {
    const drive = btn.dataset.drive || null;
    const steer = btn.dataset.steer || null;
    function tap(e) {
        e.preventDefault();
        document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        sendCmd(drive, steer);
        if (steerTimer) { clearTimeout(steerTimer); steerTimer = null; }
        if (steer === 'LEFT' || steer === 'RIGHT') {
            steerTimer = setTimeout(() => {
                sendCmd(null, 'STEER_STOP'); btn.classList.remove('active'); steerTimer = null;
            }, STEER_PULSE_MS);
        }
    }
    btn.addEventListener('click', tap);
    btn.addEventListener('touchstart', tap, {passive:false});
});
const driveMap = {w:'FORWARD', s:'BACKWARD'};
const steerMap = {a:'LEFT', d:'RIGHT'};
document.addEventListener('keydown', e => {
    const k = e.key.toLowerCase();
    if (driveMap[k]) sendCmd(driveMap[k], null);
    else if (steerMap[k]) {
        sendCmd(null, steerMap[k]);
        if (steerTimer) clearTimeout(steerTimer);
        steerTimer = setTimeout(() => { sendCmd(null,'STEER_STOP'); steerTimer=null; }, STEER_PULSE_MS);
    }
    else if (k === ' ') { e.preventDefault(); sendCmd('STOP','STEER_STOP'); }
    else if (k === 'x') sendCmd(null, 'STEER_STOP');
    else if (k === 'r') toggleRec();
});
const slider = document.getElementById('speedSlider');
const speedVal = document.getElementById('speedVal');
slider.addEventListener('input', () => {
    speedVal.textContent = slider.value + '%';
    sendCmd(null, null, slider.value);
});
function toggleRec() {
    fetch('/record/toggle').then(r=>r.json()).then(d => updateRecUI(d.recording, d.samples));
}
function updateRecUI(rec, samples) {
    const btn = document.getElementById('recBtn');
    const ov  = document.getElementById('recOverlay');
    if (rec) { btn.textContent='STOP RECORDING'; btn.classList.add('recording');
               ov.textContent='REC'; ov.className='rec-overlay on'; }
    else     { btn.textContent='START RECORDING'; btn.classList.remove('recording');
               ov.textContent='REC OFF'; ov.className='rec-overlay off'; }
    document.getElementById('sampleCount').textContent = samples || 0;
}
function colorForDist(v) {
    if (!v) return '';
    if (v < 50) return 'danger';
    if (v < 150) return 'warn';
    return 'safe';
}
function pollStatus() {
    fetch('/api/status').then(r=>r.json()).then(s => {
        document.getElementById('connDot').className = 'status-dot ' + (s.connected ? 'on' : 'off');
        document.getElementById('connText').textContent = s.connected ? 'Connected' : 'Disconnected';
        if (s.distances) {
            ['FL','FR','FW','BC','LS','RS'].forEach(k => {
                const el = document.getElementById('s'+k);
                const v = s.distances[k] || 0;
                el.textContent = v.toFixed(0);
                el.className = 'value ' + colorForDist(v);
            });
        }
        document.getElementById('cmdDrive').textContent = s.drive || 'STOP';
        document.getElementById('cmdSteer').textContent =
            (s.steer === 'STEER_STOP' ? 'STR' : s.steer) || 'STR';
        document.getElementById('actualSpeed').textContent = s.actual_speed || s.speed || 0;
        document.getElementById('autoState').textContent =
            s.nav ? (s.nav.state + (s.nav.active ? ' [ACTIVE]' : '')) : '--';
        if (s.yolo) {
            const y = s.yolo;
            const pos = ['L','C','R'][y.nearest_position] || '-';
            let t = 'objs=' + (y.num_objects||0);
            if (y.person_detected) t = '⚠ PERSON ' + pos + ' | ' + t;
            document.getElementById('yoloStatus').textContent = t;
        }
        if (s.gps) {
            document.getElementById('gpsStatus').textContent =
                s.gps.valid ? `valid ${s.gps.satellites||'?'}` : 'no fix';
        }
        if (map && s.gps && s.gps.valid) {
            const cp = {lat: s.gps.lat, lng: s.gps.lon};
            if (carMarker) carMarker.setPosition(cp);
            else carMarker = new google.maps.Marker({
                position: cp, map: map, title: 'Car',
                icon: 'http://maps.google.com/mapfiles/ms/icons/cabs.png'});
            const now = Date.now();
            if (now - lastTrailTime > 1000) {
                lastTrailTime = now;
                trailPoints.push(cp);
                if (trailPoints.length > 300) trailPoints.shift();
                if (trail) trail.setMap(null);
                trail = new google.maps.Polyline({
                    path: trailPoints, geodesic: false,
                    strokeColor: '#ff8c00', strokeOpacity: 0.85,
                    strokeWeight: 3, map: map,
                });
            }
        }
        if (s.nav) {
            const n = s.nav;
            let line = 'Nav: ' + n.state;
            if (n.distance_m != null) line += ' | ' + n.distance_m.toFixed(1) + ' m';
            if (n.wanted_action) line += ' | wants ' + n.wanted_action;
            if (n.reason) line += ' | ' + n.reason;
            document.getElementById('navStatus').textContent = line;
        }
        updateRecUI(s.recording, s.samples);
    }).catch(()=>{});
}
setInterval(pollStatus, 300);
</script>
</body>
</html>
"""


# ========================================================================
# Flask routes
# ========================================================================
@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/cmd')
def cmd():
    drive = request.args.get('drive')
    steer = request.args.get('steer')
    speed = request.args.get('speed')
    if nav_active:
        _set_nav_active(False)
    if pi_client:
        pi_client.set_command(drive=drive, steer=steer,
                              speed=int(speed) if speed else None)
    return jsonify(ok=True)


@app.route('/record/toggle')
def record_toggle():
    if recorder:
        is_rec = recorder.toggle()
        return jsonify(recording=is_rec, samples=recorder.samples_written)
    return jsonify(recording=False, samples=0)


@app.route('/api/status')
def api_status():
    st = pi_client.get_status() if pi_client else None
    if pi_client:
        drive, steer, speed = pi_client.get_state_snapshot()
    else:
        drive, steer, speed = 'STOP', 'STEER_STOP', 0
    snap = nav.snapshot()
    return jsonify(
        connected=bool(pi_client and pi_client.connected),
        drive=drive, steer=steer, speed=speed,
        actual_speed=(st or {}).get('actual_speed', 0),
        distances=(st or {}).get('distances', {}),
        gps=(st or {}).get('gps', {'valid': False, 'lat': 0.0, 'lon': 0.0}),
        recording=recorder.is_recording() if recorder else False,
        samples=recorder.samples_written if recorder else 0,
        yolo=get_yolo_snapshot(),
        nav={
            'state': snap.state, 'active': nav_active,
            'waypoints': snap.waypoints,
            'distance_m': snap.distance_m,
            'wanted_action': snap.wanted_action,
            'reason': snap.reason,
        },
    )


# Backwards-compatible /status (manual UI used it before).
@app.route('/status')
def status_compat():
    return api_status()


@app.route('/api/maps_key')
def api_maps_key():
    return jsonify(key=_load_maps_key())


@app.route('/api/destination', methods=['POST'])
def api_destination():
    body = request.get_json(silent=True) or {}
    if 'waypoints' in body:
        raw = body['waypoints']
        if not isinstance(raw, list):
            return jsonify(ok=False, error='waypoints must be a list'), 400
        pts = []
        for w in raw:
            try:
                pts.append((float(w['lat']), float(w['lon'])))
            except (KeyError, TypeError, ValueError):
                continue
        if not pts:
            return jsonify(ok=False, error='no valid waypoints'), 400
    elif 'lat' in body and 'lon' in body:
        try:
            pts = [(float(body['lat']), float(body['lon']))]
        except (TypeError, ValueError):
            return jsonify(ok=False, error='bad lat/lon'), 400
    else:
        return jsonify(ok=False, error='missing waypoints or lat/lon'), 400

    with nav_lock:
        n = nav.set_waypoints(pts)
    log.info(f"[nav] {n} waypoint(s) staged")
    return jsonify(ok=True, waypoints=n)


@app.route('/api/go', methods=['POST'])
def api_go():
    snap = nav.snapshot()
    if not snap.waypoints:
        return jsonify(ok=False, error='no destination — draw a path first'), 400
    _set_nav_active(True)
    log.info("[nav] GO")
    return jsonify(ok=True)


@app.route('/api/stop', methods=['POST'])
def api_stop():
    _set_nav_active(False)
    with nav_lock:
        nav.stop()
    return jsonify(ok=True)


def gen_frames():
    while True:
        if camera is None:
            time.sleep(0.1); continue
        frame, _ = camera.read()
        if frame is None:
            time.sleep(0.05); continue
        small = cv2.resize(frame, (480, 360))
        _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.066)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ========================================================================
# main
# ========================================================================
def main():
    global pi_client, camera, recorder, yolo, predictor

    p = argparse.ArgumentParser()
    p.add_argument('--pi', default=None, help='Pi IP address')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), 'data'))
    p.add_argument('--no-yolo', action='store_true')
    p.add_argument('--no-pi', action='store_true')
    p.add_argument('--no-camera', action='store_true')
    p.add_argument('--https', action='store_true',
                   help='Serve HTTPS with a self-signed cert (needed for browser geolocation).')
    p.add_argument('--model',
                   default=os.path.join(os.path.dirname(__file__), 'models', 'best.pt'),
                   help='Trained model path (.pt or .onnx). Pass empty to disable.')
    args = p.parse_args()

    if not args.no_camera:
        camera = Camera(device=args.camera)
        if not camera.start():
            log.error("Camera failed to start"); return
    else:
        log.info("[Camera] disabled")

    if not args.no_pi and args.pi:
        pi_client = PiClient(args.pi)
        if not pi_client.connect():
            log.warning("[PiClient] cannot connect — continuing without Pi")
            pi_client = None
        else:
            pi_client.start()
    else:
        log.info("[PiClient] disabled")

    if YOLO_AVAILABLE and not args.no_yolo and camera is not None:
        try:
            yolo = ObjectDetector(conf_threshold=0.4, device='cpu')
            threading.Thread(target=yolo_loop, daemon=True).start()
            log.info("[YOLO] started")
        except Exception as e:
            log.warning(f"[YOLO] failed: {e}")
            yolo = None

    if PREDICTOR_AVAILABLE and args.model and os.path.exists(args.model):
        try:
            predictor = Predictor(args.model, device='cpu')
            log.info(f"[Model] loaded {args.model}")
        except Exception as e:
            log.warning(f"[Model] failed to load: {e}")
            predictor = None
    else:
        log.info(f"[Model] not loaded")

    recorder = DataRecorder(args.data_dir)
    threading.Thread(target=recorder.record_loop, daemon=True).start()
    threading.Thread(target=nav_loop, daemon=True).start()

    print("=" * 60)
    print("  AUTONOMOUS VEHICLE CONTROL")
    print("=" * 60)
    proto = 'https' if args.https else 'http'
    print(f"  Open in browser: {proto}://0.0.0.0:{args.port}")
    print(f"  Pi:       {args.pi or '<disabled>'}")
    print(f"  Camera:   device {args.camera if not args.no_camera else '<disabled>'}")
    print(f"  YOLO:     {'ON' if yolo else 'OFF'}")
    print(f"  Model:    {'ON  '+args.model if predictor else 'OFF'}")
    print(f"  Data:     {args.data_dir}  (existing samples: {recorder.samples_written})")
    print(f"  Maps key: {'set' if _load_maps_key() else 'NONE'}")
    print("=" * 60)

    try:
        ssl_ctx = 'adhoc' if args.https else None
        app.run(host='0.0.0.0', port=args.port, threaded=True, ssl_context=ssl_ctx)
    except KeyboardInterrupt:
        pass
    finally:
        if recorder: recorder.close()
        if pi_client: pi_client.close()
        if camera: camera.stop()
        print("\nStopped.")


if __name__ == '__main__':
    main()
