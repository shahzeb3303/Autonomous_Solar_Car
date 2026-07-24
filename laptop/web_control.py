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

# Manual d-pad steering: real proportional angle, accumulated per tap.
manual_steer_angle = 0.0
manual_angle_lock = threading.Lock()
MANUAL_STEER_STEP = 0.3  # per-tap nudge; ~3-4 taps to reach full lock

# The speed the UI slider is sitting at. A d-pad tap carries no speed of its own,
# so without this a FORWARD sent right after a STOP (or after taking over from
# autonomous, which calls safe_stop() and zeroes the speed) would go out with
# speed=0. The Pi currently rescues that by silently turning speed<=0 into 50 —
# but that is a bug we have asked them to FIX, and the moment they do, manual
# drive would stop working entirely. So: always send a real speed.
manual_speed = 50

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
def _audit_telemetry_schema(msg: dict) -> None:
    """Check the Pi's FIRST telemetry line against what we actually depend on.

    The laptop and the Pi are separate repos evolving separately. A renamed or
    re-nested key does not raise — it silently reads as a default, and the car
    quietly stops trusting a sensor or loses its GPS fix with no error anywhere.
    That failure mode is nearly impossible to spot in the field, so: fail loud,
    once, at connect time.
    """
    gps = msg.get('gps') or {}
    dist = msg.get('distances') or {}
    checks = [
        ('distances.FL/FR/FW/BC/LS/RS', all(k in dist for k in
                                            ('FL', 'FR', 'FW', 'BC', 'LS', 'RS')), True),
        ('gps.valid',                   'valid' in gps,                            True),
        ('gps.lat/lon',                 'lat' in gps and 'lon' in gps,             False),
        ('gps.heading_deg',             'heading_deg' in gps,                      False),
        ('gps.heading_valid',           'heading_valid' in gps,                    False),
        ('gps.heading_source',          'heading_source' in gps,                   False),
        ('gps.fix_type (M8L DR)',       'fix_type' in gps,                         False),
        ('gps.fusion_mode (M8L DR)',    'fusion_mode' in gps,                      False),
        ('gps.fix_time',                'fix_time' in gps,                         False),
        ('gps.stale',                   'stale' in gps,                            False),
        ('gps.speed_mps',               'speed_mps' in gps,                        False),
        ('sensor_health',               isinstance(msg.get('sensor_health'), dict), False),
        ('steer_current_angle',         'steer_current_angle' in msg,              False),
    ]
    log.info("[PiClient] telemetry schema check:")
    for name, ok, required in checks:
        if ok:
            log.info("    ok       %s", name)
        elif required:
            log.error("    MISSING  %s  <-- REQUIRED. Navigation will not work.", name)
        else:
            log.warning("    absent   %s  <-- degraded; check the Pi's field name", name)
    if not isinstance(msg.get('sensor_health'), dict):
        log.warning("    -> without sensor_health, a DEAD sensor cannot be told apart "
                    "from a clear path; all sensors are assumed healthy.")
    log.info("    keys on the wire: %s", sorted(msg.keys()))


class PiClient:
    def __init__(self, ip: str, port: int = 5555):
        self.ip = ip
        self.port = port
        self.sock = None
        self._schema_checked = False
        self.connected = False
        self.running = True
        self.status: Optional[dict] = None
        self.status_lock = threading.Lock()
        self.drive = 'STOP'
        self.steer = 'STEER_STOP'
        self.speed = 50
        self.angle = None  # optional proportional steering target [-1..+1];
                           # None = discrete/ramp mode (manual control, older Pi)
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
                d, s, sp, a = self.drive, self.steer, self.speed, self.angle
            try:
                payload = {'command': d, 'steer': s, 'speed': sp}
                if a is not None:
                    payload['angle'] = a
                self.sock.sendall(json.dumps(payload).encode())
            except Exception:
                self.connected = False
                return
            # 10 Hz. Was 5 Hz (0.2s) — which added up to 200ms of pure latency
            # to every obstacle reaction, and left almost no margin against the
            # Pi's 0.5s watchdog when WiFi jittered (a suspected stutter cause).
            time.sleep(0.1)

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
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not self._schema_checked:
                        self._schema_checked = True
                        _audit_telemetry_schema(msg)
                    with self.status_lock:
                        self.status = msg
            except socket.timeout:
                continue
            except Exception:
                self.connected = False
                return

    def start(self):
        threading.Thread(target=self._sender, daemon=True).start()
        threading.Thread(target=self._receiver, daemon=True).start()

    def set_command(self, drive=None, steer=None, speed=None, angle=None):
        with self.cmd_lock:
            if drive is not None: self.drive = drive
            if steer is not None: self.steer = steer
            if speed is not None: self.speed = max(0, min(100, int(speed)))
            if angle is not None: self.angle = max(-1.0, min(1.0, float(angle)))

    def get_state_snapshot(self):
        with self.cmd_lock:
            return self.drive, self.steer, self.speed

    def get_status(self):
        with self.status_lock:
            return dict(self.status) if self.status else None

    def safe_stop(self):
        with self.cmd_lock:
            self.angle = None  # drop back to discrete/ramp mode
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
    # Per-run state lived on the function object and PERSISTED across STOP->GO,
    # so a new run inherited the previous run's last action and commitment timer.
    # Clear it on every transition.
    nav_loop._last_action = None
    nav_loop._last_action_t = 0.0
    nav_prev_action = STOP
    if not value and pi_client:
        pi_client.safe_stop()


def _apply_obstacle_override(wanted: str, sensors: dict, yolo_snap: dict,
                             health: dict | None = None,
                             stuck_s: float = 0.0):
    return obstacle_decide(sensors, yolo_snap, wanted_action=wanted, health=health,
                           stuck_s=stuck_s)


# GPS fix quality required before GO — only enforced when dead reckoning is NOT
# yet fused (once DR is active the M8L vouches for its own solution; see api_go).
# HDOP <2 good, 2-5 perfectly fine for a ground vehicle, >5 poor. 3.0 was too
# strict — it rejected a real HDOP-3.2 / 8-sat fix in the field.
GPS_MAX_HDOP = 5.0
GPS_MIN_SATS = 6


class GpsFreshness:
    """Detect a FROZEN GPS feed — a fix that says 'valid' but never updates.

    Observed live on this vehicle, parked: the Pi reported the identical lat/lon
    for 200 consecutive samples over 10 s (zero position wander — a real receiver
    always jitters by a metre or two), while simultaneously reporting a constant
    speed_mps of 0.732 and a constant heading_deg of 0.

    That combination is lethal. 0.732 m/s is ABOVE our COG trust threshold, so
    the heading estimator would have concluded "we're moving at 0.73 m/s, so the
    course is trustworthy" and steered on a heading that never changes — driving
    confidently in one fixed direction, forever, with a completely fake fix.

    `valid: true` means "the receiver has a fix". It does NOT mean "this data is
    fresh". A stale-but-valid fix is more dangerous than no fix at all, because
    no fix fails safe and a stale one does not.
    """
    STALE_S = 3.0

    def __init__(self):
        self._last_fix = None
        self._changed_at = None
        self._warned = False

    def is_live(self, gps: dict, now: float) -> bool:
        fix = (gps.get('lat'), gps.get('lon'), gps.get('speed_mps'))
        if fix != self._last_fix:
            self._last_fix = fix
            self._changed_at = now
            self._warned = False
            return True
        if self._changed_at is None:
            self._changed_at = now
            return True
        age = now - self._changed_at
        if age >= self.STALE_S:
            if not self._warned:
                self._warned = True
                log.error("[nav] GPS FEED IS FROZEN — identical lat/lon/speed for "
                          "%.1fs. Treating as NO FIX. (A stale 'valid' fix would "
                          "steer the car on a heading that never updates.)", age)
            return False
        return True


gps_freshness = GpsFreshness()


# The behaviour-cloned model is OFF in the autonomous path by default.
#
# Why: it is trained on human driving and, as the code below itself admitted, it
# "doesn't know where B is, so it cannot pick direction toward the goal." Yet the
# old gate let it choose the TURN direction whenever GPS heading was unusable —
# which, on a slow vehicle, is most of the time. A model with no goal knowledge
# was steering the car toward the goal. Get GPS+heading navigation working
# standalone first; then re-enable this deliberately, one gate at a time.
ENABLE_MODEL = False


def nav_loop():
    """GPS Pure-Pursuit -> obstacle override -> Pi.

    Steering is a REAL PROPORTIONAL ANGLE from the Pure-Pursuit law
    (nav/controller.py), sent in the Pi's `angle` field and held by its
    closed-loop PD controller against potentiometer feedback.

    It used to be bang-bang: the controller quantised a precise heading error to
    FORWARD/TURN_LEFT/TURN_RIGHT and this loop mapped that to an angle of exactly
    -1.0 / 0.0 / +1.0 — full lock or dead centre, nothing in between. The car
    drove straight while up to 45 degrees off course, then slammed to full lock.
    That is why it could not hold a line from A to B.
    """
    global nav_prev_action
    log.info("[nav] loop started — thread alive")

    while True:
        try:
            _nav_tick()
        except Exception:
            # This used to be fatal. nav_loop ran with NO exception handling, so a
            # single error killed the thread permanently: autonomous silently did
            # nothing forever afterwards, while manual driving kept working (it goes
            # through Flask routes, not this thread). Never again — log it, stop the
            # car, and keep the thread alive.
            log.exception("[nav] TICK FAILED — autonomous disengaged for safety")
            try:
                _set_nav_active(False)
            except Exception:
                pass
            time.sleep(0.5)


def _nav_tick():
    global nav_prev_action

    while True:
        if not nav_active or pi_client is None:
            time.sleep(0.2); continue

        status = pi_client.get_status()
        if status is None:
            time.sleep(0.1); continue

        gps = status.get('gps', {}) or {}
        # 'valid' means the receiver HAS a fix. It does NOT mean the fix is FRESH.
        # A frozen-but-valid feed reports a plausible speed and a constant heading
        # and would be steered on as if real — confirmed on this vehicle: after a
        # signal loss the Pi served one stale fix as valid for 200 consecutive
        # samples, with speed_mps pinned at 0.732 (above our heading-trust
        # threshold) while the car sat still.
        #
        # The Pi now sends an authoritative `stale` flag (fix_age_s > 2s). Prefer
        # it; keep our own value-watching heuristic as a fallback for an older Pi.
        fix_stale = bool(gps.get('stale', False))
        valid = (bool(gps.get('valid', 0))
                 and not fix_stale
                 and gps_freshness.is_live(gps, time.time()))
        if fix_stale and not getattr(nav_loop, '_warned_stale', False):
            nav_loop._warned_stale = True
            log.error("[nav] GPS fix is STALE (age %.1fs) — not navigating on it.",
                      gps.get('fix_age_s', -1.0))
        elif not fix_stale:
            nav_loop._warned_stale = False
        lat = gps.get('lat') if valid else None
        lon = gps.get('lon') if valid else None
        speed_mps = gps.get('speed_mps', 0.0) if valid else 0.0

        # ---- HEADING: the GPS is a u-blox NEO-M8L with Automotive Dead Reckoning ----
        #
        # The Pi configured the M8L's ADR (UBX NAV-PVT + ESF-STATUS) and now hands us
        # a FACTORY-FUSED vehicle heading that is valid at ANY speed — even stopped.
        # It tells us which kind of heading each sample is:
        #     heading_source == "fused" -> M8L dead-reckoning fusion. Trust at any
        #                                   speed, including 0. THE PRIMARY SOURCE.
        #     heading_source == "cog"   -> plain course-over-ground; needs motion.
        #     heading_source == "none"  -> no heading this sample.
        # heading_deg is null (never a fabricated 0.0) whenever it is unknown.
        #
        # So: route a "fused" sample into the estimator's fused slot (used directly,
        # unconditionally), and a "cog" sample into the course slot (smoothed, and
        # only trusted above the speed threshold). This replaces the raw-gyro fusion
        # we built — the M8L does the fusion internally and far better than a bench
        # gyro could. Our course-made-good fallback stays only for when the fusion is
        # still calibrating (heading_source != "fused").
        h_src = gps.get('heading_source')
        h_deg = gps.get('heading_deg')
        h_ok = valid and (gps.get('heading_valid') is not False) and h_deg is not None
        fused = h_deg if (h_ok and h_src == "fused") else None
        heading = h_deg if (h_ok and h_src == "cog") else None
        # An older Pi that predates heading_source: treat any heading as course, and
        # let heading.py's 0.0-sentinel guard catch a fabricated north.
        if h_src is None and h_ok:
            heading = h_deg

        sensors = status.get('distances', {}) or {}
        # Per-sensor health from the Pi. Absent on an older Pi -> assume healthy.
        health = status.get('sensor_health') or None
        yolo_snap = get_yolo_snapshot()

        # ---- The bench MPU-6500 gyro is ABANDONED ----
        # It read fine on a laptop but never on the Pi (I2C/power), and it is moot
        # now: the M8L's internal fused IMU supersedes it. The Pi still sends
        # imu.gyro_z but permanently valid:false — ignore it.
        gyro_z, gyro_ok = None, False

        # Were we reversing on the LAST tick? Course-over-ground reports the
        # direction of TRAVEL, so while reversing it is 180 deg from the nose. This
        # matters only for the "cog" fallback — the M8L fused heading already tracks
        # the vehicle body, so it does not need this correction (and the estimator
        # only applies it to the cog path).
        was_reversing = getattr(nav_loop, '_reversing', False)

        with nav_lock:
            wanted = nav.tick(lat, lon, heading, speed_mps,
                              gyro_z_dps=gyro_z, gyro_valid=gyro_ok,
                              fused_heading_deg=fused,
                              reversing=was_reversing)

        snap = nav.snapshot()
        if snap.state == "ARRIVED":
            log.info("[nav] ARRIVED — stopping autonomous mode")
            _set_nav_active(False)
            continue
        if snap.state == "NO_HEADING":
            log.error("[nav] %s", snap.reason)
            _set_nav_active(False)
            continue

        # ---- ML model usage (gated) ----
        # Model is behaviour-cloned -- it doesn't know where B is, so it
        # cannot pick direction toward the goal. But it can:
        #   1. Refine FORWARD -> SLOW_DOWN in cluttered scenes
        #   2. Pick a TURN side when GPS heading is unavailable
        #   3. Suggest REVERSE / REVERSE_LEFT/RIGHT when stuck
        # Cannot escalate to STOP (avoids the GO-time deadlock).
        # NOTE: gated OFF by default now — see ENABLE_MODEL above.
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
        #
        # `stuck_s` is how long we have been stopped AND blocked. It is the ONLY
        # thing that can authorise a REVERSE. Reversing inverts GPS course-over-
        # ground (our only heading source), so it must never be routine — in the
        # first outdoor run the car reversed on almost every tick and never got a
        # single metre closer to B.
        stuck_since = getattr(nav_loop, '_stuck_since', None)
        stuck_s = (time.time() - stuck_since) if stuck_since else 0.0

        final, reason = _apply_obstacle_override(wanted, sensors, yolo_snap, health,
                                                 stuck_s=stuck_s)
        overridden = (final != wanted)

        # Track "stopped and blocked" so a genuinely trapped car can escape, while a
        # car that is merely slowing for something never reverses.
        if final == 'STOP':
            if stuck_since is None:
                nav_loop._stuck_since = time.time()
        else:
            nav_loop._stuck_since = None

        now = time.time()
        action_id = ACTION_NAMES.index(final) if final in ACTION_NAMES else STOP
        nav_prev_action = action_id

        # ---- hold an ESCAPE manoeuvre so it isn't aborted mid-way ----------
        # The old "commitment window" applied to every action, including steering —
        # so at err=+4.3 deg (needing steer=+0.03) the car was still being commanded
        # steer=-1.00 (full left lock) because it was 'committed' to a stale REVERSE.
        # The navigator and the steering were fighting each other.
        #
        # Now: ONLY a reverse-escape is held (for ESCAPE_HOLD_S, so the car actually
        # backs out instead of twitching). Everything else follows the controller
        # immediately — a proportional controller does not need anti-flap logic,
        # because it does not flap.
        ESCAPE_HOLD_S = 1.5
        escape_until = getattr(nav_loop, '_escape_until', 0.0)
        is_escape = final in ('REVERSE', 'REVERSE_LEFT', 'REVERSE_RIGHT')

        if is_escape:
            nav_loop._escape_until = now + ESCAPE_HOLD_S
            nav_loop._escape_action = final
        elif now < escape_until:
            final = getattr(nav_loop, '_escape_action', final)   # finish backing out
            reason = (reason or '') + ' [escape-hold]'
            action_id = ACTION_NAMES.index(final) if final in ACTION_NAMES else STOP
            overridden = True

        reversing = final in ('REVERSE', 'REVERSE_LEFT', 'REVERSE_RIGHT')

        # ---- steering + speed --------------------------------------------
        # Not overridden -> the Pure-Pursuit controller drives, with its CONTINUOUS
        # steering angle. A 5-degree error produces a 5-degree correction.
        # Overridden  -> we are avoiding; full lock is genuinely what we want.
        cmd = action_to_pi_command(action_id)
        drive_cmd = cmd['command']

        if not overridden:
            target_pos = snap.steer_norm            # continuous, -1..+1, + = RIGHT
            speed_cmd = snap.speed
        else:
            if final in ('TURN_LEFT', 'REVERSE_LEFT'):
                target_pos = -1.0
            elif final in ('TURN_RIGHT', 'REVERSE_RIGHT'):
                target_pos = +1.0
            else:
                target_pos = 0.0
            speed_cmd = cmd['speed']

        # The Pi treats `speed <= 0` as 50 (!), so a zero speed does NOT stop the
        # car — only `command: STOP` does. Never rely on speed to stop.
        if drive_cmd == 'STOP' or final == 'STOP':
            drive_cmd = 'STOP'
            speed_cmd = 0
            target_pos = 0.0

        # Discrete fallback direction, used only if the Pi ignores `angle`
        # (e.g. while its safety governor has an override active).
        if target_pos > 0.05:
            new_steer = 'RIGHT'
        elif target_pos < -0.05:
            new_steer = 'LEFT'
        else:
            new_steer = 'STEER_STOP'

        steer_pos = status.get('steer_current_angle')  # real pot feedback
        nav_loop._reversing = reversing
        # Surface WHY the car is not moving. A silent veto looks like "GO is broken".
        nav_loop._block = (reason if (overridden and speed_cmd == 0) else None)

        pi_client.set_command(drive=drive_cmd, steer=new_steer, speed=speed_cmd,
                              angle=target_pos)

        # ---- per-second log ----
        # `nav=` is the controller's own reason (which target it picked, the error,
        # the cross-track). `obs=` is the obstacle layer's reason. Previously only
        # the obstacle reason was printed, which hid what the navigator was aiming at.
        if now - getattr(nav_loop, '_last_log', 0) > 1.0:
            nav_loop._last_log = now
            fmin = min([v for v in (sensors.get('FL'), sensors.get('FR'),
                                    sensors.get('FW'))
                        if isinstance(v, (int, float)) and v > 0] or [999])
            log.info(
                "[nav] %s want=%s final=%s%s | dist=%s err=%s xt=%s hdg=%s/%s "
                "spd=%.2f gyro=%s | steer=%+0.2f wheel=%s pwr=%d | front=%.0fcm obs=%s | %s",
                snap.state, snap.wanted_action, final,
                ' (OVERRIDE)' if overridden else '',
                f"{snap.distance_m:.1f}m" if snap.distance_m is not None else '-',
                f"{snap.heading_err_deg:+.1f}" if snap.heading_err_deg is not None else '-',
                f"{snap.cross_track_m:+.1f}" if snap.cross_track_m is not None else '-',
                f"{snap.heading_deg:.0f}" if snap.heading_deg is not None else '--',
                snap.heading_source, speed_mps or 0.0, snap.gyro_sign,
                target_pos,
                f"{steer_pos:+.2f}" if steer_pos is not None else 'N/A',
                speed_cmd, fmin, reason or '-', snap.reason,
            )

        # 10 Hz. At 0.8 m/s the car covers 8cm per tick; at the old 5 Hz it
        # covered 16cm between decisions, which is a lot of blind travel when
        # the obstacle thresholds are only ~1m out.
        time.sleep(0.1)


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
</div>

<!-- Big, phone-readable GPS/heading badge. Green = fused heading ready (car can
     navigate even while stopped). Amber = course-only / still calibrating. Red = no fix. -->
<div id="gpsBadge" style="max-width:640px;margin:4px auto;padding:8px 12px;border-radius:8px;
     text-align:center;font-size:15px;font-weight:bold;background:#3a0000;color:#fff;">
    GPS: --
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
// Each LEFT/RIGHT tap nudges the Pi's real steering angle further (server
// tracks the accumulated angle) - no auto-release timer. The Pi holds that
// angle with real potentiometer feedback until you tap STR/STOP to recenter.
document.querySelectorAll('.btn').forEach(btn => {
    const drive = btn.dataset.drive || null;
    const steer = btn.dataset.steer || null;
    function tap(e) {
        e.preventDefault();
        document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        sendCmd(drive, steer);
    }
    btn.addEventListener('click', tap);
    btn.addEventListener('touchstart', tap, {passive:false});
});
const driveMap = {w:'FORWARD', s:'BACKWARD'};
const steerMap = {a:'LEFT', d:'RIGHT'};
document.addEventListener('keydown', e => {
    const k = e.key.toLowerCase();
    if (driveMap[k]) sendCmd(driveMap[k], null);
    else if (steerMap[k]) sendCmd(null, steerMap[k]);
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
            // M8L dead-reckoning fusion state — a big colour-coded badge so it is
            // readable on a phone while driving. GREEN = fused heading ready (car can
            // navigate even stopped); AMBER = course-only / still calibrating; RED =
            // no usable fix. Watch it go GREEN during the outdoor calibration drive.
            (function(){
                const badge = document.getElementById('gpsBadge'), g = s.gps || {};
                let bg = '#3a0000', txt;
                if (g.stale)            { bg='#3a0000'; txt='GPS ⚠ STALE — not navigating'; }
                else if (!g.valid)      { bg='#3a0000'; txt='GPS: no fix (need open sky)'; }
                else if (g.heading_source === 'fused') {
                    bg='#0a3d1a';
                    txt=`✅ HEADING LOCKED (fused ${(g.heading_deg||0).toFixed(0)}°) · fix ${g.fix_type||'?'} · ${g.satellites||'?'} sat`;
                } else {
                    bg='#4a3800';
                    const cal = (g.fusion_mode ? 'DR active' : 'calibrating…');
                    txt=`⏳ heading: ${g.heading_source||'none'} · ${cal} · fix ${g.fix_type!==undefined?g.fix_type:'?'} · ${g.satellites||'?'} sat — DRIVE to calibrate`;
                }
                badge.style.background = bg;
                badge.textContent = txt;
            })();
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
            const el = document.getElementById('navStatus');
            if (n.active && n.blocked_by) {
                // The car is being vetoed by the obstacle layer. Say so — a silent
                // veto is indistinguishable from "the GO button is broken".
                el.textContent = '\u26D4 BLOCKED: ' + n.blocked_by
                                 + ' \u2014 clear the obstacle or move the car';
                el.style.color = '#f44';
                el.style.fontWeight = 'bold';
            } else {
                let line = 'Nav: ' + n.state;
                if (n.distance_m != null) line += ' | ' + n.distance_m.toFixed(1) + ' m to B';
                if (n.wanted_action) line += ' | wants ' + n.wanted_action;
                if (n.reason) line += ' | ' + n.reason;
                el.textContent = line;
                el.style.color = '#0cf';
                el.style.fontWeight = 'normal';
            }
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
    global manual_steer_angle
    drive = request.args.get('drive')
    steer = request.args.get('steer')
    speed = request.args.get('speed')

    # A speed-only change (dragging the slider) is NOT a takeover. It used to
    # silently abort an autonomous run mid-drive.
    is_takeover = (drive is not None or steer is not None)
    if nav_active and is_takeover:
        log.info("[nav] manual takeover — autonomous disengaged")
        _set_nav_active(False)
        with manual_angle_lock:
            manual_steer_angle = 0.0  # start manual control from a known-centered angle

    # Manual d-pad steering: send a real proportional angle (accumulating
    # per tap) instead of relying on the Pi's ramp - a single 200ms-ish tap
    # is too brief for the ramp to move the wheel noticeably before the next
    # command arrives. STEER_STOP always recenters immediately.
    if steer is not None:
        with manual_angle_lock:
            if steer == 'LEFT':
                manual_steer_angle = max(-1.0, manual_steer_angle - MANUAL_STEER_STEP)
            elif steer == 'RIGHT':
                manual_steer_angle = min(1.0, manual_steer_angle + MANUAL_STEER_STEP)
            elif steer == 'STEER_STOP':
                manual_steer_angle = 0.0

    with manual_angle_lock:
        angle_to_send = manual_steer_angle

    # ⚠ The Pi maps `speed <= 0` to 50 (remote_server.get_latest_speed), so
    # sliding the speed to zero while driving does NOT stop the car — it drops
    # it to 50%. Zero speed must be expressed as an explicit STOP command.
    global manual_speed
    if speed is not None:
        try:
            manual_speed = max(0, min(100, int(speed)))
        except (TypeError, ValueError):
            pass
        if manual_speed == 0:
            drive = 'STOP'

    # Always send a real speed with a motion command. A d-pad tap carries no
    # speed, and safe_stop() leaves the stored speed at 0 — so without this, the
    # first FORWARD after any STOP would be a zero-speed FORWARD.
    speed_val = 0 if drive == 'STOP' else (manual_speed or 50)

    if pi_client:
        pi_client.set_command(drive=drive, steer=steer, speed=speed_val,
                              angle=angle_to_send)
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
    with manual_angle_lock:
        manual_angle = manual_steer_angle
    return jsonify(
        connected=bool(pi_client and pi_client.connected),
        drive=drive, steer=steer, speed=speed,
        actual_speed=(st or {}).get('actual_speed', 0),
        distances=(st or {}).get('distances', {}),
        gps=(st or {}).get('gps', {'valid': False, 'lat': 0.0, 'lon': 0.0}),
        steer_current_angle=(st or {}).get('steer_current_angle'),
        manual_steer_angle=manual_angle,
        recording=recorder.is_recording() if recorder else False,
        samples=recorder.samples_written if recorder else 0,
        yolo=get_yolo_snapshot(),
        nav={
            'state': snap.state, 'active': nav_active,
            'blocked_by': getattr(nav_loop, '_block', None),
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
    if pi_client is None or not pi_client.connected:
        return jsonify(ok=False, error='not connected to the Pi'), 400
    st = pi_client.get_status() or {}
    gps = st.get('gps') or {}
    if not gps.get('valid'):
        return jsonify(ok=False, error='no GPS fix — cannot navigate'), 400
    if gps.get('stale'):
        return jsonify(ok=False, error=(
            f"GPS fix is STALE ({gps.get('fix_age_s', '?')}s old) — the receiver has "
            f"stopped updating. Wait for a live fix.")), 400

    # A "valid" fix is not necessarily a GOOD fix. HDOP is the geometric quality of
    # the satellite spread: <2 is good, 2-5 is fine for a ground vehicle, >5 is poor.
    #
    # If the M8L's dead reckoning is fused and active (fix_type 4 = GNSS+DR, or the
    # heading is already "fused"), the module is IMU-aiding the solution and a
    # middling HDOP no longer matters — skip the geometry gate entirely and trust
    # the module's own fused solution.
    hdop = gps.get('hdop')
    sats = gps.get('satellites')
    dr_fused = (gps.get('fix_type') == 4) or (gps.get('heading_source') == 'fused')

    if not dr_fused:
        if hdop is not None and hdop > GPS_MAX_HDOP:
            return jsonify(ok=False, error=(
                f'GPS fix too weak: HDOP {hdop} (need < {GPS_MAX_HDOP}), {sats} sats. '
                f'Wait a moment for the fix to improve, or do the dead-reckoning '
                f'calibration drive so the M8L fusion takes over.'
            )), 400
        if sats is not None and sats < GPS_MIN_SATS:
            return jsonify(ok=False, error=(
                f'only {sats} satellites (need >= {GPS_MIN_SATS}). Wait for more.'
            )), 400
    log.info("[nav] GPS ok: %s sats, HDOP %s, dr_fused=%s", sats, hdop, dr_fused)
    with nav_lock:
        nav.start_run()          # clear heading estimate + per-run state
    _set_nav_active(True)
    log.info("[nav] ======== GO ======== %d waypoint(s), nav_active=%s",
             len(snap.waypoints), nav_active)
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
def _open_camera(preferred: int) -> Optional[Camera]:
    """Open the camera, falling back to any working device — and NEVER fatal.

    This used to be:
        if not camera.start():
            log.error("Camera failed to start"); return      # <-- returns from main()!
    ...which meant a missing camera KILLED THE WHOLE APPLICATION before Flask even
    started. No web UI, no nav loop, no manual control, no error the user could act
    on beyond one line of OpenCV noise.

    That is exactly what happened: the USB camera re-enumerated away from
    /dev/video4, and the entire vehicle stack refused to boot because of it.

    Navigation does not need a camera. Obstacle avoidance runs on ultrasonics. The
    camera is used only for the video feed and the (currently disabled) YOLO/ML
    layer. Losing it must degrade the system, not stop it.
    """
    import glob
    cam = Camera(device=preferred)
    if cam.start():
        log.info("[Camera] opened /dev/video%d", preferred)
        return cam

    present = sorted(int(p.rsplit('video', 1)[1]) for p in glob.glob('/dev/video*')
                     if p.rsplit('video', 1)[1].isdigit())
    log.warning("[Camera] /dev/video%d did not open. Devices present: %s",
                preferred, present or 'NONE')
    for idx in present:
        if idx == preferred:
            continue
        cam = Camera(device=idx)
        if cam.start():
            log.warning("[Camera] FELL BACK to /dev/video%d. Pass --camera %d next "
                        "time, or re-plug the camera you wanted.", idx, idx)
            return cam

    log.error("[Camera] NO WORKING CAMERA. Continuing WITHOUT it — driving and "
              "navigation are unaffected (obstacle avoidance uses the ultrasonics); "
              "the video feed and YOLO are disabled. Pass --no-camera to silence this.")
    return None


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
        camera = _open_camera(args.camera)
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
