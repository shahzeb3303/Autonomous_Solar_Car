#!/usr/bin/env python3
"""Smoke test: does the autonomous loop actually RUN, and does the car get closer to B?

This exists because of a bug that cost a whole field session. `_nav_tick()` called
`_apply_obstacle_override(..., stuck_s=...)` but the function had no `stuck_s`
parameter. It raised TypeError on the FIRST tick after every GO, autonomous
disengaged instantly, and the car never moved. Manual driving kept working
perfectly (different code path), so it looked like a mysterious "GO doesn't work".

Nothing caught it, because every existing test exercised the components in
isolation — the controller, the obstacle layer, the geometry — and they were all
fine. Nobody ever ran the loop that WIRES THEM TOGETHER.

So: run the real nav_loop, against a fake Pi, and assert the car gets closer to B.

    ./venv/bin/python tests/test_nav_smoke.py
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
logging.disable(logging.INFO)

import web_control as W  # noqa: E402

M_PER_DEG_LAT = 111320.0
START = (33.54890, 73.18330)
GOAL = (33.54980, 73.18330)          # 100 m due north


class FakePi:
    """Behaves like the real Pi did in the field: open space, good fix, no heading
    at standstill (heading_valid False), and it moves when told to."""
    connected = True

    def __init__(self):
        self.lat, self.lon = START
        self.moving = False
        self.drive, self.steer, self.speed = 'STOP', 'STEER_STOP', 0
        self.commands = []

    def get_status(self):
        if self.moving:
            self.lat += 0.7 / M_PER_DEG_LAT      # 0.7 m/s, one nav tick = 0.1 s... close enough
        return {
            'distances': {k: 400.0 for k in ('FL', 'FR', 'FW', 'BC', 'LS', 'RS')},
            'sensor_health': {k: True for k in ('FL', 'FR', 'FW', 'BC', 'LS', 'RS')},
            'steer_current_angle': 0.0,
            'safety_violation': 'NONE',
            'gps': {
                'valid': True, 'stale': False, 'fix_age_s': 0.2,
                'lat': self.lat, 'lon': self.lon,
                'speed_mps': 0.7 if self.moving else 0.01,
                # The real Pi sends null + heading_valid False at standstill.
                'heading_deg': None, 'heading_valid': False,
                'satellites': 10, 'hdop': 1.0,
            },
        }

    def get_state_snapshot(self):
        return self.drive, self.steer, self.speed

    def set_command(self, drive=None, steer=None, speed=None, angle=None):
        if drive is not None:
            self.drive = drive
        if speed is not None:
            self.speed = speed
        self.commands.append((drive, speed, angle))
        if drive in ('FORWARD', 'BACKWARD') and (speed or 0) > 0:
            self.moving = True
        if drive == 'STOP':
            self.moving = False

    def safe_stop(self):
        self.set_command(drive='STOP', speed=0)


def main():
    fake = FakePi()
    W.pi_client = fake
    W.camera = W.predictor = W.recorder = None

    client = W.app.test_client()
    client.post('/api/destination', json={'waypoints': [
        {'lat': START[0], 'lon': START[1]},
        {'lat': GOAL[0], 'lon': GOAL[1]},
    ]})
    go = client.post('/api/go').get_json()
    assert go.get('ok'), f"GO was refused: {go}"

    threading.Thread(target=W.nav_loop, daemon=True).start()

    start_dist = None
    for _ in range(60):                       # ~6 s
        time.sleep(0.1)
        snap = W.nav.snapshot()
        if snap.distance_m is not None and start_dist is None:
            start_dist = snap.distance_m

    snap = W.nav.snapshot()
    failures = []

    # 1. The loop must still be running. A TypeError in the tick used to disengage
    #    autonomous on the very first pass — this is the regression that hurt.
    if not W.nav_active:
        failures.append("autonomous DISENGAGED — the nav loop crashed. "
                        "Check the traceback above (logging is disabled here).")

    # 2. The car must actually be commanded to drive.
    if not fake.commands or fake.commands[-1][0] != 'FORWARD':
        failures.append(f"car is not being commanded FORWARD "
                        f"(last: {fake.commands[-1] if fake.commands else None})")

    # 3. THE ONE THAT MATTERS: the car must get CLOSER to B.
    if start_dist is None or snap.distance_m is None:
        failures.append("no distance-to-goal was ever computed")
    elif snap.distance_m >= start_dist:
        failures.append(f"car is NOT getting closer to B: "
                        f"{start_dist:.1f}m -> {snap.distance_m:.1f}m")

    print(f"state       : {snap.state}")
    print(f"heading src : {snap.heading_source}")
    print(f"distance    : {start_dist:.1f} m -> {snap.distance_m:.1f} m")
    print(f"commands    : {len(fake.commands)}  last={fake.commands[-1]}")
    print(f"car moved   : {(fake.lat - START[0]) * M_PER_DEG_LAT:.1f} m")
    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("PASS — the nav loop runs, commands the car, and closes on B.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
