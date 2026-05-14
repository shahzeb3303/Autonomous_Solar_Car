"""GPS waypoint navigation state machine.

Two modes, picked automatically by the number of waypoints:

* 1 waypoint  -> go-to-goal (point bearing). Simple, oscillates when target
                 is close because there's no path to track.
* 2+ waypoints -> Pure Pursuit along the polyline. The controller picks a
                 "carrot" point LOOKAHEAD_M ahead on the path and aims at it.
                 This is the standard approach for GPS-guided ground vehicles
                 (Coulter 1992; used by Stanley, MIT Cheetah, Nav2, etc.).

The car still emits coarse FORWARD / TURN_LEFT / TURN_RIGHT / STOP actions
that the outer loop composes with the ML model and obstacle override.
"""

from __future__ import annotations
import threading
import time
from dataclasses import dataclass

from .geo import (
    haversine_m, bearing_deg, heading_error_deg, lookahead_point,
)

# --- Tunables ---
ARRIVE_M = 3.0                    # waypoint reached if within this many metres
CALIBRATE_S = 3.0                 # forced FORWARD-at-max for this long after GO
                                  # (drives a few metres so GPS COG locks in)
TURN_ENTER_DEG = 45.0             # error > this -> start turning
TURN_EXIT_DEG = 15.0              # while turning, stop only after error drops below this
HEADING_MIN_SPEED_MPS = 0.35      # below this, GPS heading is unreliable
HEADING_EMA_ALPHA = 0.25          # low-pass filter on GPS heading (0=none, 1=raw)
LOOKAHEAD_M = 15.0                # Pure-Pursuit lookahead distance.
                                  # Larger = smoother / more straight-line stable.
                                  # Smaller = follows curves tighter (but oscillates).


@dataclass
class Waypoint:
    lat: float
    lon: float


@dataclass
class NavSnapshot:
    state: str
    waypoints: list[dict]
    active_idx: int
    distance_m: float | None
    bearing_to_target: float | None
    heading_err_deg: float | None
    wanted_action: str
    reason: str
    cross_track_m: float | None = None
    lookahead_lat: float | None = None
    lookahead_lon: float | None = None


class NavController:
    """GPS waypoint follower. Thread-safe via an internal lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self._waypoints: list[Waypoint] = []
        self._idx = 0
        self._state = "IDLE"
        self._calib_started: float | None = None
        self._last_dist: float | None = None
        self._last_bearing: float | None = None
        self._last_err: float | None = None
        self._last_wanted = "STOP"
        self._reason = ""
        self._smoothed_heading: float | None = None
        self._last_cross_track: float | None = None
        self._last_lookahead: tuple[float, float] | None = None

    # ---------- public api ----------

    def set_waypoints(self, points: list[tuple[float, float]]) -> int:
        with self._lock:
            self._waypoints = [Waypoint(*p) for p in points if p is not None]
            self._idx = 0
            self._state = "CALIBRATING" if self._waypoints else "IDLE"
            self._calib_started = time.time() if self._waypoints else None
            self._reason = "new waypoints"
            return len(self._waypoints)

    def stop(self) -> None:
        with self._lock:
            self._waypoints = []
            self._idx = 0
            self._state = "IDLE"
            self._calib_started = None
            self._last_wanted = "STOP"
            self._reason = "stopped"

    def snapshot(self) -> NavSnapshot:
        with self._lock:
            return NavSnapshot(
                state=self._state,
                waypoints=[{"lat": w.lat, "lon": w.lon} for w in self._waypoints],
                active_idx=self._idx,
                distance_m=self._last_dist,
                bearing_to_target=self._last_bearing,
                heading_err_deg=self._last_err,
                wanted_action=self._last_wanted,
                reason=self._reason,
                cross_track_m=self._last_cross_track,
                lookahead_lat=(self._last_lookahead[0] if self._last_lookahead else None),
                lookahead_lon=(self._last_lookahead[1] if self._last_lookahead else None),
            )

    # ---------- core tick ----------

    def tick(self, lat: float | None, lon: float | None,
             gps_heading_deg: float | None, gps_speed_mps: float | None) -> str:
        with self._lock:
            if not self._waypoints or self._state in ("IDLE", "ARRIVED"):
                self._last_wanted = "STOP"
                return "STOP"

            if lat is None or lon is None:
                self._last_wanted = "STOP"
                self._reason = "no gps"
                return "STOP"

            cur = (lat, lon)
            final_wp = self._waypoints[-1]
            final_pt = (final_wp.lat, final_wp.lon)

            # Arrival check is always against the FINAL waypoint.
            dist_to_final = haversine_m(cur, final_pt)
            self._last_dist = dist_to_final
            if dist_to_final <= ARRIVE_M:
                self._state = "ARRIVED"
                self._last_wanted = "STOP"
                self._reason = "arrived"
                return "STOP"

            # ---- choose the steering target ----
            # 1+ waypoint -> point bearing
            # 2+ waypoints -> Pure Pursuit along the polyline (path following)
            path = [(w.lat, w.lon) for w in self._waypoints]
            if len(path) >= 2:
                # If car is far from path start AND has not yet entered the path,
                # head straight for the first waypoint to "enter" the path.
                d_to_start = haversine_m(cur, path[0])
                if d_to_start > LOOKAHEAD_M and d_to_start > ARRIVE_M:
                    target = path[0]
                    self._last_cross_track = None
                    self._last_lookahead = target
                    target_label = "A"
                else:
                    target, cross, _seg = lookahead_point(path, cur, LOOKAHEAD_M)
                    self._last_cross_track = cross
                    self._last_lookahead = target
                    target_label = "carrot"
            else:
                target = final_pt
                self._last_cross_track = None
                self._last_lookahead = target
                target_label = "B"

            brg = bearing_deg(cur, target)
            self._last_bearing = brg

            # ---- heading from GPS COG, low-passed ----
            heading_usable = (
                gps_heading_deg is not None
                and gps_speed_mps is not None
                and gps_speed_mps >= HEADING_MIN_SPEED_MPS
            )
            if heading_usable:
                if self._smoothed_heading is None:
                    self._smoothed_heading = gps_heading_deg
                else:
                    diff = (gps_heading_deg - self._smoothed_heading + 540.0) % 360.0 - 180.0
                    self._smoothed_heading = (self._smoothed_heading
                                              + HEADING_EMA_ALPHA * diff) % 360.0

            if self._state == "CALIBRATING":
                elapsed = (time.time() - self._calib_started) if self._calib_started else 0
                # Stay in calibration for the FULL CALIBRATE_S window AND wait
                # for heading to be usable. This is the user's idea: drive
                # forward at max speed first, lock in heading from motion,
                # then start steering. Don't exit just because COG flickered
                # valid for one noisy tick.
                if elapsed >= CALIBRATE_S and heading_usable:
                    self._state = "NAVIGATING"
                    self._reason = "calibration complete — heading locked"
                else:
                    self._last_wanted = "FORWARD"
                    self._last_err = None
                    self._reason = (f"calibrating {elapsed:0.1f}/{CALIBRATE_S:.1f}s"
                                    + (" (heading not yet usable)" if not heading_usable else ""))
                    return "FORWARD"

            if not heading_usable:
                self._last_err = None
                self._last_wanted = "FORWARD"
                self._reason = "heading lost, nudging fwd"
                return "FORWARD"

            # ---- pick action from heading error to target ----
            err = heading_error_deg(self._smoothed_heading or gps_heading_deg, brg)
            self._last_err = err

            prev = self._last_wanted
            already_turning = prev in ("TURN_LEFT", "TURN_RIGHT")
            band = TURN_EXIT_DEG if already_turning else TURN_ENTER_DEG

            if abs(err) < band:
                act = "FORWARD"
            elif err > 0:
                act = "TURN_RIGHT"
            else:
                act = "TURN_LEFT"

            self._last_wanted = act
            ct_str = f" xtrack={self._last_cross_track:+.1f}m" if self._last_cross_track is not None else ""
            self._reason = f"target={target_label} err={err:+.1f}° dist={dist_to_final:.1f}m{ct_str}"
            return act
