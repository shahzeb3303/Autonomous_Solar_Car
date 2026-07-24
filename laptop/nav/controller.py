"""GPS waypoint navigation state machine.

Two modes, picked automatically by the number of waypoints:

* 1 waypoint  -> go-to-goal (point bearing).
* 2+ waypoints -> Pure Pursuit along the polyline. The controller picks a
                 "carrot" point LOOKAHEAD_M ahead on the path and aims at it.

WHAT CHANGED (2026-07-12) — read this before tuning anything:

  The controller now emits a CONTINUOUS steering command (`steer_norm`, -1..+1)
  computed with the real Pure-Pursuit law, instead of only a coarse action word.

  It used to compute a precise heading error and then throw it away, quantising
  to FORWARD / TURN_LEFT / TURN_RIGHT. web_control then mapped those to an angle
  of exactly -1.0 / 0.0 / +1.0. So the car drove dead straight while up to 45
  degrees off course, then slammed to FULL LOCK. That is a bang-bang relay, and
  it cannot track a line — it weaves and overshoots. That was the A->B bug.

  The coarse action is still emitted, because the obstacle-override layer and
  the (now-disabled) ML model speak that vocabulary. But steering is driven by
  `steer_norm`. The action word only *overrides* steering in an avoidance
  manoeuvre, where full lock is genuinely what you want.

  Heading is no longer raw GPS course-over-ground (see nav/heading.py for why
  that cannot work on a slow vehicle).
"""

from __future__ import annotations
import threading
import time
from dataclasses import dataclass

from .geo import (
    haversine_m, bearing_deg, heading_error_deg, lookahead_point,
    pure_pursuit_steer_deg,
)
from .heading import HeadingEstimator

# ---------------------------------------------------------------------------
# VEHICLE GEOMETRY — *** THESE TWO ARE PHYSICAL MEASUREMENTS. MEASURE THEM. ***
# ---------------------------------------------------------------------------
# Both are currently ESTIMATES. The steering law is only as good as these.
# Note the two errors partially cancel (WHEELBASE_M/STEER_MAX_DEG acts as one
# overall gain), so the car will still track with rough values — but measure
# them properly before you trust the tuning.
WHEELBASE_M = 0.60        # front axle -> rear axle, metres.       MEASURE ME.
STEER_MAX_DEG = 30.0      # road-wheel angle at full lock (angle=±1). MEASURE ME.

# --- Tunables ---
ARRIVE_M = 2.0                    # waypoint reached if within this many metres
# Acquiring a heading. At GO the car is stationary, so it has NO heading — GPS
# course does not exist at standstill and the Pi correctly reports
# heading_valid: false. The car must DRIVE to discover which way it points.
#
# This used to time out after 8 s and then give up, disengaging autonomous and
# stopping. That was wrong: at 0.7 m/s, 8 s is only 5.6 m, and course-made-good
# needs ~4 m of clean displacement before it yields a heading. One slow start or
# one late GPS fix and the car would shut itself off before it could ever acquire
# one — looking exactly like "autonomous does nothing".
#
# Time is the wrong thing to measure. What matters is whether the car is MOVING:
#   - moving but no heading yet  -> keep going, it is working on it
#   - moved a long way, still no heading -> the heading source is genuinely broken
#   - not moving at all          -> the car is not driving; something else is wrong
HEADING_ACQUIRE_MAX_M = 25.0      # driven this far with still no heading -> give up
HEADING_ACQUIRE_STALL_S = 12.0    # commanded forward this long without moving -> give up
HEADING_ACQUIRE_MIN_MOVE_M = 1.5  # displacement that counts as "the car is moving"
TURN_ENTER_DEG = 60.0             # error > this -> the geometry is hopeless for
                                  # pure pursuit; do a discrete pivot turn
TURN_EXIT_DEG = 25.0              # while pivoting, resume normal tracking below this

# Pivot thresholds when steering on the LAGGED course-made-good fallback. Much
# wider, so a lag-induced error spike does not trigger a pivot — only a genuine
# "I am pointing the wrong way" does. See the note at the pivot branch in tick().
TURN_ENTER_DEG_CMG = 110.0
TURN_EXIT_DEG_CMG = 45.0
LOOKAHEAD_M = 8.0                 # Pure-Pursuit lookahead (Ld).
                                  # Raised 5->8: a longer lookahead is the primary
                                  # anti-weave lever — the carrot sits further ahead,
                                  # so GPS position noise turns into a much smaller
                                  # steering angle and the car stops sawing side to
                                  # side. Too long would cut corners, but on a mostly
                                  # straight run 8m is smooth and still tracks.
                                  #
                                  # THE CENTRAL TRADE-OFF, tune this in the field:
                                  #   short Ld -> tight tracking, but GPS position
                                  #     noise (~2-3 m) gets amplified -> weaving.
                                  #   long Ld  -> smooth and noise-immune, but the
                                  #     car cuts corners and responds sluggishly.
                                  # The old 15 m was far too long for a path a few
                                  # tens of metres long — the carrot sat at or past
                                  # the goal, so the car barely steered at all.

STEER_GAIN = 1.1                  # Authority multiplier on the Pure-Pursuit output.
                                  # Lowered 1.6->1.1: 1.6 was overcorrecting, which
                                  # shows up as weaving (steer swings +/- around the
                                  # line instead of converging). With a now-steady
                                  # FUSED heading, less aggression = a straighter line.
                                  #
                                  # Pure Pursuit is geometrically exact but, on a
                                  # SHORT wheelbase (0.6 m) with a lookahead sized
                                  # for GPS noise (5 m), the geometrically-correct
                                  # steering angle is small: a 45-degree heading
                                  # error asks for only ~10 degrees of wheel, i.e.
                                  # a third of available lock. Correct, but slow to
                                  # converge. This gain buys back responsiveness
                                  # without shortening Ld into the GPS noise floor.
                                  #
                                  # TUNING: raise it if the car converges onto the
                                  # line too lazily; LOWER it if the car weaves or
                                  # oscillates around the line. 1.0 = pure geometry.

CMG_GAIN_SCALE = 0.45             # Steering authority when the heading comes from our
                                  # own course-made-good (source='cmg') rather than the
                                  # receiver's COG field.
                                  #
                                  # CMG is the bearing over the last few metres
                                  # TRAVELLED, so in a turn it reports where we WERE,
                                  # not where we point — it lags badly. Lag in a
                                  # feedback loop oscillates: simulated at full gain the
                                  # car sat at full steering lock 64% of ticks and took
                                  # 339 s over a route it does in 146 s on a healthy
                                  # COG. Damping trades convergence speed for stability,
                                  # which is the right trade for a degraded sensor.
                                  #
                                  # CMG is a SAFETY NET, not a good heading source. The
                                  # real fix is Pi-side: stop emitting 0.0 for an empty
                                  # NMEA course field; send heading_valid=false instead.
STEER_SMOOTH = 0.5                # Low-pass on the steering command, 0..1. 0 = raw
                                  # (jittery), higher = smoother but laggier. 0.5
                                  # blends half the previous command with the new one.
CRUISE_SPEED = 70                 # % duty while tracking
ACQUIRE_SPEED = 60                # % duty while acquiring heading (must be enough
                                  # to get above heading.COG_TRUST_MPS!)
TURN_SPEED = 55                   # % duty during a discrete pivot turn


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
    # NEW: the continuous steering command and the heading we steered on.
    steer_norm: float = 0.0            # -1..+1, + = RIGHT (matches the Pi's `angle`)
    speed: int = 0                     # % duty this tick
    heading_deg: float | None = None
    heading_source: str = "none"
    gyro_sign: str = "-"


class NavController:
    """GPS waypoint follower. Thread-safe via an internal lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self._waypoints: list[Waypoint] = []
        self._idx = 0
        self._state = "IDLE"
        self._acquire_started: float | None = None
        self._acquire_origin: tuple[float, float] | None = None
        self._last_dist: float | None = None
        self._last_bearing: float | None = None
        self._last_err: float | None = None
        self._last_wanted = "STOP"
        self._reason = ""
        self._last_cross_track: float | None = None
        self._last_lookahead: tuple[float, float] | None = None
        self._steer_norm = 0.0
        self._speed = 0
        self._min_seg = 0          # monotonic path progress; never regresses
        self._heading = HeadingEstimator()

    # ---------- public api ----------

    def set_waypoints(self, points: list[tuple[float, float]]) -> int:
        with self._lock:
            self._waypoints = [Waypoint(*p) for p in points if p is not None]
            self._idx = 0
            self._state = "ACQUIRING" if self._waypoints else "IDLE"
            self._acquire_started = None
            self._reason = "new waypoints"
            self._reset_run_state()
            return len(self._waypoints)

    def start_run(self) -> None:
        """Call on GO. Clears all per-run state so a new run never inherits the
        last one's heading estimate, steering, or timers."""
        with self._lock:
            if self._waypoints:
                self._state = "ACQUIRING"
                self._acquire_started = None
            self._reset_run_state()

    def _reset_run_state(self) -> None:
        self._acquire_origin = None
        self._heading.reset()
        self._steer_norm = 0.0
        self._speed = 0
        self._min_seg = 0
        self._last_wanted = "STOP"

    def stop(self) -> None:
        with self._lock:
            self._waypoints = []
            self._idx = 0
            self._state = "IDLE"
            self._acquire_started = None
            self._reason = "stopped"
            self._reset_run_state()

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
                steer_norm=self._steer_norm,
                speed=self._speed,
                heading_deg=self._heading.heading,
                heading_source=self._heading.source,
                gyro_sign=self._heading.sign_status,
            )

    # ---------- core tick ----------

    def tick(self, lat: float | None, lon: float | None,
             gps_heading_deg: float | None, gps_speed_mps: float | None,
             gyro_z_dps: float | None = None, gyro_valid: bool = False,
             fused_heading_deg: float | None = None,
             reversing: bool = False) -> str:
        """Advance the controller. Returns the coarse action; read `snapshot()`
        for the continuous `steer_norm` and `speed` that actually drive the car."""
        with self._lock:
            if not self._waypoints or self._state in ("IDLE", "ARRIVED"):
                return self._halt("idle")

            if lat is None or lon is None:
                return self._halt("no gps fix")

            cur = (lat, lon)
            final_pt = (self._waypoints[-1].lat, self._waypoints[-1].lon)

            # Arrival is always against the FINAL waypoint.
            dist_to_final = haversine_m(cur, final_pt)
            self._last_dist = dist_to_final
            if dist_to_final <= ARRIVE_M:
                self._state = "ARRIVED"
                return self._halt("arrived")

            # ---- heading: best available source (see nav/heading.py) ----
            # `reversing` is critical: COG is the direction of TRAVEL, so while
            # reversing it points 180 degrees away from the nose.
            self._heading.update(
                cog_deg=gps_heading_deg,
                speed_mps=gps_speed_mps,
                lat=lat, lon=lon,
                gyro_z_dps=gyro_z_dps,
                gyro_valid=gyro_valid,
                fused_deg=fused_heading_deg,
                reversing=reversing,
            )

            # ---- choose the steering target (carrot) ----
            #
            # There used to be an "if we're more than LOOKAHEAD_M from waypoint A,
            # drive to A first" branch. It was removed: if A was BEHIND the car (very
            # easy — you tap it on a map, and GPS puts the car metres from where you
            # tapped), the car would dutifully turn around and drive AWAY from B to go
            # and touch A first. Pure Pursuit does not need it: lookahead_point
            # projects the car onto the path and aims at a carrot ahead of that
            # projection, which pulls an off-path car back onto the line smoothly.
            path = [(w.lat, w.lon) for w in self._waypoints]
            if len(path) >= 2:
                target, cross, seg = lookahead_point(
                    path, cur, LOOKAHEAD_M, min_seg=self._min_seg)
                # Progress along the path may never go BACKWARD. Without this, GPS
                # noise (or a shove from the obstacle layer) can snap the closest
                # point back to an earlier segment and the car re-drives the route.
                self._min_seg = max(self._min_seg, seg)
                self._last_cross_track = cross
                label = f"carrot[seg{seg}]"
            else:
                target = final_pt
                self._last_cross_track = None
                label = "B"
            self._last_lookahead = target

            brg = bearing_deg(cur, target)
            self._last_bearing = brg

            # ---- no usable heading: acquire it, but NEVER drive blind forever ----
            if not self._heading.valid:
                now_t = time.time()
                if self._acquire_started is None:
                    self._acquire_started = now_t
                    self._acquire_origin = cur
                elapsed = now_t - self._acquire_started
                moved = haversine_m(self._acquire_origin, cur) if self._acquire_origin else 0.0

                # Driven a long way and STILL no heading -> the heading source is
                # genuinely broken. Don't keep driving blind.
                if moved >= HEADING_ACQUIRE_MAX_M:
                    self._state = "NO_HEADING"
                    return self._halt(
                        f"drove {moved:.0f}m and still cannot determine which way the "
                        f"car is pointing — STOPPED. GPS course and course-made-good "
                        f"both failed. Check gps.heading_valid and that lat/lon are "
                        f"actually changing.")

                # Commanded FORWARD for a while but the car has NOT MOVED -> the car
                # is not driving. That is not a heading problem; report it as itself.
                if elapsed >= HEADING_ACQUIRE_STALL_S and moved < HEADING_ACQUIRE_MIN_MOVE_M:
                    self._state = "NOT_MOVING"
                    return self._halt(
                        f"commanded FORWARD for {elapsed:.0f}s but the car has only "
                        f"moved {moved:.1f}m — IT IS NOT DRIVING. The laptop is sending "
                        f"the command; check the Pi (safety_violation?), the motors, "
                        f"and the battery.")

                # Otherwise: keep driving straight. The car is building up the
                # displacement it needs to work out its own heading. This is normal
                # and expected for the first few metres of every run.
                self._state = "ACQUIRING"
                self._steer_norm = 0.0
                self._speed = ACQUIRE_SPEED
                self._last_err = None
                self._last_wanted = "FORWARD"
                self._reason = (f"acquiring heading: driven {moved:.1f}m in {elapsed:.0f}s "
                                f"(need ~4m of travel; src={self._heading.source})")
                return "FORWARD"

            self._acquire_started = None
            hdg = self._heading.heading

            # ---- heading error to the carrot ----
            err = heading_error_deg(hdg, brg)
            self._last_err = err

            # ---- discrete pivot only when the geometry is hopeless ----
            # Pure Pursuit assumes the target is roughly ahead. Beyond ~60 deg the
            # law saturates and a pivot turn is genuinely the right move. Hysteresis
            # stops it flapping in and out.
            # The pivot branch is NOT optional. Pure Pursuit has a singularity at
            # 180 degrees: sin(180) = 0, so the steering command is ZERO and a car
            # pointing exactly away from its target would drive straight forever,
            # unable to turn around. The pivot is what escapes that.
            #
            # But on a LAGGED heading (course-made-good reports where we WERE, not
            # where we point) the error spikes spuriously, and pivoting tightens the
            # turn, which worsens the lag — a vicious cycle. Simulated at the normal
            # threshold, the car sat at full steering lock 62% of the time.
            #
            # So: keep the pivot, but make it much harder to trigger on a lagged
            # heading. Only a genuinely large, sustained error (the real
            # "I am pointing the wrong way" case) gets one.
            on_lagged_heading = (self._heading.source == "cmg")
            enter = TURN_ENTER_DEG_CMG if on_lagged_heading else TURN_ENTER_DEG
            exit_ = TURN_EXIT_DEG_CMG if on_lagged_heading else TURN_EXIT_DEG

            pivoting = self._last_wanted in ("TURN_LEFT", "TURN_RIGHT")
            band = exit_ if pivoting else enter

            if abs(err) >= band:
                act = "TURN_RIGHT" if err > 0 else "TURN_LEFT"
                self._steer_norm = 1.0 if err > 0 else -1.0
                self._speed = TURN_SPEED
            else:
                # ---- THE FIX: continuous Pure-Pursuit steering ----
                ld = max(haversine_m(cur, target), 1.0)
                delta = pure_pursuit_steer_deg(err, ld, WHEELBASE_M)
                # Damp the gain when steering on the LAGGED course-made-good
                # fallback — lag in a feedback loop oscillates.
                gain = STEER_GAIN * (CMG_GAIN_SCALE
                                     if self._heading.source == "cmg" else 1.0)
                raw_steer = max(-1.0, min(1.0, gain * delta / STEER_MAX_DEG))
                # Low-pass the steering command. Every source (GPS, the actuator,
                # the road) adds a little tick-to-tick jitter; feeding it straight to
                # the wheels makes the car saw side to side. Blending with the last
                # command smooths that out. This is the third anti-weave lever
                # alongside a longer lookahead and a lower gain.
                self._steer_norm = (STEER_SMOOTH * self._steer_norm
                                    + (1.0 - STEER_SMOOTH) * raw_steer)
                self._speed = CRUISE_SPEED
                act = "FORWARD"

            self._state = "NAVIGATING"
            self._last_wanted = act
            ct = (f" xtrack={self._last_cross_track:+.1f}m"
                  if self._last_cross_track is not None else "")
            self._reason = (f"target={label} err={err:+.1f}° steer={self._steer_norm:+.2f} "
                            f"dist={dist_to_final:.1f}m hdg={hdg:.0f}°/{self._heading.source}{ct}")
            return act

    def _halt(self, reason: str) -> str:
        self._steer_norm = 0.0
        self._speed = 0
        self._last_wanted = "STOP"
        self._reason = reason
        return "STOP"
