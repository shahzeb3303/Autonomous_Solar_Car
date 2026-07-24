"""Rule-based obstacle override.

The GPS controller says where it WANTS to go. This layer may downgrade that to
something safer. It is the *strategy* layer — the Pi's hard stop at 50 cm is the
reflex underneath it, and should essentially never fire if this works.

WHY THIS WAS REWRITTEN (2026-07-12, after the first real outdoor run)
--------------------------------------------------------------------
The car never got a metre closer to its destination. The log showed almost every
tick was an override to REVERSE_LEFT / REVERSE_RIGHT, with reasons like
PERSON_AVOID_REV_LEFT and OBST_REVERSE_RIGHT. The car spent the whole run
reversing away from the operator standing next to it. Three design errors:

1. **The camera was given distance authority.** A YOLO "person" whose bounding box
   filled >35% of the frame was converted into a fake ultrasonic reading INSIDE the
   emergency threshold, which forced a REVERSE. But box area is a terrible proxy for
   distance — a person 3 m away fills a huge fraction of the frame. So the operator
   watching the car made it flee. **Cameras classify; ultrasonics measure. The camera
   may now only slow the car and bias which way it steers around something the
   ULTRASONICS have already detected. It can never trigger a reverse or a stop.**

2. **The thresholds were far too wide** (emergency 100 cm, slow 150 cm). In any real
   test area something is within a metre most of the time, so the car was permanently
   "in an emergency". Backed down to values that leave real margin above the Pi's
   50 cm reflex without firing constantly.

3. **Reversing was treated as routine avoidance.** It must not be. The car's ONLY
   heading source is GPS course-over-ground, which measures the direction of TRAVEL —
   so the moment the car reverses, its heading estimate inverts by 180 degrees and
   navigation is poisoned. Reverse is now an ESCAPE manoeuvre of last resort, used
   only when the car is genuinely stuck (front blocked, no side open, already stopped
   for several seconds). Normal avoidance is: slow down, steer around, or stop.

Inputs
------
sensors: dict with FL, FR, FW, BC, LS, RS in centimetres.
         -1 (or any negative) = no echo = nothing in range = CLEAR (400 cm).
health:  dict {'FL': True, 'LS': False, ...} from the Pi. False = do not trust.
yolo:    dict with person_detected, nearest_area_ratio, nearest_position (-1=L,0=C,+1=R)
stuck_s: how long the car has been stopped-and-blocked. Only above STUCK_ESCAPE_S
         may this layer command a REVERSE.

Returns (final_action_name, reason)
"""

from __future__ import annotations

# ---- distance thresholds (cm) -------------------------------------------
#
# LAYERING. The Pi's reflex must be the INNERMOST layer — it fires only when this
# layer has already failed:
#
#     us: slow down / steer around   < 120 cm
#     us: stop                       <  70 cm
#     Pi reflex hard-stop            <  50 cm   (last resort, do not lower)
#
# The car does 0.5-0.8 m/s with ~550 ms of reaction latency (~45 cm of blind
# travel at the top of that range), so 70 cm is tight but real. If you raise the
# speed, raise all of these — they scale linearly with it.
FRONT_SLOW_CM = 120          # obstacle ahead -> slow, and steer around it if possible
FRONT_STOP_CM = 70           # too close -> stop. NOT reverse. Stop.
SIDE_PINCH_CM = 40           # a side with less room than this is NOT an escape route
BACK_SAFE_CM = 60            # below this behind us, reversing is not safe

# ---- camera (YOLO) ------------------------------------------------------
# The camera may ONLY slow the car and bias the direction of a turn. It has no
# distance authority and can never trigger a stop or a reverse — see the header.
PERSON_SLOW_AREA = 0.20      # person occupying >20% of frame -> ease off the throttle

VALID_MIN, VALID_MAX = 2.0, 401.0
MAX_RANGE_CM = 400.0         # what "no echo" means: nothing within range

# How long the car must be stopped-and-blocked before it is allowed to reverse out.
STUCK_ESCAPE_S = 3.0


def _v(d: dict, k: str, health: dict | None = None) -> float | None:
    """Resolve one sensor to a distance, or None = UNKNOWN (do not trust).

        health False        -> None       (never trusted, never an escape route)
        value  negative     -> 400.0      (no echo = nothing in range = CLEAR)
        value  in range     -> the value
        anything else       -> None       (garbage)

    A dead sensor and a clear path both produce "no echo", so they cannot be told
    apart from a single reading — that is what `health` (computed over time on the
    Pi) is for.
    """
    if not d:
        return None
    if health is not None and health.get(k) is False:
        return None
    x = d.get(k)
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if x < 0:
        return MAX_RANGE_CM
    if x < VALID_MIN or x > VALID_MAX:
        return None
    return x


def decide(sensors: dict, yolo: dict | None = None,
           wanted_action: str = "FORWARD",
           health: dict | None = None,
           stuck_s: float = 0.0) -> tuple[str, str]:
    """Return (final_action, reason)."""
    yolo = yolo or {}
    fl, fr, fw = (_v(sensors, "FL", health), _v(sensors, "FR", health),
                  _v(sensors, "FW", health))
    bc, ls, rs = (_v(sensors, "BC", health), _v(sensors, "LS", health),
                  _v(sensors, "RS", health))

    # Front distance comes from the ULTRASONICS ONLY. The camera does not get a vote
    # here — that was the bug that made the car reverse away from its own operator.
    front_vals = [v for v in (fl, fr, fw) if v is not None]
    front_min = min(front_vals) if front_vals else MAX_RANGE_CM

    # An unknown side is NOT an escape route (0.0 = "no room"), so a dead sensor can
    # never win the "which way has more space" comparison.
    room_l = ls if ls is not None else 0.0
    room_r = rs if rs is not None else 0.0

    person = int(yolo.get("person_detected", 0) or 0)
    area = float(yolo.get("nearest_area_ratio", 0.0) or 0.0)
    person_pos = int(yolo.get("nearest_position", 0) or 0)   # -1=L 0=C +1=R
    person_near = bool(person and area >= PERSON_SLOW_AREA)

    # ---- 1. STUCK: the only path to a REVERSE ----------------------------
    # Reversing inverts GPS course-over-ground, which is our only heading source,
    # so it wrecks navigation. It is an escape of last resort, never routine.
    if front_min < FRONT_STOP_CM and stuck_s >= STUCK_ESCAPE_S:
        if bc is not None and bc <= BACK_SAFE_CM:
            return "STOP", "STUCK_REAR_BLOCKED"
        if bc is None:
            return "STOP", "STUCK_REAR_UNKNOWN"
        # Back out toward whichever side has real, measured room.
        if room_l > room_r and room_l > SIDE_PINCH_CM:
            return "REVERSE_LEFT", "ESCAPE_REVERSE_LEFT"
        if room_r > SIDE_PINCH_CM:
            return "REVERSE_RIGHT", "ESCAPE_REVERSE_RIGHT"
        return "REVERSE", "ESCAPE_REVERSE"

    # ---- 2. Too close in front: STOP. Do not reverse, do not swerve. -----
    if front_min < FRONT_STOP_CM:
        return "STOP", f"OBST_STOP_{front_min:.0f}cm"

    # ---- 3. Obstacle ahead: steer around it, or slow down ----------------
    if front_min < FRONT_SLOW_CM:
        # Bias away from a person the camera sees — but only to choose WHICH WAY to
        # go around something the ultrasonics have already found. Never to reverse.
        if person_near and person_pos > 0 and room_l > SIDE_PINCH_CM:
            return "TURN_LEFT", "AVOID_LEFT_PERSON_RIGHT"
        if person_near and person_pos < 0 and room_r > SIDE_PINCH_CM:
            return "TURN_RIGHT", "AVOID_RIGHT_PERSON_LEFT"
        if room_l > room_r and room_l > SIDE_PINCH_CM:
            return "TURN_LEFT", f"OBST_AROUND_LEFT_{front_min:.0f}cm"
        if room_r > SIDE_PINCH_CM:
            return "TURN_RIGHT", f"OBST_AROUND_RIGHT_{front_min:.0f}cm"
        # Nowhere to go around it — just slow and keep closing; rule 2 will stop us.
        return "SLOW_DOWN", f"OBST_NO_GAP_{front_min:.0f}cm"

    # ---- 4. Camera-only caution: ease off, but KEEP NAVIGATING -----------
    # This is the strongest thing the camera is allowed to do on its own.
    if person_near and wanted_action == "FORWARD":
        return "SLOW_DOWN", "PERSON_IN_VIEW_SLOW"

    # ---- 5. Clear. The GPS controller keeps the wheel. -------------------
    return wanted_action, ""
