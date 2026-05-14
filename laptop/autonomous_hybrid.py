"""Rule-based obstacle override.

Goal: when the GPS-nav (or model) wants to drive in some direction, this
function checks the live sensors + YOLO and may downgrade the action to a
safer one. It actively tries to AVOID rather than just STOP — only stops as
a last resort when no side or rear escape is available.

Inputs
------
sensors: dict with FL, FR, FW, BC, LS, RS in centimetres.
         Values <= 0 or > MAX are treated as 'no reading' (clear).
yolo:    dict with person_detected, object_detected,
         nearest_area_ratio (0..1), nearest_position (-1=L, 0=C, +1=R),
         num_objects.
wanted_action: name string from ml.actions (FORWARD / TURN_* / etc).

Returns
-------
(final_action_name, reason)
"""

from __future__ import annotations

# ---- distance thresholds (cm) -------------------------------------------
FRONT_EMERGENCY_CM = 30      # too close in front -> reverse / stop
FRONT_SLOW_CM = 70           # close-ish in front -> slow / turn aside
SIDE_PINCH_CM = 25           # below this on a side -> avoid that side
SIDE_OPEN_CM = 60            # above this on a side -> good for a turn
BACK_SAFE_CM = 40            # below this on rear -> can't reverse

# ---- YOLO thresholds ----------------------------------------------------
# Raised so a person standing a few metres away while you watch the car
# doesn't constantly trigger avoidance. Sensors still catch close hits.
PERSON_BLOCKING_AREA = 0.35  # >35% of frame -> person right in front
PERSON_NEAR_AREA = 0.15      # 15..35% -> close enough to slow down

VALID_MIN, VALID_MAX = 2.0, 401.0


def _v(d: dict, k: str) -> float | None:
    """Get a sensor value, return None if missing/invalid."""
    if not d:
        return None
    x = d.get(k)
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if x < VALID_MIN or x > VALID_MAX:
        return None
    return x


def decide(sensors: dict, yolo: dict | None = None,
           wanted_action: str = "FORWARD") -> tuple[str, str]:
    """Return (final_action, reason)."""
    yolo = yolo or {}
    fl, fr, fw = _v(sensors, "FL"), _v(sensors, "FR"), _v(sensors, "FW")
    bc, ls, rs = _v(sensors, "BC"), _v(sensors, "LS"), _v(sensors, "RS")

    # ---- 0. Veto any caller-requested REVERSE if rear isn't clear ----
    # Without this, the ML model (or any upstream) saying "REVERSE" gets
    # passed through and the car backs into whatever is behind it.
    if wanted_action in ("REVERSE", "REVERSE_LEFT", "REVERSE_RIGHT"):
        if bc is None or bc < BACK_SAFE_CM:
            return "STOP", "REAR_BLOCKED"
        # rear clear -> safe to honour the requested reverse
        return wanted_action, ""

    person = int(yolo.get("person_detected", 0) or 0)
    area = float(yolo.get("nearest_area_ratio", 0.0) or 0.0)
    person_pos = int(yolo.get("nearest_position", 0) or 0)   # -1=L 0=C 1=R

    # Reduce front_min by a margin when YOLO sees a close person — treats the
    # camera evidence as an additional "ghost" front reading.
    front_vals = [v for v in (fl, fr, fw) if v is not None]
    front_min = min(front_vals) if front_vals else 999.0

    person_blocks = person and area >= PERSON_BLOCKING_AREA
    if person_blocks:
        # Force avoidance to consider the person as 'object very close in front'.
        front_min = min(front_min, FRONT_EMERGENCY_CM - 1)

    # ---- 1. Hard emergency: front too close ----
    if front_min < FRONT_EMERGENCY_CM:
        # Prefer reverse if back is clear.
        if bc is None or bc > BACK_SAFE_CM:
            left_room = (ls or 999.0)
            right_room = (rs or 999.0)
            # If YOLO sees a person, bias AWAY from them.
            if person_blocks:
                if person_pos > 0:        # person on right
                    return "REVERSE_LEFT", "PERSON_AVOID_REV_LEFT"
                if person_pos < 0:        # person on left
                    return "REVERSE_RIGHT", "PERSON_AVOID_REV_RIGHT"
            if left_room > right_room and left_room > SIDE_PINCH_CM:
                return "REVERSE_LEFT", "OBST_REVERSE_LEFT"
            if right_room > SIDE_PINCH_CM:
                return "REVERSE_RIGHT", "OBST_REVERSE_RIGHT"
            return "REVERSE", "OBST_REVERSE"
        # Back blocked too -> dead stop.
        return "STOP", "OBST_PINNED"

    # ---- 2. Close-ish obstacle in front: turn around it if possible ----
    if front_min < FRONT_SLOW_CM:
        side_l = ls if ls is not None else 999.0
        side_r = rs if rs is not None else 999.0
        # YOLO bias — person on right -> prefer turning left, and vice versa.
        if person_blocks or (person and area >= PERSON_NEAR_AREA):
            if person_pos > 0 and side_l > SIDE_PINCH_CM:
                return "TURN_LEFT", "PERSON_AVOID_LEFT"
            if person_pos < 0 and side_r > SIDE_PINCH_CM:
                return "TURN_RIGHT", "PERSON_AVOID_RIGHT"
        if side_l > side_r and side_l > SIDE_PINCH_CM:
            return "TURN_LEFT", "OBST_NEAR_TURN_LEFT"
        if side_r > SIDE_PINCH_CM:
            return "TURN_RIGHT", "OBST_NEAR_TURN_RIGHT"
        return "SLOW_DOWN", "OBST_NEAR_SLOW"

    # ---- 3. Person ahead at medium range — slow but keep driving ----
    if person and area >= PERSON_NEAR_AREA:
        if wanted_action == "FORWARD":
            return "SLOW_DOWN", "PERSON_NEAR"

    # ---- 4. Nothing dangerous: pass through ----
    return wanted_action, ""
