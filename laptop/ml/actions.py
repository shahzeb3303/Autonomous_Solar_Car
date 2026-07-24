#!/usr/bin/env python3
"""Action space for the decision model (single source of truth)."""

FORWARD = 0
SLOW_DOWN = 1
TURN_LEFT = 2
TURN_RIGHT = 3
STOP = 4
REVERSE_LEFT = 5
REVERSE_RIGHT = 6
REVERSE = 7

ACTION_NAMES = [
    "FORWARD", "SLOW_DOWN", "TURN_LEFT", "TURN_RIGHT",
    "STOP", "REVERSE_LEFT", "REVERSE_RIGHT", "REVERSE",
]
NUM_ACTIONS = len(ACTION_NAMES)

# Wiring inversion is handled on the Pi side (config.MOTOR_INVERTED in
# raspberry_pi/config.py), NOT here. From this side the mapping is direct:
# semantic FORWARD -> Pi command 'FORWARD' -> Pi rewires to physical FORWARD.
# Keep this flag here for backwards-compat with code that imports it; do not
# flip it on the laptop side or you'll double-invert and lose the dataset
# labelling convention.
DRIVE_INVERTED = False


def action_to_pi_command(action_id: int) -> dict:
    """Map action id -> Pi TCP command dict (physical direction)."""
    fwd_cmd = 'BACKWARD' if DRIVE_INVERTED else 'FORWARD'
    rev_cmd = 'FORWARD' if DRIVE_INVERTED else 'BACKWARD'
    m = {
        FORWARD:       {'command': fwd_cmd, 'steer': 'STEER_STOP', 'speed': 80},
        SLOW_DOWN:     {'command': fwd_cmd, 'steer': 'STEER_STOP', 'speed': 45},
        TURN_LEFT:     {'command': fwd_cmd, 'steer': 'LEFT',       'speed': 55},
        TURN_RIGHT:    {'command': fwd_cmd, 'steer': 'RIGHT',      'speed': 55},
        STOP:          {'command': 'STOP',  'steer': 'STEER_STOP', 'speed': 0},
        REVERSE_LEFT:  {'command': rev_cmd, 'steer': 'LEFT',       'speed': 45},
        REVERSE_RIGHT: {'command': rev_cmd, 'steer': 'RIGHT',      'speed': 45},
        REVERSE:       {'command': rev_cmd, 'steer': 'STEER_STOP', 'speed': 45},
    }
    return m.get(action_id, m[STOP])


SLOW_DOWN_SPEED_THRESHOLD = 35  # speed < this while moving forward → SLOW_DOWN


def manual_to_action(drive: str, steer: str, speed: int = 50) -> int:
    """Convert manual drive+steer -> action id (used when labeling training data).
    Accounts for DRIVE_INVERTED: protocol drive='BACKWARD' = physical FORWARD.

    Intent-based labeling:
        - STOP + steer=LEFT/RIGHT → TURN_LEFT/TURN_RIGHT (user wants to turn)
        - Forward at low speed → SLOW_DOWN
    """
    # Determine physical direction
    physical_forward = (drive == 'BACKWARD') if DRIVE_INVERTED else (drive == 'FORWARD')
    physical_backward = (drive == 'FORWARD') if DRIVE_INVERTED else (drive == 'BACKWARD')

    # STOP + steer captures turn intent
    if drive == 'STOP':
        if steer == 'LEFT':
            return TURN_LEFT
        if steer == 'RIGHT':
            return TURN_RIGHT
        return STOP

    if physical_forward:
        if steer == 'LEFT':
            return TURN_LEFT
        if steer == 'RIGHT':
            return TURN_RIGHT
        # Low speed forward = SLOW_DOWN
        if 0 < speed < SLOW_DOWN_SPEED_THRESHOLD:
            return SLOW_DOWN
        return FORWARD
    if physical_backward:
        if steer == 'LEFT':
            return REVERSE_LEFT
        if steer == 'RIGHT':
            return REVERSE_RIGHT
        return REVERSE
    return STOP
