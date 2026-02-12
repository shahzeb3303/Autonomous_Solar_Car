"""
Differential Steering Controller for Tank-Style Movement.

This module provides high-level steering control for a vehicle with
differential (tank-style) steering. The vehicle has 4 DC motors:
- 2 left-side motors (controlled together)
- 2 right-side motors (controlled together)

Turning is achieved by running the left and right motors at different
speeds. This is different from Ackermann steering (like a car) which
uses a steering servo.

Movement Types:
- Forward/Reverse: Both sides same speed, same direction
- Turn: Outside wheels faster than inside wheels
- Pivot: One side forward, other side reverse (spin in place)
- Arc: Curved path with different speeds on each side

Usage:
    from src.control.steering import DifferentialSteering
    from src.control.motor_controller import MotorController

    motor = MotorController()
    motor.initialize()

    steering = DifferentialSteering(motor)
    steering.move_forward(speed=50)
    steering.turn(angle=-30)  # Turn left 30 degrees
    steering.stop()
"""

import time
import math
import threading
from typing import Optional, Tuple
from enum import Enum
import logging

from src.control.motor_controller import MotorController, MotorDirection
from config.settings import get_settings

logger = logging.getLogger(__name__)


class SteeringMode(Enum):
    """Steering behavior modes."""
    STOPPED = "stopped"
    FORWARD = "forward"
    REVERSE = "reverse"
    TURNING_LEFT = "turning_left"
    TURNING_RIGHT = "turning_right"
    PIVOTING_LEFT = "pivoting_left"
    PIVOTING_RIGHT = "pivoting_right"
    REVERSING_LEFT = "reversing_left"
    REVERSING_RIGHT = "reversing_right"


class DifferentialSteering:
    """
    High-level differential steering controller.

    Provides intuitive steering commands that are translated to
    differential motor speeds. Supports both proportional turns
    (for path following) and timed turns.

    Args:
        motor_controller: Initialized MotorController instance
        wheel_base_cm: Distance between left and right wheels (for arc calculation)
    """

    # Default wheel base - adjust based on actual vehicle dimensions
    DEFAULT_WHEEL_BASE_CM = 30.0

    def __init__(
        self,
        motor_controller: MotorController,
        wheel_base_cm: float = DEFAULT_WHEEL_BASE_CM,
    ):
        """Initialize the steering controller."""
        self._motor = motor_controller
        self._settings = get_settings()
        self._wheel_base = wheel_base_cm

        self._current_mode = SteeringMode.STOPPED
        self._current_speed = 0
        self._lock = threading.Lock()

        logger.info(f"DifferentialSteering initialized, wheel_base={wheel_base_cm}cm")

    def move_forward(self, speed: Optional[int] = None) -> None:
        """
        Move forward at the specified speed.

        Args:
            speed: Speed percentage (0-100). Defaults to SPEED_NORMAL.
        """
        with self._lock:
            if speed is None:
                speed = self._settings.SPEED_NORMAL

            self._motor.forward(speed)
            self._current_mode = SteeringMode.FORWARD
            self._current_speed = speed

    def move_reverse(self, speed: Optional[int] = None) -> None:
        """
        Move backward at the specified speed.

        Args:
            speed: Speed percentage (0-100). Defaults to SPEED_SLOW.
        """
        with self._lock:
            if speed is None:
                speed = self._settings.SPEED_SLOW

            self._motor.reverse(speed)
            self._current_mode = SteeringMode.REVERSE
            self._current_speed = speed

    def turn(
        self,
        angle: float,
        speed: Optional[int] = None,
        gradual: bool = True,
    ) -> None:
        """
        Turn by adjusting wheel speeds (for path following).

        This method adjusts the differential between left and right
        wheels based on the desired turn angle. Positive angle turns
        right, negative angle turns left.

        Args:
            angle: Turn angle in degrees (-90 to +90)
                   Negative = left, Positive = right
            speed: Base forward speed (0-100)
            gradual: If True, maintain forward motion while turning
                    If False, use sharper pivot-style turn
        """
        with self._lock:
            if speed is None:
                speed = self._settings.SPEED_NORMAL

            # Clamp angle to valid range
            angle = max(-90, min(90, angle))

            # Calculate differential based on angle
            # At 0 degrees: both wheels same speed
            # At ±90 degrees: one wheel at minimum speed
            turn_ratio = abs(angle) / 90.0

            if gradual:
                # Gradual turn: reduce inside wheel speed
                inside_speed = int(speed * (1 - turn_ratio * 0.7))
                outside_speed = speed
            else:
                # Sharp turn: more aggressive differential
                inside_speed = int(speed * (1 - turn_ratio))
                outside_speed = speed

            inside_speed = max(0, inside_speed)

            if angle < 0:
                # Turn left: left wheel slower
                left_speed = inside_speed
                right_speed = outside_speed
                self._current_mode = SteeringMode.TURNING_LEFT
            else:
                # Turn right: right wheel slower
                left_speed = outside_speed
                right_speed = inside_speed
                self._current_mode = SteeringMode.TURNING_RIGHT

            self._motor.set_motors(
                left_speed=left_speed,
                right_speed=right_speed,
                left_direction=MotorDirection.FORWARD,
                right_direction=MotorDirection.FORWARD,
            )
            self._current_speed = speed

            logger.debug(f"Turn {angle}°: L={left_speed}%, R={right_speed}%")

    def turn_left(self, speed: Optional[int] = None) -> None:
        """
        Execute a left turn.

        Args:
            speed: Outer wheel speed (0-100)
        """
        with self._lock:
            self._motor.turn_left(speed)
            self._current_mode = SteeringMode.TURNING_LEFT
            self._current_speed = speed or self._settings.TURN_SPEED_OUTER

    def turn_right(self, speed: Optional[int] = None) -> None:
        """
        Execute a right turn.

        Args:
            speed: Outer wheel speed (0-100)
        """
        with self._lock:
            self._motor.turn_right(speed)
            self._current_mode = SteeringMode.TURNING_RIGHT
            self._current_speed = speed or self._settings.TURN_SPEED_OUTER

    def pivot_left(self, speed: Optional[int] = None) -> None:
        """
        Pivot (spin) left in place.

        Args:
            speed: Motor speed (0-100)
        """
        with self._lock:
            self._motor.pivot_left(speed)
            self._current_mode = SteeringMode.PIVOTING_LEFT
            self._current_speed = speed or self._settings.TURN_SPEED_INNER

    def pivot_right(self, speed: Optional[int] = None) -> None:
        """
        Pivot (spin) right in place.

        Args:
            speed: Motor speed (0-100)
        """
        with self._lock:
            self._motor.pivot_right(speed)
            self._current_mode = SteeringMode.PIVOTING_RIGHT
            self._current_speed = speed or self._settings.TURN_SPEED_INNER

    def reverse_turn_left(self, speed: Optional[int] = None) -> None:
        """
        Reverse while turning left.

        Args:
            speed: Base speed (0-100)
        """
        with self._lock:
            self._motor.reverse_left(speed)
            self._current_mode = SteeringMode.REVERSING_LEFT
            self._current_speed = speed or self._settings.SPEED_SLOW

    def reverse_turn_right(self, speed: Optional[int] = None) -> None:
        """
        Reverse while turning right.

        Args:
            speed: Base speed (0-100)
        """
        with self._lock:
            self._motor.reverse_right(speed)
            self._current_mode = SteeringMode.REVERSING_RIGHT
            self._current_speed = speed or self._settings.SPEED_SLOW

    def turn_for_duration(
        self,
        direction: str,
        duration_s: float,
        speed: Optional[int] = None,
    ) -> None:
        """
        Execute a timed turn.

        Turns for the specified duration then stops.

        Args:
            direction: "left" or "right"
            duration_s: Turn duration in seconds
            speed: Turn speed (0-100)
        """
        if direction.lower() == "left":
            self.turn_left(speed)
        else:
            self.turn_right(speed)

        time.sleep(duration_s)
        self.stop()

    def pivot_for_duration(
        self,
        direction: str,
        duration_s: float,
        speed: Optional[int] = None,
    ) -> None:
        """
        Execute a timed pivot (spin in place).

        Args:
            direction: "left" or "right"
            duration_s: Pivot duration in seconds
            speed: Pivot speed (0-100)
        """
        if direction.lower() == "left":
            self.pivot_left(speed)
        else:
            self.pivot_right(speed)

        time.sleep(duration_s)
        self.stop()

    def arc_turn(
        self,
        radius_cm: float,
        angle_deg: float,
        speed: int,
    ) -> Tuple[int, int]:
        """
        Calculate wheel speeds for an arc turn.

        Given the desired turning radius and base speed, calculates
        the speeds needed for each wheel.

        Args:
            radius_cm: Turn radius in centimeters (positive = right, negative = left)
            angle_deg: Arc angle in degrees (not used for speed calc, for reference)
            speed: Outer wheel speed

        Returns:
            Tuple of (left_speed, right_speed)
        """
        if abs(radius_cm) < 1:
            # Very tight turn, essentially a pivot
            if radius_cm >= 0:
                return (speed, 0)
            else:
                return (0, speed)

        # Calculate inner wheel speed based on radius
        # v_inner / v_outer = (R - W/2) / (R + W/2)
        # where R is radius, W is wheel base
        half_base = self._wheel_base / 2
        abs_radius = abs(radius_cm)

        ratio = (abs_radius - half_base) / (abs_radius + half_base)
        inner_speed = int(speed * max(0, ratio))

        if radius_cm > 0:
            # Turning right: right wheel is inner
            return (speed, inner_speed)
        else:
            # Turning left: left wheel is inner
            return (inner_speed, speed)

    def stop(self) -> None:
        """Stop all movement (coast)."""
        with self._lock:
            self._motor.stop()
            self._current_mode = SteeringMode.STOPPED
            self._current_speed = 0

    def brake(self) -> None:
        """Stop all movement with braking."""
        with self._lock:
            self._motor.brake()
            self._current_mode = SteeringMode.STOPPED
            self._current_speed = 0

    def emergency_stop(self) -> None:
        """Immediate emergency stop."""
        self._motor.emergency_stop()
        self._current_mode = SteeringMode.STOPPED
        self._current_speed = 0

    def slow_down(self, factor: Optional[float] = None) -> None:
        """
        Reduce current speed by a factor.

        Args:
            factor: Speed reduction factor (0-1). Defaults to SLOW_DOWN_FACTOR.
        """
        with self._lock:
            if factor is None:
                factor = self._settings.SLOW_DOWN_FACTOR

            new_speed = int(self._current_speed * (1 - factor))
            new_speed = max(self._settings.SPEED_SLOW, new_speed)

            self._motor.set_speed(new_speed)
            self._current_speed = new_speed
            logger.info(f"Slowed down to {new_speed}%")

    def adjust_heading(self, error_deg: float, gain: float = 0.5) -> None:
        """
        Adjust heading to correct for error (for path following).

        Uses proportional control to make small corrections based
        on heading error.

        Args:
            error_deg: Heading error in degrees (positive = need to turn right)
            gain: Proportional gain (how aggressively to correct)
        """
        # Convert error to turn angle
        turn_angle = error_deg * gain
        turn_angle = max(-45, min(45, turn_angle))  # Limit correction

        if abs(turn_angle) > 5:  # Only correct if error is significant
            self.turn(turn_angle, speed=self._current_speed, gradual=True)

    @property
    def current_mode(self) -> SteeringMode:
        """Get current steering mode."""
        return self._current_mode

    @property
    def current_speed(self) -> int:
        """Get current speed setting."""
        return self._current_speed

    @property
    def is_moving(self) -> bool:
        """Check if vehicle is currently moving."""
        return self._current_mode != SteeringMode.STOPPED

    @property
    def is_reversing(self) -> bool:
        """Check if vehicle is moving in reverse."""
        return self._current_mode in (
            SteeringMode.REVERSE,
            SteeringMode.REVERSING_LEFT,
            SteeringMode.REVERSING_RIGHT,
        )
