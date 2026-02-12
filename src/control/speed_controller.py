"""
Speed Controller with PID and Smooth Acceleration.

This module provides advanced speed control features:
- Smooth acceleration and deceleration
- Speed limiting based on proximity to obstacles
- Simple PID control for maintaining target speed

Usage:
    from src.control.speed_controller import SpeedController
    from src.control.motor_controller import MotorController

    motor = MotorController()
    motor.initialize()

    speed_ctrl = SpeedController(motor)
    speed_ctrl.set_target_speed(50)
    speed_ctrl.update()  # Call periodically in main loop
"""

import time
import threading
from typing import Optional, Callable
from dataclasses import dataclass
import logging

from src.control.motor_controller import MotorController
from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class PIDGains:
    """PID controller gains."""
    kp: float = 1.0   # Proportional gain
    ki: float = 0.1   # Integral gain
    kd: float = 0.05  # Derivative gain


class SpeedController:
    """
    Advanced speed controller with smooth acceleration.

    Provides:
    - Gradual speed ramping (no jerky acceleration)
    - Speed limiting based on obstacle proximity
    - Optional PID control for cruise control
    - Thread-safe operation

    Args:
        motor_controller: Initialized MotorController instance
        pid_gains: Optional PID gains for cruise control
    """

    def __init__(
        self,
        motor_controller: MotorController,
        pid_gains: Optional[PIDGains] = None,
    ):
        """Initialize the speed controller."""
        self._motor = motor_controller
        self._settings = get_settings()
        self._pid = pid_gains or PIDGains()

        # Speed state
        self._target_speed = 0
        self._current_speed = 0
        self._max_allowed_speed = self._settings.SPEED_MAX

        # PID state
        self._integral = 0.0
        self._last_error = 0.0
        self._last_update_time = time.time()

        # Acceleration control
        self._acceleration_step = self._settings.ACCELERATION_STEP
        self._acceleration_delay = self._settings.ACCELERATION_DELAY_S

        self._lock = threading.Lock()
        self._enabled = True

        logger.info("SpeedController initialized")

    def set_target_speed(self, speed: int) -> None:
        """
        Set the target speed.

        The controller will gradually adjust actual speed toward
        this target.

        Args:
            speed: Target speed (0-100)
        """
        with self._lock:
            self._target_speed = max(0, min(100, speed))
            logger.debug(f"Target speed set to {self._target_speed}%")

    def set_max_speed(self, max_speed: int) -> None:
        """
        Set maximum allowed speed.

        Useful for limiting speed near obstacles.

        Args:
            max_speed: Maximum speed (0-100)
        """
        with self._lock:
            self._max_allowed_speed = max(0, min(100, max_speed))

            # If current target exceeds new max, reduce it
            if self._target_speed > self._max_allowed_speed:
                self._target_speed = self._max_allowed_speed

            logger.debug(f"Max speed limited to {self._max_allowed_speed}%")

    def limit_speed_by_distance(self, distance_cm: float) -> None:
        """
        Limit maximum speed based on distance to nearest obstacle.

        Uses a linear relationship between distance and allowed speed.

        Args:
            distance_cm: Distance to nearest obstacle in cm
        """
        settings = self._settings

        if distance_cm < settings.CRITICAL_DISTANCE_CM:
            # Very close - stop
            max_speed = 0
        elif distance_cm < settings.DANGER_DISTANCE_CM:
            # Close - slow speed only
            max_speed = settings.SPEED_SLOW
        elif distance_cm < settings.CAUTION_DISTANCE_CM:
            # Medium distance - limited speed
            max_speed = settings.SPEED_NORMAL
        else:
            # Far - full speed allowed
            max_speed = settings.SPEED_MAX

        self.set_max_speed(max_speed)

    def update(self) -> int:
        """
        Update speed controller and apply changes.

        This should be called periodically in the main control loop.
        It gradually adjusts speed toward the target.

        Returns:
            Current actual speed after update
        """
        with self._lock:
            if not self._enabled:
                return self._current_speed

            # Calculate effective target (limited by max)
            effective_target = min(self._target_speed, self._max_allowed_speed)

            # Calculate speed difference
            speed_diff = effective_target - self._current_speed

            if speed_diff == 0:
                return self._current_speed

            # Determine step size based on direction
            if abs(speed_diff) <= self._acceleration_step:
                # Close enough, jump to target
                new_speed = effective_target
            elif speed_diff > 0:
                # Accelerating
                new_speed = self._current_speed + self._acceleration_step
            else:
                # Decelerating
                new_speed = self._current_speed - self._acceleration_step

            # Apply new speed
            new_speed = max(0, min(100, new_speed))
            self._motor.set_speed(new_speed)
            self._current_speed = new_speed

            return self._current_speed

    def update_pid(self, actual_speed: float) -> int:
        """
        Update using PID control (for cruise control).

        Uses the difference between target and actual measured
        speed to calculate corrections.

        Args:
            actual_speed: Measured actual speed (e.g., from encoder)

        Returns:
            Corrected speed output
        """
        with self._lock:
            current_time = time.time()
            dt = current_time - self._last_update_time
            self._last_update_time = current_time

            if dt <= 0:
                return self._current_speed

            # Calculate error
            error = self._target_speed - actual_speed

            # Proportional term
            p_term = self._pid.kp * error

            # Integral term (with anti-windup)
            self._integral += error * dt
            self._integral = max(-50, min(50, self._integral))  # Limit integral
            i_term = self._pid.ki * self._integral

            # Derivative term
            d_term = self._pid.kd * (error - self._last_error) / dt
            self._last_error = error

            # Calculate output
            output = self._current_speed + p_term + i_term + d_term
            output = max(0, min(self._max_allowed_speed, output))

            new_speed = int(output)
            self._motor.set_speed(new_speed)
            self._current_speed = new_speed

            return new_speed

    def ramp_to_speed(
        self,
        target: int,
        blocking: bool = True,
    ) -> None:
        """
        Smoothly ramp to a target speed.

        Args:
            target: Target speed (0-100)
            blocking: If True, block until target reached
        """
        self.set_target_speed(target)

        if blocking:
            while self._current_speed != min(target, self._max_allowed_speed):
                self.update()
                time.sleep(self._acceleration_delay)

    def stop(self, gradual: bool = True) -> None:
        """
        Stop the vehicle.

        Args:
            gradual: If True, gradually decelerate. If False, immediate stop.
        """
        if gradual:
            self.ramp_to_speed(0, blocking=True)
        else:
            with self._lock:
                self._target_speed = 0
                self._current_speed = 0
                self._motor.stop()

    def emergency_stop(self) -> None:
        """Immediate emergency stop."""
        with self._lock:
            self._target_speed = 0
            self._current_speed = 0
            self._motor.emergency_stop()
            self._integral = 0  # Reset PID state

    def enable(self) -> None:
        """Enable the speed controller."""
        with self._lock:
            self._enabled = True
            logger.info("SpeedController enabled")

    def disable(self) -> None:
        """Disable the speed controller (maintains current speed)."""
        with self._lock:
            self._enabled = False
            logger.info("SpeedController disabled")

    def reset(self) -> None:
        """Reset controller state."""
        with self._lock:
            self._target_speed = 0
            self._current_speed = 0
            self._integral = 0
            self._last_error = 0
            self._max_allowed_speed = self._settings.SPEED_MAX

    @property
    def target_speed(self) -> int:
        """Get current target speed."""
        return self._target_speed

    @property
    def current_speed(self) -> int:
        """Get current actual speed."""
        return self._current_speed

    @property
    def max_allowed_speed(self) -> int:
        """Get current maximum allowed speed."""
        return self._max_allowed_speed

    @property
    def is_at_target(self) -> bool:
        """Check if current speed equals target."""
        return self._current_speed == min(self._target_speed, self._max_allowed_speed)

    @property
    def is_enabled(self) -> bool:
        """Check if controller is enabled."""
        return self._enabled


class AdaptiveSpeedController(SpeedController):
    """
    Adaptive speed controller that adjusts based on conditions.

    Extends SpeedController with:
    - Automatic speed adjustment based on sensor data
    - Learning optimal speeds for different conditions
    - Battery-aware speed limiting
    """

    def __init__(
        self,
        motor_controller: MotorController,
        get_front_distance: Callable[[], float],
    ):
        """
        Initialize adaptive speed controller.

        Args:
            motor_controller: MotorController instance
            get_front_distance: Callback to get current front distance
        """
        super().__init__(motor_controller)
        self._get_front_distance = get_front_distance

    def update_adaptive(self) -> int:
        """
        Update with adaptive speed limiting.

        Automatically limits speed based on front distance sensor.

        Returns:
            Current speed after update
        """
        # Get current distance and limit speed
        distance = self._get_front_distance()
        self.limit_speed_by_distance(distance)

        # Run normal update
        return self.update()
