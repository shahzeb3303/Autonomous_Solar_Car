"""
Odometry - Vehicle position tracking.

Estimates vehicle position based on motor commands and elapsed time.
This is a dead-reckoning system - errors accumulate over time.

For better accuracy, this should be augmented with:
- Wheel encoders
- GPS
- Visual odometry
"""

import time
import math
from dataclasses import dataclass
from typing import Optional
import logging

from .grid_map import Position

logger = logging.getLogger(__name__)


@dataclass
class OdometryConfig:
    """Odometry configuration."""
    wheel_base_cm: float = 25.0      # Distance between wheels
    wheel_radius_cm: float = 5.0     # Wheel radius
    max_speed_cm_per_s: float = 50.0 # Max forward speed
    turn_rate_deg_per_s: float = 90.0 # Turn rate at full speed


class Odometry:
    """
    Dead-reckoning odometry for position estimation.

    Updates position based on:
    - Current action (FORWARD, TURN_LEFT, etc.)
    - Speed percentage
    - Elapsed time

    Position is in centimeters, heading in degrees.
    Coordinate system:
    - X: positive = right
    - Y: positive = forward (north)
    - Heading: 0 = north, 90 = east, 180 = south, 270 = west
    """

    def __init__(self, config: Optional[OdometryConfig] = None):
        """Initialize odometry."""
        self.config = config or OdometryConfig()

        # Current position
        self.x = 0.0          # cm
        self.y = 0.0          # cm
        self.heading = 0.0    # degrees (0 = north)

        # Tracking
        self.last_update_time = time.time()
        self.total_distance = 0.0
        self.total_rotation = 0.0

        logger.info("Odometry initialized at origin (0, 0), heading=0")

    def update(self, action: str, speed_percent: float) -> Position:
        """
        Update position based on current action and speed.

        Args:
            action: Current action (FORWARD, TURN_LEFT, TURN_RIGHT, STOP, etc.)
            speed_percent: Speed as percentage (0-100)

        Returns:
            Updated position
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Calculate actual speed
        speed_fraction = speed_percent / 100.0
        linear_speed = self.config.max_speed_cm_per_s * speed_fraction  # cm/s
        angular_speed = self.config.turn_rate_deg_per_s * speed_fraction  # deg/s

        # Update based on action
        if action == "FORWARD":
            self._move_forward(linear_speed, dt)

        elif action == "TURN_LEFT":
            self._turn(angular_speed, dt, direction=-1)
            self._move_forward(linear_speed * 0.5, dt)  # Move slower while turning

        elif action == "TURN_RIGHT":
            self._turn(angular_speed, dt, direction=1)
            self._move_forward(linear_speed * 0.5, dt)

        elif action == "REVERSE_LEFT":
            self._turn(angular_speed, dt, direction=-1)
            self._move_backward(linear_speed * 0.3, dt)

        elif action == "REVERSE_RIGHT":
            self._turn(angular_speed, dt, direction=1)
            self._move_backward(linear_speed * 0.3, dt)

        elif action == "SLOW_DOWN":
            self._move_forward(linear_speed * 0.5, dt)

        # STOP - no movement

        return self.get_position()

    def _move_forward(self, speed: float, dt: float):
        """Move forward at given speed for dt seconds."""
        distance = speed * dt
        heading_rad = math.radians(self.heading)

        # Update position (heading 0 = positive Y)
        self.x += distance * math.sin(heading_rad)
        self.y += distance * math.cos(heading_rad)
        self.total_distance += distance

    def _move_backward(self, speed: float, dt: float):
        """Move backward at given speed for dt seconds."""
        distance = speed * dt
        heading_rad = math.radians(self.heading)

        # Move in opposite direction
        self.x -= distance * math.sin(heading_rad)
        self.y -= distance * math.cos(heading_rad)
        self.total_distance += distance

    def _turn(self, angular_speed: float, dt: float, direction: int):
        """
        Turn at given angular speed.

        Args:
            angular_speed: Degrees per second
            dt: Time delta
            direction: -1 for left, +1 for right
        """
        angle_change = angular_speed * dt * direction
        self.heading = (self.heading + angle_change) % 360
        self.total_rotation += abs(angle_change)

    def get_position(self) -> Position:
        """Get current position."""
        return Position(self.x, self.y, self.heading)

    def set_position(self, x: float, y: float, heading: float = 0.0):
        """
        Set position manually (e.g., from GPS or user input).

        Args:
            x: X position in cm
            y: Y position in cm
            heading: Heading in degrees
        """
        self.x = x
        self.y = y
        self.heading = heading % 360
        logger.info(f"Position set to ({x:.1f}, {y:.1f}), heading={heading:.1f}")

    def reset(self):
        """Reset odometry to origin."""
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.total_distance = 0.0
        self.total_rotation = 0.0
        self.last_update_time = time.time()
        logger.info("Odometry reset to origin")

    def get_statistics(self) -> dict:
        """Get odometry statistics."""
        return {
            'position': (self.x, self.y),
            'heading': self.heading,
            'total_distance_cm': self.total_distance,
            'total_rotation_deg': self.total_rotation,
        }
