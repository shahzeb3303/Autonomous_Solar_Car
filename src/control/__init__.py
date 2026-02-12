"""
Control Module for Autonomous Solar Vehicle.

This module provides motor control and steering interfaces:
- MotorController: Low-level L298N motor driver interface
- Steering: Differential steering logic for tank-style movement
- SpeedController: Speed management with ramping

All control classes handle GPIO cleanup on shutdown and provide
emergency stop capabilities.
"""

from src.control.motor_controller import MotorController, MotorDirection
from src.control.steering import DifferentialSteering
from src.control.speed_controller import SpeedController

__all__ = [
    "MotorController",
    "MotorDirection",
    "DifferentialSteering",
    "SpeedController",
]
