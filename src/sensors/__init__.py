"""
Sensor Module for Autonomous Solar Vehicle.

This module provides hardware interfaces for all vehicle sensors:
- Ultrasonic sensors (HC-SR04): Distance measurement
- Cameras: Visual perception
- Sensor fusion: Combined data processing

All sensor classes are designed to be thread-safe and handle
hardware errors gracefully.
"""

from src.sensors.ultrasonic import UltrasonicSensor, UltrasonicSensorArray
from src.sensors.sensor_manager import SensorManager

__all__ = [
    "UltrasonicSensor",
    "UltrasonicSensorArray",
    "SensorManager",
]
