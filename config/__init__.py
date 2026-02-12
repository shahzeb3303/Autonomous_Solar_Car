"""
Configuration module for Autonomous Solar Vehicle.

This module provides centralized configuration management including:
- GPIO pin mappings
- Sensor thresholds and parameters
- Model paths and inference settings
- Logging configuration

All configurable values should be imported from this module to ensure
a single source of truth across the codebase.
"""

from config.settings import Settings, get_settings
from config.gpio_map import GPIOMap
from config.logging_config import setup_logging, get_logger

__all__ = [
    "Settings",
    "get_settings",
    "GPIOMap",
    "setup_logging",
    "get_logger",
]
