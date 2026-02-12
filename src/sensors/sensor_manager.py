"""
Sensor Manager for Autonomous Solar Vehicle.

This module provides high-level management of all vehicle sensors,
coordinating sensor reading, caching, and health monitoring.

Usage:
    from src.sensors.sensor_manager import SensorManager

    manager = SensorManager()
    manager.start()

    data = manager.get_sensor_data()
    manager.stop()
"""

import time
import threading
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import logging

from src.sensors.ultrasonic import UltrasonicSensorArray, SensorReading
from src.sensors.camera import CameraManager
from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SensorHealth:
    """Health status of a sensor."""
    sensor_id: str
    is_healthy: bool
    last_reading_time: float
    error_count: int
    last_error: Optional[str] = None


class SensorManager:
    """
    High-level manager for all vehicle sensors.

    Provides:
    - Coordinated sensor initialization and cleanup
    - Continuous sensor reading (threaded)
    - Health monitoring and error tracking
    - Cached access to latest readings

    Args:
        auto_start: Start sensor threads automatically
    """

    def __init__(self, auto_start: bool = False):
        """Initialize the sensor manager."""
        self._settings = get_settings()

        # Create sensor instances
        self._ultrasonic_array = UltrasonicSensorArray()
        self._camera_manager = CameraManager()

        # State
        self._running = False
        self._lock = threading.Lock()

        # Cached readings
        self._latest_ultrasonic: Dict[str, float] = {}
        self._latest_readings_time: float = 0

        # Health tracking
        self._sensor_health: Dict[str, SensorHealth] = {}
        self._error_counts: Dict[str, int] = {}

        # Threads
        self._sensor_thread: Optional[threading.Thread] = None

        logger.info("SensorManager created")

        if auto_start:
            self.start()

    def start(self) -> bool:
        """
        Start all sensors and sensor reading thread.

        Returns:
            True if started successfully.
        """
        try:
            with self._lock:
                if self._running:
                    return True

                # Initialize ultrasonic sensors
                init_results = self._ultrasonic_array.initialize_all()
                for sensor_id, success in init_results.items():
                    self._sensor_health[sensor_id] = SensorHealth(
                        sensor_id=sensor_id,
                        is_healthy=success,
                        last_reading_time=time.time(),
                        error_count=0 if success else 1,
                        last_error=None if success else "Initialization failed",
                    )

                # Start camera
                cam_results = self._camera_manager.start()

                # Start sensor reading thread
                self._running = True
                self._sensor_thread = threading.Thread(
                    target=self._sensor_loop,
                    name="SensorLoop",
                    daemon=True,
                )
                self._sensor_thread.start()

                logger.info("SensorManager started")
                return True

        except Exception as e:
            logger.error(f"SensorManager start failed: {e}")
            return False

    def stop(self) -> None:
        """Stop all sensors and reading thread."""
        with self._lock:
            self._running = False

        # Wait for thread to finish
        if self._sensor_thread:
            self._sensor_thread.join(timeout=2.0)
            self._sensor_thread = None

        # Clean up sensors
        self._ultrasonic_array.cleanup_all()
        self._camera_manager.stop()

        logger.info("SensorManager stopped")

    def _sensor_loop(self) -> None:
        """Background thread for continuous sensor reading."""
        loop_period = 1.0 / self._settings.MAIN_LOOP_FREQUENCY_HZ

        while self._running:
            loop_start = time.time()

            try:
                # Read all ultrasonic sensors
                distances = self._ultrasonic_array.read_all_distances()

                with self._lock:
                    self._latest_ultrasonic = distances
                    self._latest_readings_time = time.time()

                # Update health status
                for sensor_id, distance in distances.items():
                    health = self._sensor_health.get(sensor_id)
                    if health:
                        if distance == 0:
                            health.error_count += 1
                            health.is_healthy = health.error_count < 5
                        else:
                            health.is_healthy = True
                            health.last_reading_time = time.time()

            except Exception as e:
                logger.error(f"Sensor loop error: {e}")

            # Maintain loop frequency
            elapsed = time.time() - loop_start
            sleep_time = loop_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_distances(self) -> Dict[str, float]:
        """
        Get latest ultrasonic distances.

        Returns:
            Dictionary mapping sensor_id to distance in cm.
        """
        with self._lock:
            return self._latest_ultrasonic.copy()

    def get_front_distance(self) -> float:
        """
        Get minimum front distance (redundant sensors).

        Returns:
            Minimum of front and front_wp sensors.
        """
        distances = self.get_distances()
        front = distances.get("US_FRONT", 0)
        front_wp = distances.get("US_FRONT_WP", 0)
        return min(front, front_wp) if front > 0 and front_wp > 0 else max(front, front_wp)

    def get_sensor_health(self) -> Dict[str, SensorHealth]:
        """
        Get health status of all sensors.

        Returns:
            Dictionary mapping sensor_id to SensorHealth.
        """
        return self._sensor_health.copy()

    def get_all_healthy(self) -> bool:
        """Check if all sensors are healthy."""
        return all(h.is_healthy for h in self._sensor_health.values())

    def get_sensor_data(self) -> Dict[str, Any]:
        """
        Get comprehensive sensor data package.

        Returns:
            Dictionary with all sensor data.
        """
        distances = self.get_distances()
        health = self.get_sensor_health()

        return {
            "ultrasonic": distances,
            "front_min": min(
                distances.get("US_FRONT", 999),
                distances.get("US_FRONT_WP", 999)
            ),
            "timestamp": self._latest_readings_time,
            "all_healthy": self.get_all_healthy(),
            "health": {k: v.is_healthy for k, v in health.items()},
        }

    @property
    def ultrasonic_array(self) -> UltrasonicSensorArray:
        """Get the ultrasonic sensor array instance."""
        return self._ultrasonic_array

    @property
    def camera_manager(self) -> CameraManager:
        """Get the camera manager instance."""
        return self._camera_manager

    @property
    def is_running(self) -> bool:
        """Check if sensor manager is running."""
        return self._running
