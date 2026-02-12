"""
Ultrasonic Sensor Driver for HC-SR04 sensors.

This module provides a robust driver for HC-SR04 ultrasonic distance sensors
with timeout protection, measurement filtering, and error handling.

Hardware:
- HC-SR04 ultrasonic ranging module
- Operating voltage: 5V (trigger/echo are 3.3V compatible)
- Range: 2cm - 400cm
- Accuracy: ±3mm

Operating principle:
1. Send 10μs HIGH pulse on TRIGGER pin
2. Sensor sends 8 cycles of 40kHz ultrasonic burst
3. ECHO pin goes HIGH
4. ECHO pin stays HIGH for time proportional to distance
5. Distance = (pulse_duration × speed_of_sound) / 2

Usage:
    from config import GPIOMap
    from src.sensors.ultrasonic import UltrasonicSensor

    sensor = UltrasonicSensor(
        sensor_id="US_FRONT",
        trigger_pin=GPIOMap.ULTRASONIC_FRONT.trigger,
        echo_pin=GPIOMap.ULTRASONIC_FRONT.echo
    )

    distance = sensor.measure()
    print(f"Distance: {distance:.1f} cm")
"""

import time
import threading
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Try to import RPi.GPIO, fall back to mock for development
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    GPIO = None

from config.settings import get_settings
from config.gpio_map import GPIOMap, UltrasonicPins

logger = logging.getLogger(__name__)


class SensorStatus(Enum):
    """Status codes for sensor readings."""
    OK = "ok"
    TIMEOUT = "timeout"
    OUT_OF_RANGE = "out_of_range"
    ERROR = "error"
    NOT_INITIALIZED = "not_initialized"


@dataclass
class SensorReading:
    """
    Container for a single sensor reading with metadata.

    Attributes:
        distance_cm: Measured distance in centimeters (None if invalid)
        status: Status code indicating reading validity
        timestamp: Unix timestamp of the reading
        raw_pulse_duration: Raw pulse duration in seconds (for debugging)
    """
    distance_cm: Optional[float]
    status: SensorStatus
    timestamp: float
    raw_pulse_duration: Optional[float] = None

    @property
    def is_valid(self) -> bool:
        """Check if the reading is valid and usable."""
        return self.status == SensorStatus.OK and self.distance_cm is not None


class MockGPIO:
    """
    Mock GPIO interface for development/testing on non-RPi systems.

    Simulates GPIO operations and returns plausible sensor readings.
    """

    BCM = "BCM"
    OUT = "OUT"
    IN = "IN"
    HIGH = True
    LOW = False

    _output_states: Dict[int, bool] = {}
    _mock_distances: Dict[int, float] = {}

    @classmethod
    def setmode(cls, mode: str) -> None:
        """Set GPIO numbering mode (no-op in mock)."""
        logger.debug(f"MockGPIO: setmode({mode})")

    @classmethod
    def setwarnings(cls, state: bool) -> None:
        """Set warnings state (no-op in mock)."""
        pass

    @classmethod
    def setup(cls, pin: int, direction: str) -> None:
        """Set up a GPIO pin."""
        logger.debug(f"MockGPIO: setup(pin={pin}, dir={direction})")

    @classmethod
    def output(cls, pin: int, state: bool) -> None:
        """Set output pin state."""
        cls._output_states[pin] = state

    @classmethod
    def input(cls, pin: int) -> bool:
        """
        Read input pin state.

        For testing, this simulates echo pin behavior.
        """
        # Simulate random-ish distance for testing
        import random
        return random.choice([True, False])

    @classmethod
    def cleanup(cls, pin: Optional[int] = None) -> None:
        """Clean up GPIO (no-op in mock)."""
        logger.debug(f"MockGPIO: cleanup(pin={pin})")

    @classmethod
    def set_mock_distance(cls, echo_pin: int, distance_cm: float) -> None:
        """Set a mock distance for testing purposes."""
        cls._mock_distances[echo_pin] = distance_cm


class UltrasonicSensor:
    """
    Driver for a single HC-SR04 ultrasonic distance sensor.

    Features:
    - Timeout protection to prevent GPIO blocking
    - Measurement filtering (multiple samples, outlier rejection)
    - Thread-safe operation
    - Graceful error handling

    Args:
        sensor_id: Unique identifier for this sensor (e.g., "US_FRONT")
        trigger_pin: GPIO pin number for trigger (BCM numbering)
        echo_pin: GPIO pin number for echo (BCM numbering)
        timeout_s: Maximum time to wait for echo (default from settings)
        sample_count: Number of samples for averaging (default from settings)
    """

    def __init__(
        self,
        sensor_id: str,
        trigger_pin: int,
        echo_pin: int,
        timeout_s: Optional[float] = None,
        sample_count: Optional[int] = None,
    ):
        """Initialize the ultrasonic sensor."""
        self._settings = get_settings()

        self.sensor_id = sensor_id
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.timeout_s = timeout_s or self._settings.ULTRASONIC_TIMEOUT_S
        self.sample_count = sample_count or self._settings.ULTRASONIC_SAMPLE_COUNT

        self._initialized = False
        self._lock = threading.Lock()
        self._last_reading: Optional[SensorReading] = None

        # Use mock GPIO if real GPIO not available
        self._gpio = GPIO if GPIO_AVAILABLE else MockGPIO

        logger.info(
            f"UltrasonicSensor created: id={sensor_id}, "
            f"trigger={trigger_pin}, echo={echo_pin}"
        )

    def initialize(self) -> bool:
        """
        Initialize GPIO pins for this sensor.

        Must be called before taking measurements. Can be called multiple
        times safely (idempotent).

        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            with self._lock:
                if self._initialized:
                    return True

                # Set up GPIO mode (only if not already set)
                try:
                    self._gpio.setmode(self._gpio.BCM)
                except Exception:
                    pass  # Mode may already be set

                self._gpio.setwarnings(False)

                # Configure pins
                self._gpio.setup(self.trigger_pin, self._gpio.OUT)
                self._gpio.setup(self.echo_pin, self._gpio.IN)

                # Ensure trigger starts LOW
                self._gpio.output(self.trigger_pin, self._gpio.LOW)

                # Allow sensor to settle
                time.sleep(self._settings.ULTRASONIC_SETTLE_TIME_S)

                self._initialized = True
                logger.info(f"Sensor {self.sensor_id} initialized successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to initialize sensor {self.sensor_id}: {e}")
            return False

    def cleanup(self) -> None:
        """
        Clean up GPIO resources.

        Should be called when done with the sensor or on program exit.
        """
        try:
            with self._lock:
                if self._initialized:
                    self._gpio.cleanup(self.trigger_pin)
                    self._gpio.cleanup(self.echo_pin)
                    self._initialized = False
                    logger.info(f"Sensor {self.sensor_id} cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup of {self.sensor_id}: {e}")

    def _measure_single(self) -> SensorReading:
        """
        Take a single distance measurement.

        This is the core measurement function that handles the trigger/echo
        protocol with timeout protection.

        Returns:
            SensorReading with distance or error status.
        """
        if not self._initialized:
            return SensorReading(
                distance_cm=None,
                status=SensorStatus.NOT_INITIALIZED,
                timestamp=time.time(),
            )

        try:
            # Send 10μs trigger pulse
            self._gpio.output(self.trigger_pin, self._gpio.HIGH)
            time.sleep(self._settings.ULTRASONIC_TRIGGER_PULSE_S)
            self._gpio.output(self.trigger_pin, self._gpio.LOW)

            # Wait for echo to go HIGH (start of pulse)
            pulse_start = time.time()
            timeout_time = pulse_start + self.timeout_s

            while self._gpio.input(self.echo_pin) == self._gpio.LOW:
                pulse_start = time.time()
                if pulse_start > timeout_time:
                    return SensorReading(
                        distance_cm=None,
                        status=SensorStatus.TIMEOUT,
                        timestamp=time.time(),
                    )

            # Wait for echo to go LOW (end of pulse)
            pulse_end = time.time()
            timeout_time = pulse_end + self.timeout_s

            while self._gpio.input(self.echo_pin) == self._gpio.HIGH:
                pulse_end = time.time()
                if pulse_end > timeout_time:
                    return SensorReading(
                        distance_cm=None,
                        status=SensorStatus.TIMEOUT,
                        timestamp=time.time(),
                    )

            # Calculate distance from pulse duration
            pulse_duration = pulse_end - pulse_start
            distance_cm = (pulse_duration * self._settings.SOUND_SPEED_CM_PER_S) / 2.0

            # Validate range
            if distance_cm < self._settings.ULTRASONIC_MIN_RANGE_CM:
                return SensorReading(
                    distance_cm=distance_cm,
                    status=SensorStatus.OUT_OF_RANGE,
                    timestamp=time.time(),
                    raw_pulse_duration=pulse_duration,
                )

            if distance_cm > self._settings.ULTRASONIC_MAX_RANGE_CM:
                return SensorReading(
                    distance_cm=None,
                    status=SensorStatus.OUT_OF_RANGE,
                    timestamp=time.time(),
                    raw_pulse_duration=pulse_duration,
                )

            return SensorReading(
                distance_cm=distance_cm,
                status=SensorStatus.OK,
                timestamp=time.time(),
                raw_pulse_duration=pulse_duration,
            )

        except Exception as e:
            logger.error(f"Error measuring {self.sensor_id}: {e}")
            return SensorReading(
                distance_cm=None,
                status=SensorStatus.ERROR,
                timestamp=time.time(),
            )

    def measure(self) -> SensorReading:
        """
        Take a filtered distance measurement.

        Takes multiple samples, rejects outliers, and returns the median.
        This provides more stable readings than single measurements.

        Returns:
            SensorReading with averaged/filtered distance.
        """
        with self._lock:
            if not self._initialized:
                if not self.initialize():
                    return SensorReading(
                        distance_cm=None,
                        status=SensorStatus.NOT_INITIALIZED,
                        timestamp=time.time(),
                    )

            # Collect multiple samples
            readings: List[float] = []
            error_count = 0

            for _ in range(self.sample_count):
                reading = self._measure_single()

                if reading.is_valid and reading.distance_cm is not None:
                    readings.append(reading.distance_cm)
                else:
                    error_count += 1

                # Small delay between readings
                time.sleep(self._settings.ULTRASONIC_SETTLE_TIME_S)

            # If all readings failed, return error
            if not readings:
                return SensorReading(
                    distance_cm=None,
                    status=SensorStatus.ERROR,
                    timestamp=time.time(),
                )

            # Filter outliers and calculate median
            filtered_distance = self._filter_readings(readings)

            self._last_reading = SensorReading(
                distance_cm=filtered_distance,
                status=SensorStatus.OK,
                timestamp=time.time(),
            )

            return self._last_reading

    def _filter_readings(self, readings: List[float]) -> float:
        """
        Filter readings to reject outliers and return median.

        Args:
            readings: List of distance measurements in cm.

        Returns:
            Filtered/median distance value.
        """
        if len(readings) == 1:
            return readings[0]

        # Sort readings
        sorted_readings = sorted(readings)

        # Calculate median
        mid = len(sorted_readings) // 2
        if len(sorted_readings) % 2 == 0:
            median = (sorted_readings[mid - 1] + sorted_readings[mid]) / 2.0
        else:
            median = sorted_readings[mid]

        # Reject outliers (readings too far from median)
        threshold = self._settings.ULTRASONIC_OUTLIER_THRESHOLD
        filtered = [r for r in readings if abs(r - median) <= threshold]

        # Return average of filtered readings, or median if all rejected
        if filtered:
            return sum(filtered) / len(filtered)
        return median

    def get_distance(self) -> float:
        """
        Get distance measurement, returning 0 on error (fail-safe).

        This is a convenience method that always returns a valid number.
        Returns 0 (indicating obstacle present) on any error - this is
        the safest failure mode for obstacle detection.

        Returns:
            Distance in cm, or 0.0 on error.
        """
        reading = self.measure()
        if reading.is_valid and reading.distance_cm is not None:
            return reading.distance_cm
        else:
            # Fail-safe: assume obstacle is very close
            logger.warning(
                f"Sensor {self.sensor_id} error ({reading.status.value}), "
                f"returning 0 (fail-safe)"
            )
            return 0.0

    @property
    def last_reading(self) -> Optional[SensorReading]:
        """Get the most recent reading without taking a new measurement."""
        return self._last_reading

    @property
    def is_initialized(self) -> bool:
        """Check if sensor has been initialized."""
        return self._initialized


class UltrasonicSensorArray:
    """
    Manager for multiple ultrasonic sensors.

    Provides coordinated access to all vehicle ultrasonic sensors with:
    - Bulk initialization and cleanup
    - Parallel or sequential reading
    - Sensor redundancy handling (front sensors)

    Usage:
        array = UltrasonicSensorArray()
        array.initialize_all()

        readings = array.read_all()
        front = readings["US_FRONT"]
    """

    def __init__(self):
        """Initialize the sensor array with all configured sensors."""
        self._settings = get_settings()
        gpio_map = GPIOMap()

        # Create all sensors from GPIO map
        self._sensors: Dict[str, UltrasonicSensor] = {}

        sensor_configs = gpio_map.get_all_ultrasonic_sensors()
        for sensor_id, pins in sensor_configs.items():
            self._sensors[sensor_id] = UltrasonicSensor(
                sensor_id=sensor_id,
                trigger_pin=pins.trigger,
                echo_pin=pins.echo,
            )

        self._lock = threading.Lock()
        logger.info(f"UltrasonicSensorArray created with {len(self._sensors)} sensors")

    def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all sensors.

        Returns:
            Dictionary mapping sensor_id to initialization success.
        """
        results = {}
        for sensor_id, sensor in self._sensors.items():
            results[sensor_id] = sensor.initialize()
        return results

    def cleanup_all(self) -> None:
        """Clean up all sensors."""
        for sensor in self._sensors.values():
            sensor.cleanup()

    def read_all(self) -> Dict[str, SensorReading]:
        """
        Read all sensors sequentially.

        Sequential reading is used because HC-SR04 sensors can interfere
        with each other if triggered simultaneously.

        Returns:
            Dictionary mapping sensor_id to SensorReading.
        """
        with self._lock:
            readings = {}
            for sensor_id, sensor in self._sensors.items():
                readings[sensor_id] = sensor.measure()
            return readings

    def read_all_distances(self) -> Dict[str, float]:
        """
        Read all sensors and return distances only.

        Uses fail-safe mode: returns 0 for any sensor errors.

        Returns:
            Dictionary mapping sensor_id to distance in cm.
        """
        with self._lock:
            distances = {}
            for sensor_id, sensor in self._sensors.items():
                distances[sensor_id] = sensor.get_distance()
            return distances

    def get_front_distance(self) -> float:
        """
        Get front distance with redundancy.

        Uses minimum of US_FRONT and US_FRONT_WP for safety.
        This ensures we always use the more cautious reading.

        Returns:
            Minimum front distance in cm.
        """
        front = self._sensors["US_FRONT"].get_distance()
        front_wp = self._sensors["US_FRONT_WP"].get_distance()

        # Use minimum (more cautious reading)
        min_distance = min(front, front_wp)

        # Log if sensors disagree significantly
        if abs(front - front_wp) > 50:
            logger.warning(
                f"Front sensors disagree: US_FRONT={front:.1f}, "
                f"US_FRONT_WP={front_wp:.1f}, using min={min_distance:.1f}"
            )

        return min_distance

    def get_sensor(self, sensor_id: str) -> Optional[UltrasonicSensor]:
        """
        Get a specific sensor by ID.

        Args:
            sensor_id: The sensor identifier (e.g., "US_FRONT")

        Returns:
            UltrasonicSensor instance or None if not found.
        """
        return self._sensors.get(sensor_id)

    @property
    def sensor_ids(self) -> List[str]:
        """Get list of all sensor IDs."""
        return list(self._sensors.keys())
