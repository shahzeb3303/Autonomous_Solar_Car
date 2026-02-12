"""
Unit Tests for Ultrasonic Sensor Module.

Tests cover:
- Sensor initialization and GPIO setup
- Distance measurement and filtering
- Timeout handling
- Error conditions
- Sensor array operations
- Redundancy handling

Run with: pytest tests/test_ultrasonic.py -v
"""

import pytest
import time
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestUltrasonicSensor:
    """Tests for the UltrasonicSensor class."""

    def test_sensor_creation(self, mock_gpio):
        """Test sensor object creation."""
        from src.sensors.ultrasonic import UltrasonicSensor

        sensor = UltrasonicSensor(
            sensor_id="US_TEST",
            trigger_pin=17,
            echo_pin=27,
        )

        assert sensor.sensor_id == "US_TEST"
        assert sensor.trigger_pin == 17
        assert sensor.echo_pin == 27
        assert not sensor.is_initialized

    def test_sensor_initialization(self, mock_gpio):
        """Test sensor GPIO initialization."""
        from src.sensors.ultrasonic import UltrasonicSensor

        sensor = UltrasonicSensor(
            sensor_id="US_TEST",
            trigger_pin=17,
            echo_pin=27,
        )

        result = sensor.initialize()

        assert result is True
        assert sensor.is_initialized

        # Verify GPIO setup was called
        call_log = mock_gpio.get_call_log()
        setup_calls = [c for c in call_log if c[0] == 'setup']
        assert len(setup_calls) >= 2  # trigger and echo

    def test_sensor_cleanup(self, mock_gpio):
        """Test sensor cleanup."""
        from src.sensors.ultrasonic import UltrasonicSensor

        sensor = UltrasonicSensor(
            sensor_id="US_TEST",
            trigger_pin=17,
            echo_pin=27,
        )
        sensor.initialize()
        sensor.cleanup()

        assert not sensor.is_initialized

    def test_sensor_double_initialization(self, mock_gpio):
        """Test that double initialization is safe."""
        from src.sensors.ultrasonic import UltrasonicSensor

        sensor = UltrasonicSensor(
            sensor_id="US_TEST",
            trigger_pin=17,
            echo_pin=27,
        )

        sensor.initialize()
        result = sensor.initialize()  # Second call

        assert result is True
        assert sensor.is_initialized

    def test_get_distance_returns_float(self, mock_gpio):
        """Test that get_distance always returns a float."""
        from src.sensors.ultrasonic import UltrasonicSensor

        sensor = UltrasonicSensor(
            sensor_id="US_TEST",
            trigger_pin=17,
            echo_pin=27,
        )

        # Without initialization, should return 0 (fail-safe)
        distance = sensor.get_distance()
        assert isinstance(distance, float)
        assert distance == 0.0  # Fail-safe value

    def test_uninitialized_sensor_fails_safe(self, mock_gpio):
        """Test that uninitialized sensor returns fail-safe value."""
        from src.sensors.ultrasonic import UltrasonicSensor, SensorStatus

        sensor = UltrasonicSensor(
            sensor_id="US_TEST",
            trigger_pin=17,
            echo_pin=27,
        )

        reading = sensor.measure()

        # Should auto-initialize and try to measure
        # In mock mode, will likely timeout or error
        assert reading.distance_cm is None or reading.distance_cm >= 0

    def test_sensor_reading_dataclass(self, mock_gpio):
        """Test SensorReading dataclass behavior."""
        from src.sensors.ultrasonic import SensorReading, SensorStatus

        # Valid reading
        valid = SensorReading(
            distance_cm=50.0,
            status=SensorStatus.OK,
            timestamp=time.time(),
        )
        assert valid.is_valid
        assert valid.distance_cm == 50.0

        # Invalid reading (timeout)
        timeout = SensorReading(
            distance_cm=None,
            status=SensorStatus.TIMEOUT,
            timestamp=time.time(),
        )
        assert not timeout.is_valid

        # Invalid reading (out of range)
        oor = SensorReading(
            distance_cm=1.0,  # Below minimum
            status=SensorStatus.OUT_OF_RANGE,
            timestamp=time.time(),
        )
        assert not oor.is_valid


class TestUltrasonicSensorFiltering:
    """Tests for sensor reading filtering."""

    def test_filter_single_reading(self, mock_gpio):
        """Test filtering with single reading."""
        from src.sensors.ultrasonic import UltrasonicSensor

        sensor = UltrasonicSensor(
            sensor_id="US_TEST",
            trigger_pin=17,
            echo_pin=27,
        )

        readings = [50.0]
        result = sensor._filter_readings(readings)

        assert result == 50.0

    def test_filter_multiple_readings_median(self, mock_gpio):
        """Test that median is calculated correctly."""
        from src.sensors.ultrasonic import UltrasonicSensor

        sensor = UltrasonicSensor(
            sensor_id="US_TEST",
            trigger_pin=17,
            echo_pin=27,
        )

        # Odd number of readings
        readings = [45.0, 50.0, 55.0]
        result = sensor._filter_readings(readings)
        assert result == 50.0  # Median

        # Even number of readings
        readings = [45.0, 50.0, 55.0, 60.0]
        result = sensor._filter_readings(readings)
        assert result == 52.5  # Average of middle two

    def test_filter_rejects_outliers(self, mock_gpio):
        """Test that outliers are rejected."""
        from src.sensors.ultrasonic import UltrasonicSensor

        sensor = UltrasonicSensor(
            sensor_id="US_TEST",
            trigger_pin=17,
            echo_pin=27,
        )

        # Reading with one outlier (500 is way off from ~50)
        readings = [48.0, 50.0, 52.0, 500.0]
        result = sensor._filter_readings(readings)

        # Should exclude the 500 outlier and average the rest
        assert 45.0 < result < 55.0


class TestUltrasonicSensorArray:
    """Tests for the UltrasonicSensorArray class."""

    def test_array_creation(self, mock_gpio):
        """Test sensor array creation."""
        from src.sensors.ultrasonic import UltrasonicSensorArray

        array = UltrasonicSensorArray()

        # Should have 5 sensors
        assert len(array.sensor_ids) == 5
        assert "US_FRONT" in array.sensor_ids
        assert "US_FRONT_WP" in array.sensor_ids
        assert "US_RIGHT" in array.sensor_ids
        assert "US_LEFT" in array.sensor_ids
        assert "US_REAR" in array.sensor_ids

    def test_array_initialize_all(self, mock_gpio):
        """Test initializing all sensors."""
        from src.sensors.ultrasonic import UltrasonicSensorArray

        array = UltrasonicSensorArray()
        results = array.initialize_all()

        # All should initialize successfully
        assert all(results.values())

    def test_array_get_sensor(self, mock_gpio):
        """Test getting individual sensor from array."""
        from src.sensors.ultrasonic import UltrasonicSensorArray, UltrasonicSensor

        array = UltrasonicSensorArray()

        sensor = array.get_sensor("US_FRONT")
        assert sensor is not None
        assert isinstance(sensor, UltrasonicSensor)
        assert sensor.sensor_id == "US_FRONT"

        # Non-existent sensor
        assert array.get_sensor("US_NONEXISTENT") is None

    def test_array_cleanup_all(self, mock_gpio):
        """Test cleaning up all sensors."""
        from src.sensors.ultrasonic import UltrasonicSensorArray

        array = UltrasonicSensorArray()
        array.initialize_all()
        array.cleanup_all()

        # All sensors should be cleaned up
        for sensor_id in array.sensor_ids:
            sensor = array.get_sensor(sensor_id)
            assert not sensor.is_initialized


class TestSensorRedundancy:
    """Tests for front sensor redundancy."""

    def test_front_distance_uses_minimum(self, mock_gpio):
        """Test that get_front_distance returns minimum of both front sensors."""
        from src.sensors.ultrasonic import UltrasonicSensorArray
        from unittest.mock import patch

        array = UltrasonicSensorArray()

        # Mock the individual sensor get_distance methods
        with patch.object(array.get_sensor("US_FRONT"), 'get_distance', return_value=100.0):
            with patch.object(array.get_sensor("US_FRONT_WP"), 'get_distance', return_value=80.0):
                result = array.get_front_distance()
                assert result == 80.0  # Should use the smaller value

    def test_front_distance_with_sensor_disagreement(self, mock_gpio):
        """Test behavior when front sensors significantly disagree."""
        from src.sensors.ultrasonic import UltrasonicSensorArray
        from unittest.mock import patch

        array = UltrasonicSensorArray()

        # Sensors disagree by more than 50cm
        with patch.object(array.get_sensor("US_FRONT"), 'get_distance', return_value=100.0):
            with patch.object(array.get_sensor("US_FRONT_WP"), 'get_distance', return_value=30.0):
                result = array.get_front_distance()

                # Should still return minimum for safety
                assert result == 30.0


class TestSensorStatus:
    """Tests for SensorStatus enum."""

    def test_status_values(self):
        """Test all status enum values exist."""
        from src.sensors.ultrasonic import SensorStatus

        assert SensorStatus.OK.value == "ok"
        assert SensorStatus.TIMEOUT.value == "timeout"
        assert SensorStatus.OUT_OF_RANGE.value == "out_of_range"
        assert SensorStatus.ERROR.value == "error"
        assert SensorStatus.NOT_INITIALIZED.value == "not_initialized"


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_very_close_object(self, mock_gpio):
        """Test handling of object closer than minimum range."""
        from src.sensors.ultrasonic import SensorReading, SensorStatus

        # Distance below minimum (2cm)
        reading = SensorReading(
            distance_cm=1.5,
            status=SensorStatus.OUT_OF_RANGE,
            timestamp=time.time(),
        )

        assert not reading.is_valid
        assert reading.status == SensorStatus.OUT_OF_RANGE

    def test_very_far_object(self, mock_gpio):
        """Test handling of object beyond maximum range."""
        from src.sensors.ultrasonic import SensorReading, SensorStatus

        # Distance above maximum (400cm)
        reading = SensorReading(
            distance_cm=None,  # Should be None when out of range
            status=SensorStatus.OUT_OF_RANGE,
            timestamp=time.time(),
        )

        assert not reading.is_valid

    def test_concurrent_access(self, mock_gpio):
        """Test thread safety of sensor reading."""
        from src.sensors.ultrasonic import UltrasonicSensor
        import threading

        sensor = UltrasonicSensor(
            sensor_id="US_TEST",
            trigger_pin=17,
            echo_pin=27,
        )
        sensor.initialize()

        results = []
        errors = []

        def read_sensor():
            try:
                distance = sensor.get_distance()
                results.append(distance)
            except Exception as e:
                errors.append(e)

        # Create multiple threads reading simultaneously
        threads = [threading.Thread(target=read_sensor) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0
        assert len(results) == 5
