"""
Pytest Configuration and Shared Fixtures.

This module provides fixtures used across all test files:
- Mock GPIO for testing without hardware
- Mock sensors and controllers
- Test data generators
- Configuration overrides

All fixtures are designed to work on any system, not just Raspberry Pi.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Mock GPIO Module
# ============================================================================

class MockGPIOModule:
    """
    Complete mock of RPi.GPIO module for testing.

    This mock simulates GPIO behavior and tracks all calls
    for test verification.
    """

    BCM = "BCM"
    BOARD = "BOARD"
    OUT = "OUT"
    IN = "IN"
    HIGH = True
    LOW = False
    PUD_UP = "PUD_UP"
    PUD_DOWN = "PUD_DOWN"

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all mock state."""
        self._mode = None
        self._pin_modes: Dict[int, str] = {}
        self._pin_states: Dict[int, bool] = {}
        self._pwm_objects: Dict[int, 'MockPWM'] = {}
        self._warnings = True
        self._call_log = []

    def setmode(self, mode: str) -> None:
        self._mode = mode
        self._call_log.append(('setmode', mode))

    def getmode(self) -> str:
        return self._mode

    def setwarnings(self, state: bool) -> None:
        self._warnings = state
        self._call_log.append(('setwarnings', state))

    def setup(self, pin: int, direction: str, pull_up_down=None) -> None:
        self._pin_modes[pin] = direction
        self._pin_states[pin] = False
        self._call_log.append(('setup', pin, direction))

    def output(self, pin: int, state: bool) -> None:
        self._pin_states[pin] = state
        self._call_log.append(('output', pin, state))

    def input(self, pin: int) -> bool:
        return self._pin_states.get(pin, False)

    def cleanup(self, pin=None) -> None:
        if pin is not None:
            self._pin_modes.pop(pin, None)
            self._pin_states.pop(pin, None)
        else:
            self._pin_modes.clear()
            self._pin_states.clear()
        self._call_log.append(('cleanup', pin))

    def PWM(self, pin: int, frequency: int) -> 'MockPWM':
        pwm = MockPWM(pin, frequency)
        self._pwm_objects[pin] = pwm
        return pwm

    def set_pin_state(self, pin: int, state: bool) -> None:
        """Helper to set pin state for testing."""
        self._pin_states[pin] = state

    def get_call_log(self):
        """Get log of all GPIO calls for verification."""
        return self._call_log


class MockPWM:
    """Mock PWM class."""

    def __init__(self, pin: int, frequency: int):
        self.pin = pin
        self.frequency = frequency
        self.duty_cycle = 0
        self.running = False

    def start(self, duty_cycle: int) -> None:
        self.duty_cycle = duty_cycle
        self.running = True

    def ChangeDutyCycle(self, duty_cycle: int) -> None:
        self.duty_cycle = duty_cycle

    def ChangeFrequency(self, frequency: int) -> None:
        self.frequency = frequency

    def stop(self) -> None:
        self.running = False


# Create singleton mock GPIO
_mock_gpio = MockGPIOModule()


@pytest.fixture
def mock_gpio():
    """
    Provide mock GPIO module for testing.

    This fixture patches RPi.GPIO with a mock that tracks
    all GPIO operations.
    """
    _mock_gpio.reset()

    # Patch the GPIO module in all relevant places
    with patch.dict('sys.modules', {'RPi': MagicMock(), 'RPi.GPIO': _mock_gpio}):
        yield _mock_gpio


@pytest.fixture
def mock_gpio_with_echo():
    """
    Mock GPIO that simulates ultrasonic echo response.

    Simulates a 100cm distance reading by default.
    """
    _mock_gpio.reset()

    # Calculate pulse duration for 100cm
    # distance = pulse_duration * 34300 / 2
    # pulse_duration = distance * 2 / 34300
    target_distance = 100.0
    pulse_duration = target_distance * 2 / 34300  # ~0.00583s

    call_count = [0]
    start_time = [None]

    def mock_input(pin: int) -> bool:
        if pin in [27, 23, 25, 6, 16]:  # Echo pins
            call_count[0] += 1

            if call_count[0] == 1:
                # First call - waiting for HIGH
                start_time[0] = time.time()
                return False  # Still low
            elif call_count[0] == 2:
                return True  # Now high
            elif call_count[0] == 3:
                # Simulate pulse duration
                if time.time() - start_time[0] < pulse_duration:
                    return True  # Still high
                else:
                    return False  # Pulse ended
            else:
                return False
        return _mock_gpio._pin_states.get(pin, False)

    _mock_gpio.input = mock_input

    with patch.dict('sys.modules', {'RPi': MagicMock(), 'RPi.GPIO': _mock_gpio}):
        yield _mock_gpio


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_settings():
    """Provide test-specific settings."""
    from config.settings import Settings

    # Create settings with test-friendly values
    settings = Settings()
    return settings


@pytest.fixture
def mock_settings(monkeypatch):
    """Provide mocked settings for testing edge cases."""
    mock = MagicMock()
    mock.CRITICAL_DISTANCE_CM = 20.0
    mock.DANGER_DISTANCE_CM = 50.0
    mock.CAUTION_DISTANCE_CM = 100.0
    mock.ULTRASONIC_TIMEOUT_S = 0.03
    mock.ULTRASONIC_TRIGGER_PULSE_S = 0.00001
    mock.ULTRASONIC_SETTLE_TIME_S = 0.002
    mock.SOUND_SPEED_CM_PER_S = 34300.0
    mock.ULTRASONIC_MIN_RANGE_CM = 2.0
    mock.ULTRASONIC_MAX_RANGE_CM = 400.0
    mock.ULTRASONIC_SAMPLE_COUNT = 3
    mock.ULTRASONIC_OUTLIER_THRESHOLD = 50.0
    mock.PWM_FREQUENCY_HZ = 1000
    mock.SPEED_NORMAL = 50
    mock.SPEED_SLOW = 30
    mock.SPEED_MAX = 85
    mock.TURN_SPEED_INNER = 20
    mock.TURN_SPEED_OUTER = 50
    mock.ACCELERATION_STEP = 5
    mock.ACCELERATION_DELAY_S = 0.05

    return mock


# ============================================================================
# Sensor Fixtures
# ============================================================================

@pytest.fixture
def mock_ultrasonic_sensor(mock_gpio):
    """Create a mock ultrasonic sensor."""
    from src.sensors.ultrasonic import UltrasonicSensor

    sensor = UltrasonicSensor(
        sensor_id="US_TEST",
        trigger_pin=17,
        echo_pin=27,
    )
    return sensor


@pytest.fixture
def mock_sensor_array(mock_gpio):
    """Create a mock sensor array."""
    from src.sensors.ultrasonic import UltrasonicSensorArray

    array = UltrasonicSensorArray()
    return array


# ============================================================================
# Motor Fixtures
# ============================================================================

@pytest.fixture
def mock_motor_controller(mock_gpio):
    """Create a mock motor controller."""
    from src.control.motor_controller import MotorController

    controller = MotorController(enable_cleanup_handlers=False)
    return controller


@pytest.fixture
def initialized_motor_controller(mock_motor_controller):
    """Create an initialized motor controller."""
    mock_motor_controller.initialize()
    return mock_motor_controller


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_sensor_readings():
    """Provide sample sensor readings for testing."""
    return {
        "US_FRONT": 100.0,
        "US_FRONT_WP": 105.0,
        "US_RIGHT": 80.0,
        "US_LEFT": 75.0,
        "US_REAR": 200.0,
    }


@pytest.fixture
def sample_feature_vector():
    """Provide a sample feature vector for ML testing."""
    return {
        "front_distance": 100.0,
        "front_wp_distance": 105.0,
        "right_distance": 80.0,
        "left_distance": 75.0,
        "rear_distance": 200.0,
        "front_min_distance": 100.0,
        "camera_object_detected": 0,
        "camera_object_class": 0,
        "camera_object_distance_estimate": 0.0,
        "camera_object_position": 1,
        "current_speed": 50,
        "heading_error": 0.0,
    }


@pytest.fixture
def critical_zone_readings():
    """Sensor readings in critical zone (< 20cm front)."""
    return {
        "US_FRONT": 15.0,
        "US_FRONT_WP": 18.0,
        "US_RIGHT": 80.0,
        "US_LEFT": 75.0,
        "US_REAR": 200.0,
    }


@pytest.fixture
def danger_zone_readings():
    """Sensor readings in danger zone (20-50cm front)."""
    return {
        "US_FRONT": 35.0,
        "US_FRONT_WP": 40.0,
        "US_RIGHT": 100.0,
        "US_LEFT": 40.0,
        "US_REAR": 200.0,
    }


@pytest.fixture
def clear_zone_readings():
    """Sensor readings in clear zone (> 100cm front)."""
    return {
        "US_FRONT": 150.0,
        "US_FRONT_WP": 155.0,
        "US_RIGHT": 120.0,
        "US_LEFT": 130.0,
        "US_REAR": 200.0,
    }


# ============================================================================
# Camera/Detection Fixtures
# ============================================================================

@pytest.fixture
def sample_detection():
    """Sample object detection result."""
    return {
        "class_id": 0,  # Person
        "class_name": "person",
        "confidence": 0.85,
        "bbox": [100, 150, 200, 350],  # [x1, y1, x2, y2]
        "center": (150, 250),
        "position": "center",  # left/center/right
    }


@pytest.fixture
def empty_detection():
    """Empty detection result (no objects)."""
    return {
        "class_id": None,
        "class_name": None,
        "confidence": 0.0,
        "bbox": None,
        "center": None,
        "position": None,
    }


# ============================================================================
# Action Fixtures
# ============================================================================

@pytest.fixture
def action_names():
    """List of valid action names."""
    return [
        "FORWARD",
        "SLOW_DOWN",
        "TURN_LEFT",
        "TURN_RIGHT",
        "STOP",
        "REVERSE_LEFT",
        "REVERSE_RIGHT",
    ]


# ============================================================================
# Utility Functions
# ============================================================================

def create_distance_readings(
    front: float = 100.0,
    front_wp: float = 100.0,
    right: float = 100.0,
    left: float = 100.0,
    rear: float = 100.0,
) -> Dict[str, float]:
    """Helper to create sensor readings dict."""
    return {
        "US_FRONT": front,
        "US_FRONT_WP": front_wp,
        "US_RIGHT": right,
        "US_LEFT": left,
        "US_REAR": rear,
    }
