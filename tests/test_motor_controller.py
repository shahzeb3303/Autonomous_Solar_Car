"""
Unit Tests for Motor Controller Module.

Tests cover:
- Motor controller initialization
- Direction and speed control
- All movement commands (forward, reverse, turns)
- Emergency stop functionality
- PWM configuration
- Thread safety

Run with: pytest tests/test_motor_controller.py -v
"""

import pytest
import time
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestMotorControllerInit:
    """Tests for motor controller initialization."""

    def test_controller_creation(self, mock_gpio):
        """Test motor controller object creation."""
        from src.control.motor_controller import MotorController

        controller = MotorController(enable_cleanup_handlers=False)

        assert not controller.is_initialized
        assert controller.current_speed == 0
        assert not controller.is_moving

    def test_controller_initialization(self, mock_gpio):
        """Test motor controller GPIO initialization."""
        from src.control.motor_controller import MotorController

        controller = MotorController(enable_cleanup_handlers=False)
        result = controller.initialize()

        assert result is True
        assert controller.is_initialized

    def test_double_initialization_safe(self, mock_gpio):
        """Test that double initialization is safe."""
        from src.control.motor_controller import MotorController

        controller = MotorController(enable_cleanup_handlers=False)

        controller.initialize()
        result = controller.initialize()  # Second call

        assert result is True
        assert controller.is_initialized


class TestMotorDirection:
    """Tests for motor direction control."""

    def test_direction_enum_values(self):
        """Test MotorDirection enum values."""
        from src.control.motor_controller import MotorDirection

        assert MotorDirection.FORWARD.value == "forward"
        assert MotorDirection.REVERSE.value == "reverse"
        assert MotorDirection.STOP.value == "stop"
        assert MotorDirection.BRAKE.value == "brake"

    def test_motor_state_dataclass(self):
        """Test MotorState dataclass."""
        from src.control.motor_controller import MotorState, MotorDirection

        state = MotorState(
            direction=MotorDirection.FORWARD,
            speed=50,
            pwm_active=True,
        )

        assert state.direction == MotorDirection.FORWARD
        assert state.speed == 50
        assert state.pwm_active is True


class TestMotorMovement:
    """Tests for motor movement commands."""

    def test_forward_command(self, initialized_motor_controller):
        """Test forward movement command."""
        from src.control.motor_controller import MotorDirection

        controller = initialized_motor_controller
        controller.forward(speed=50)

        assert controller.left_state.direction == MotorDirection.FORWARD
        assert controller.right_state.direction == MotorDirection.FORWARD
        assert controller.left_state.speed == 50
        assert controller.right_state.speed == 50
        assert controller.is_moving

    def test_forward_default_speed(self, initialized_motor_controller, test_settings):
        """Test forward with default speed."""
        controller = initialized_motor_controller
        controller.forward()

        assert controller.left_state.speed == test_settings.SPEED_NORMAL

    def test_reverse_command(self, initialized_motor_controller):
        """Test reverse movement command."""
        from src.control.motor_controller import MotorDirection

        controller = initialized_motor_controller
        controller.reverse(speed=30)

        assert controller.left_state.direction == MotorDirection.REVERSE
        assert controller.right_state.direction == MotorDirection.REVERSE
        assert controller.left_state.speed == 30
        assert controller.right_state.speed == 30

    def test_turn_left_command(self, initialized_motor_controller, test_settings):
        """Test left turn command."""
        from src.control.motor_controller import MotorDirection

        controller = initialized_motor_controller
        controller.turn_left(speed=50)

        # Left wheel should be slower (inner wheel)
        assert controller.left_state.speed == test_settings.TURN_SPEED_INNER
        assert controller.right_state.speed == 50
        assert controller.left_state.direction == MotorDirection.FORWARD
        assert controller.right_state.direction == MotorDirection.FORWARD

    def test_turn_right_command(self, initialized_motor_controller, test_settings):
        """Test right turn command."""
        from src.control.motor_controller import MotorDirection

        controller = initialized_motor_controller
        controller.turn_right(speed=50)

        # Right wheel should be slower (inner wheel)
        assert controller.left_state.speed == 50
        assert controller.right_state.speed == test_settings.TURN_SPEED_INNER

    def test_pivot_left_command(self, initialized_motor_controller):
        """Test pivot left (spin in place) command."""
        from src.control.motor_controller import MotorDirection

        controller = initialized_motor_controller
        controller.pivot_left(speed=30)

        # Left wheel reverse, right wheel forward
        assert controller.left_state.direction == MotorDirection.REVERSE
        assert controller.right_state.direction == MotorDirection.FORWARD

    def test_pivot_right_command(self, initialized_motor_controller):
        """Test pivot right (spin in place) command."""
        from src.control.motor_controller import MotorDirection

        controller = initialized_motor_controller
        controller.pivot_right(speed=30)

        # Right wheel reverse, left wheel forward
        assert controller.left_state.direction == MotorDirection.FORWARD
        assert controller.right_state.direction == MotorDirection.REVERSE

    def test_reverse_left_command(self, initialized_motor_controller):
        """Test reverse while turning left."""
        from src.control.motor_controller import MotorDirection

        controller = initialized_motor_controller
        controller.reverse_left(speed=30)

        assert controller.left_state.direction == MotorDirection.REVERSE
        assert controller.right_state.direction == MotorDirection.REVERSE
        # Inner (left) wheel should be slower
        assert controller.left_state.speed < controller.right_state.speed

    def test_reverse_right_command(self, initialized_motor_controller):
        """Test reverse while turning right."""
        from src.control.motor_controller import MotorDirection

        controller = initialized_motor_controller
        controller.reverse_right(speed=30)

        assert controller.left_state.direction == MotorDirection.REVERSE
        assert controller.right_state.direction == MotorDirection.REVERSE
        # Inner (right) wheel should be slower
        assert controller.right_state.speed < controller.left_state.speed


class TestMotorStop:
    """Tests for stop functionality."""

    def test_stop_command(self, initialized_motor_controller):
        """Test stop command (coast)."""
        from src.control.motor_controller import MotorDirection

        controller = initialized_motor_controller

        # First move forward
        controller.forward(speed=50)
        assert controller.is_moving

        # Then stop
        controller.stop()

        assert controller.left_state.speed == 0
        assert controller.right_state.speed == 0
        assert controller.left_state.direction == MotorDirection.STOP
        assert controller.right_state.direction == MotorDirection.STOP
        assert not controller.is_moving

    def test_brake_command(self, initialized_motor_controller):
        """Test brake command (active braking)."""
        from src.control.motor_controller import MotorDirection

        controller = initialized_motor_controller

        controller.forward(speed=50)
        controller.brake()

        assert controller.left_state.speed == 0
        assert controller.right_state.speed == 0
        assert controller.left_state.direction == MotorDirection.BRAKE
        assert controller.right_state.direction == MotorDirection.BRAKE

    def test_emergency_stop(self, initialized_motor_controller):
        """Test emergency stop."""
        from src.control.motor_controller import MotorDirection

        controller = initialized_motor_controller

        controller.forward(speed=80)
        controller.emergency_stop()

        assert controller.left_state.speed == 0
        assert controller.right_state.speed == 0
        assert not controller.is_moving


class TestSpeedControl:
    """Tests for speed control."""

    def test_speed_clamping_high(self, initialized_motor_controller):
        """Test that speed is clamped to maximum."""
        controller = initialized_motor_controller

        controller.forward(speed=150)  # Above max

        assert controller.left_state.speed == 100  # Clamped to 100

    def test_speed_clamping_low(self, initialized_motor_controller):
        """Test that negative speed is clamped to zero."""
        controller = initialized_motor_controller

        controller.forward(speed=-10)  # Below min

        assert controller.left_state.speed == 0

    def test_set_speed(self, initialized_motor_controller):
        """Test set_speed method."""
        controller = initialized_motor_controller

        controller.forward(speed=50)
        controller.set_speed(70)

        assert controller.left_state.speed == 70
        assert controller.right_state.speed == 70

    def test_current_speed_property(self, initialized_motor_controller):
        """Test current_speed property (average of both motors)."""
        controller = initialized_motor_controller

        controller.set_motors(
            left_speed=40,
            right_speed=60,
        )

        assert controller.current_speed == 50  # Average


class TestSetMotors:
    """Tests for low-level set_motors method."""

    def test_set_motors_direct(self, initialized_motor_controller):
        """Test direct motor control."""
        from src.control.motor_controller import MotorDirection

        controller = initialized_motor_controller

        controller.set_motors(
            left_speed=40,
            right_speed=60,
            left_direction=MotorDirection.FORWARD,
            right_direction=MotorDirection.REVERSE,
        )

        assert controller.left_state.speed == 40
        assert controller.right_state.speed == 60
        assert controller.left_state.direction == MotorDirection.FORWARD
        assert controller.right_state.direction == MotorDirection.REVERSE

    def test_set_motors_without_initialization(self, mock_gpio):
        """Test that set_motors logs warning when not initialized."""
        from src.control.motor_controller import MotorController

        controller = MotorController(enable_cleanup_handlers=False)
        # Don't initialize

        # Should not crash, just warn
        controller.set_motors(left_speed=50, right_speed=50)

        assert controller.left_state.speed == 0  # Unchanged


class TestCleanup:
    """Tests for cleanup and resource management."""

    def test_cleanup(self, initialized_motor_controller, mock_gpio):
        """Test cleanup releases GPIO resources."""
        controller = initialized_motor_controller

        controller.cleanup()

        assert not controller.is_initialized

        # Verify cleanup was called
        call_log = mock_gpio.get_call_log()
        cleanup_calls = [c for c in call_log if c[0] == 'cleanup']
        assert len(cleanup_calls) > 0

    def test_cleanup_stops_motors_first(self, initialized_motor_controller):
        """Test that cleanup stops motors before releasing GPIO."""
        controller = initialized_motor_controller

        controller.forward(speed=70)
        controller.cleanup()

        # Motors should be stopped
        assert controller.left_state.speed == 0
        assert controller.right_state.speed == 0


class TestRampToSpeed:
    """Tests for smooth speed ramping."""

    def test_ramp_up(self, initialized_motor_controller, test_settings):
        """Test ramping up speed."""
        from src.control.motor_controller import MotorDirection

        controller = initialized_motor_controller

        # Start at 0, ramp to 50
        controller.ramp_to_speed(
            target_speed=50,
            direction=MotorDirection.FORWARD,
            step=10,
            delay=0.001,  # Fast for testing
        )

        assert controller.left_state.speed == 50
        assert controller.right_state.speed == 50

    def test_ramp_down(self, initialized_motor_controller):
        """Test ramping down speed."""
        from src.control.motor_controller import MotorDirection

        controller = initialized_motor_controller

        # Start at 80
        controller.forward(speed=80)

        # Ramp down to 30
        controller.ramp_to_speed(
            target_speed=30,
            direction=MotorDirection.FORWARD,
            step=10,
            delay=0.001,
        )

        assert controller.left_state.speed == 30


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_motor_commands(self, initialized_motor_controller):
        """Test thread safety of motor commands."""
        import threading

        controller = initialized_motor_controller
        errors = []

        def send_commands():
            try:
                for _ in range(10):
                    controller.forward(speed=50)
                    controller.turn_left()
                    controller.stop()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=send_commands) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
