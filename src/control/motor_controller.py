"""
Motor Controller for L298N Motor Driver.

This module provides low-level control of DC motors via the L298N
dual H-bridge motor driver board.

Hardware Configuration:
- L298N board powered by 12V battery (with 5V regulator enabled)
- Two motor channels: Left (both left motors) and Right (both right motors)
- PWM speed control via ENA/ENB pins
- Direction control via IN1-IN4 pins

Direction Control Logic:
    | IN1 | IN2 | Motor State    |
    |-----|-----|----------------|
    |  L  |  L  | Coast/Stop     |
    |  H  |  L  | Forward        |
    |  L  |  H  | Reverse        |
    |  H  |  H  | Brake/Stop     |

Usage:
    from src.control.motor_controller import MotorController

    controller = MotorController()
    controller.initialize()

    controller.forward(speed=50)  # Move forward at 50% speed
    controller.turn_left(speed=40)  # Turn left
    controller.stop()  # Stop all motors

    controller.cleanup()  # Clean up GPIO on exit
"""

import time
import atexit
import signal
import threading
from typing import Optional, Dict, Tuple
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
from config.gpio_map import GPIOMap, MotorPins

logger = logging.getLogger(__name__)


class MotorDirection(Enum):
    """Motor rotation direction."""
    FORWARD = "forward"
    REVERSE = "reverse"
    STOP = "stop"
    BRAKE = "brake"


@dataclass
class MotorState:
    """Current state of a motor channel."""
    direction: MotorDirection
    speed: int  # PWM duty cycle (0-100)
    pwm_active: bool


class MockPWM:
    """Mock PWM class for development on non-RPi systems."""

    def __init__(self, pin: int, frequency: int):
        self.pin = pin
        self.frequency = frequency
        self.duty_cycle = 0
        self._running = False
        logger.debug(f"MockPWM created: pin={pin}, freq={frequency}")

    def start(self, duty_cycle: int) -> None:
        self.duty_cycle = duty_cycle
        self._running = True
        logger.debug(f"MockPWM pin {self.pin}: start at {duty_cycle}%")

    def ChangeDutyCycle(self, duty_cycle: int) -> None:
        self.duty_cycle = duty_cycle
        logger.debug(f"MockPWM pin {self.pin}: duty cycle -> {duty_cycle}%")

    def ChangeFrequency(self, frequency: int) -> None:
        self.frequency = frequency

    def stop(self) -> None:
        self._running = False
        logger.debug(f"MockPWM pin {self.pin}: stopped")


class MockGPIO:
    """Mock GPIO interface for development/testing."""

    BCM = "BCM"
    OUT = "OUT"
    IN = "IN"
    HIGH = True
    LOW = False

    _pin_states: Dict[int, bool] = {}

    @classmethod
    def setmode(cls, mode: str) -> None:
        logger.debug(f"MockGPIO: setmode({mode})")

    @classmethod
    def setwarnings(cls, state: bool) -> None:
        pass

    @classmethod
    def setup(cls, pin: int, direction: str) -> None:
        logger.debug(f"MockGPIO: setup(pin={pin}, dir={direction})")
        cls._pin_states[pin] = False

    @classmethod
    def output(cls, pin: int, state: bool) -> None:
        cls._pin_states[pin] = state
        logger.debug(f"MockGPIO: output(pin={pin}, state={state})")

    @classmethod
    def input(cls, pin: int) -> bool:
        return cls._pin_states.get(pin, False)

    @classmethod
    def cleanup(cls, pin: Optional[int] = None) -> None:
        logger.debug(f"MockGPIO: cleanup(pin={pin})")
        if pin is not None:
            cls._pin_states.pop(pin, None)
        else:
            cls._pin_states.clear()

    @classmethod
    def PWM(cls, pin: int, frequency: int) -> MockPWM:
        return MockPWM(pin, frequency)


class MotorController:
    """
    Low-level motor controller for L298N dual H-bridge driver.

    Provides direct control of left and right motor channels with:
    - PWM speed control (0-100%)
    - Direction control (forward/reverse)
    - Emergency stop capability
    - Automatic GPIO cleanup on exit
    - Thread-safe operation

    The controller uses differential steering - turning is achieved
    by running left and right motors at different speeds.

    Args:
        enable_cleanup_handlers: Register atexit and signal handlers
                                for automatic cleanup (default True)
    """

    def __init__(self, enable_cleanup_handlers: bool = True):
        """Initialize the motor controller."""
        self._settings = get_settings()
        self._gpio_map = GPIOMap()

        # Use mock GPIO if real GPIO not available
        self._gpio = GPIO if GPIO_AVAILABLE else MockGPIO

        # Motor pin configurations
        self._left_pins = self._gpio_map.MOTOR_LEFT
        self._right_pins = self._gpio_map.MOTOR_RIGHT

        # PWM objects (created during initialization)
        self._left_pwm: Optional[MockPWM] = None
        self._right_pwm: Optional[MockPWM] = None

        # Current motor states
        self._left_state = MotorState(
            direction=MotorDirection.STOP,
            speed=0,
            pwm_active=False
        )
        self._right_state = MotorState(
            direction=MotorDirection.STOP,
            speed=0,
            pwm_active=False
        )

        self._initialized = False
        self._lock = threading.Lock()

        # Register cleanup handlers
        if enable_cleanup_handlers:
            atexit.register(self.cleanup)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("MotorController created")

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle termination signals for clean shutdown."""
        logger.info(f"Received signal {signum}, initiating emergency stop")
        self.emergency_stop()
        self.cleanup()

    def initialize(self) -> bool:
        """
        Initialize GPIO pins and PWM for motor control.

        Must be called before using motor control functions.

        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            with self._lock:
                if self._initialized:
                    return True

                # Set up GPIO mode
                try:
                    self._gpio.setmode(self._gpio.BCM)
                except Exception:
                    pass  # Mode may already be set

                self._gpio.setwarnings(False)

                # Set up left motor pins
                self._gpio.setup(self._left_pins.enable, self._gpio.OUT)
                self._gpio.setup(self._left_pins.in1, self._gpio.OUT)
                self._gpio.setup(self._left_pins.in2, self._gpio.OUT)

                # Set up right motor pins
                self._gpio.setup(self._right_pins.enable, self._gpio.OUT)
                self._gpio.setup(self._right_pins.in1, self._gpio.OUT)
                self._gpio.setup(self._right_pins.in2, self._gpio.OUT)

                # Initialize PWM on enable pins
                self._left_pwm = self._gpio.PWM(
                    self._left_pins.enable,
                    self._settings.PWM_FREQUENCY_HZ
                )
                self._right_pwm = self._gpio.PWM(
                    self._right_pins.enable,
                    self._settings.PWM_FREQUENCY_HZ
                )

                # Start PWM with 0% duty cycle (motors stopped)
                self._left_pwm.start(0)
                self._right_pwm.start(0)

                # Ensure motors are stopped
                self._set_motor_direction("left", MotorDirection.STOP)
                self._set_motor_direction("right", MotorDirection.STOP)

                self._initialized = True
                logger.info("MotorController initialized successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to initialize MotorController: {e}")
            return False

    def cleanup(self) -> None:
        """
        Clean up GPIO resources and stop motors.

        This is automatically called on program exit if cleanup
        handlers are enabled. Can also be called manually.
        """
        try:
            with self._lock:
                if not self._initialized:
                    return

                # Stop motors first
                self._emergency_stop_internal()

                # Stop PWM
                if self._left_pwm:
                    self._left_pwm.stop()
                if self._right_pwm:
                    self._right_pwm.stop()

                # Clean up GPIO
                self._gpio.cleanup(self._left_pins.enable)
                self._gpio.cleanup(self._left_pins.in1)
                self._gpio.cleanup(self._left_pins.in2)
                self._gpio.cleanup(self._right_pins.enable)
                self._gpio.cleanup(self._right_pins.in1)
                self._gpio.cleanup(self._right_pins.in2)

                self._initialized = False
                logger.info("MotorController cleaned up")

        except Exception as e:
            logger.error(f"Error during MotorController cleanup: {e}")

    def _set_motor_direction(self, motor: str, direction: MotorDirection) -> None:
        """
        Set direction for a single motor channel.

        Args:
            motor: "left" or "right"
            direction: Desired direction
        """
        pins = self._left_pins if motor == "left" else self._right_pins

        if direction == MotorDirection.FORWARD:
            self._gpio.output(pins.in1, self._gpio.HIGH)
            self._gpio.output(pins.in2, self._gpio.LOW)
        elif direction == MotorDirection.REVERSE:
            self._gpio.output(pins.in1, self._gpio.LOW)
            self._gpio.output(pins.in2, self._gpio.HIGH)
        elif direction == MotorDirection.BRAKE:
            self._gpio.output(pins.in1, self._gpio.HIGH)
            self._gpio.output(pins.in2, self._gpio.HIGH)
        else:  # STOP (coast)
            self._gpio.output(pins.in1, self._gpio.LOW)
            self._gpio.output(pins.in2, self._gpio.LOW)

    def _set_motor_speed(self, motor: str, speed: int) -> None:
        """
        Set speed for a single motor channel.

        Args:
            motor: "left" or "right"
            speed: PWM duty cycle (0-100)
        """
        speed = max(0, min(100, speed))  # Clamp to valid range

        pwm = self._left_pwm if motor == "left" else self._right_pwm
        if pwm:
            pwm.ChangeDutyCycle(speed)

    def set_motors(
        self,
        left_speed: int,
        right_speed: int,
        left_direction: MotorDirection = MotorDirection.FORWARD,
        right_direction: MotorDirection = MotorDirection.FORWARD,
    ) -> None:
        """
        Set both motors to specific speed and direction.

        This is the primary method for controlling motor movement.
        All other movement methods (forward, turn_left, etc.) use this.

        Args:
            left_speed: Left motor speed (0-100)
            right_speed: Right motor speed (0-100)
            left_direction: Left motor direction
            right_direction: Right motor direction
        """
        with self._lock:
            if not self._initialized:
                logger.warning("MotorController not initialized, cannot set motors")
                return

            # Set directions
            self._set_motor_direction("left", left_direction)
            self._set_motor_direction("right", right_direction)

            # Set speeds
            self._set_motor_speed("left", left_speed)
            self._set_motor_speed("right", right_speed)

            # Update state
            self._left_state = MotorState(
                direction=left_direction,
                speed=left_speed,
                pwm_active=left_speed > 0
            )
            self._right_state = MotorState(
                direction=right_direction,
                speed=right_speed,
                pwm_active=right_speed > 0
            )

            logger.debug(
                f"Motors set: L={left_speed}% {left_direction.value}, "
                f"R={right_speed}% {right_direction.value}"
            )

    def forward(self, speed: Optional[int] = None) -> None:
        """
        Move forward at the specified speed.

        Args:
            speed: Speed percentage (0-100). Defaults to SPEED_NORMAL.
        """
        if speed is None:
            speed = self._settings.SPEED_NORMAL

        self.set_motors(
            left_speed=speed,
            right_speed=speed,
            left_direction=MotorDirection.FORWARD,
            right_direction=MotorDirection.FORWARD,
        )
        logger.info(f"Moving forward at {speed}%")

    def reverse(self, speed: Optional[int] = None) -> None:
        """
        Move backward at the specified speed.

        Args:
            speed: Speed percentage (0-100). Defaults to SPEED_SLOW.
        """
        if speed is None:
            speed = self._settings.SPEED_SLOW

        self.set_motors(
            left_speed=speed,
            right_speed=speed,
            left_direction=MotorDirection.REVERSE,
            right_direction=MotorDirection.REVERSE,
        )
        logger.info(f"Moving reverse at {speed}%")

    def turn_left(self, speed: Optional[int] = None) -> None:
        """
        Turn left using differential steering.

        Left motor runs slower (or reverse), right motor runs faster.

        Args:
            speed: Outer wheel speed (0-100). Defaults to TURN_SPEED_OUTER.
        """
        if speed is None:
            speed = self._settings.TURN_SPEED_OUTER

        inner_speed = self._settings.TURN_SPEED_INNER

        self.set_motors(
            left_speed=inner_speed,
            right_speed=speed,
            left_direction=MotorDirection.FORWARD,
            right_direction=MotorDirection.FORWARD,
        )
        logger.info(f"Turning left: inner={inner_speed}%, outer={speed}%")

    def turn_right(self, speed: Optional[int] = None) -> None:
        """
        Turn right using differential steering.

        Right motor runs slower, left motor runs faster.

        Args:
            speed: Outer wheel speed (0-100). Defaults to TURN_SPEED_OUTER.
        """
        if speed is None:
            speed = self._settings.TURN_SPEED_OUTER

        inner_speed = self._settings.TURN_SPEED_INNER

        self.set_motors(
            left_speed=speed,
            right_speed=inner_speed,
            left_direction=MotorDirection.FORWARD,
            right_direction=MotorDirection.FORWARD,
        )
        logger.info(f"Turning right: outer={speed}%, inner={inner_speed}%")

    def pivot_left(self, speed: Optional[int] = None) -> None:
        """
        Pivot (spin) left in place.

        Left motor runs in reverse, right motor runs forward.

        Args:
            speed: Motor speed (0-100). Defaults to TURN_SPEED_INNER.
        """
        if speed is None:
            speed = self._settings.TURN_SPEED_INNER

        self.set_motors(
            left_speed=speed,
            right_speed=speed,
            left_direction=MotorDirection.REVERSE,
            right_direction=MotorDirection.FORWARD,
        )
        logger.info(f"Pivoting left at {speed}%")

    def pivot_right(self, speed: Optional[int] = None) -> None:
        """
        Pivot (spin) right in place.

        Right motor runs in reverse, left motor runs forward.

        Args:
            speed: Motor speed (0-100). Defaults to TURN_SPEED_INNER.
        """
        if speed is None:
            speed = self._settings.TURN_SPEED_INNER

        self.set_motors(
            left_speed=speed,
            right_speed=speed,
            left_direction=MotorDirection.FORWARD,
            right_direction=MotorDirection.REVERSE,
        )
        logger.info(f"Pivoting right at {speed}%")

    def reverse_left(self, speed: Optional[int] = None) -> None:
        """
        Reverse while turning left.

        Args:
            speed: Base speed (0-100). Defaults to SPEED_SLOW.
        """
        if speed is None:
            speed = self._settings.SPEED_SLOW

        inner_speed = speed // 2

        self.set_motors(
            left_speed=inner_speed,
            right_speed=speed,
            left_direction=MotorDirection.REVERSE,
            right_direction=MotorDirection.REVERSE,
        )
        logger.info(f"Reverse left: inner={inner_speed}%, outer={speed}%")

    def reverse_right(self, speed: Optional[int] = None) -> None:
        """
        Reverse while turning right.

        Args:
            speed: Base speed (0-100). Defaults to SPEED_SLOW.
        """
        if speed is None:
            speed = self._settings.SPEED_SLOW

        inner_speed = speed // 2

        self.set_motors(
            left_speed=speed,
            right_speed=inner_speed,
            left_direction=MotorDirection.REVERSE,
            right_direction=MotorDirection.REVERSE,
        )
        logger.info(f"Reverse right: outer={speed}%, inner={inner_speed}%")

    def stop(self) -> None:
        """
        Stop motors by coasting (no brake).

        Motors will gradually slow down due to friction.
        """
        self.set_motors(
            left_speed=0,
            right_speed=0,
            left_direction=MotorDirection.STOP,
            right_direction=MotorDirection.STOP,
        )
        logger.info("Motors stopped (coast)")

    def brake(self) -> None:
        """
        Stop motors with braking.

        Motors will stop quickly due to electrical braking.
        """
        self.set_motors(
            left_speed=0,
            right_speed=0,
            left_direction=MotorDirection.BRAKE,
            right_direction=MotorDirection.BRAKE,
        )
        logger.info("Motors stopped (brake)")

    def _emergency_stop_internal(self) -> None:
        """Internal emergency stop without logging (for cleanup)."""
        self._set_motor_direction("left", MotorDirection.STOP)
        self._set_motor_direction("right", MotorDirection.STOP)
        self._set_motor_speed("left", 0)
        self._set_motor_speed("right", 0)

    def emergency_stop(self) -> None:
        """
        Immediate emergency stop.

        This should be called in safety-critical situations.
        Bypasses normal locking for immediate response.
        """
        try:
            # Direct GPIO control for immediate stop
            if self._initialized:
                self._set_motor_direction("left", MotorDirection.STOP)
                self._set_motor_direction("right", MotorDirection.STOP)
                self._set_motor_speed("left", 0)
                self._set_motor_speed("right", 0)

            self._left_state = MotorState(MotorDirection.STOP, 0, False)
            self._right_state = MotorState(MotorDirection.STOP, 0, False)

            logger.warning("EMERGENCY STOP executed")

        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")

    def set_speed(self, speed: int) -> None:
        """
        Set speed for both motors while maintaining current direction.

        Args:
            speed: Speed percentage (0-100)
        """
        with self._lock:
            self._set_motor_speed("left", speed)
            self._set_motor_speed("right", speed)
            self._left_state.speed = speed
            self._right_state.speed = speed

    def ramp_to_speed(
        self,
        target_speed: int,
        direction: MotorDirection = MotorDirection.FORWARD,
        step: Optional[int] = None,
        delay: Optional[float] = None,
    ) -> None:
        """
        Gradually ramp speed to target for smooth acceleration.

        Args:
            target_speed: Target speed (0-100)
            direction: Motor direction
            step: Speed increment per step (default from settings)
            delay: Delay between steps in seconds (default from settings)
        """
        if step is None:
            step = self._settings.ACCELERATION_STEP
        if delay is None:
            delay = self._settings.ACCELERATION_DELAY_S

        current_speed = self._left_state.speed

        if target_speed > current_speed:
            # Accelerating
            for speed in range(current_speed, target_speed + 1, step):
                self.set_motors(speed, speed, direction, direction)
                time.sleep(delay)
        else:
            # Decelerating
            for speed in range(current_speed, target_speed - 1, -step):
                self.set_motors(speed, speed, direction, direction)
                time.sleep(delay)

        # Ensure we hit the exact target
        self.set_motors(target_speed, target_speed, direction, direction)

    @property
    def left_state(self) -> MotorState:
        """Get current left motor state."""
        return self._left_state

    @property
    def right_state(self) -> MotorState:
        """Get current right motor state."""
        return self._right_state

    @property
    def current_speed(self) -> int:
        """Get average current speed of both motors."""
        return (self._left_state.speed + self._right_state.speed) // 2

    @property
    def is_moving(self) -> bool:
        """Check if any motor is currently active."""
        return self._left_state.speed > 0 or self._right_state.speed > 0

    @property
    def is_initialized(self) -> bool:
        """Check if controller has been initialized."""
        return self._initialized
