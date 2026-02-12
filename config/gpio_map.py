"""
GPIO Pin Mapping for Autonomous Solar Vehicle.

This is the SINGLE SOURCE OF TRUTH for all GPIO pin assignments.
All hardware components must reference pins from this module.

Hardware Platform: Raspberry Pi 4 Model B
GPIO Mode: BCM (Broadcom SOC channel numbers)

Pin Layout Reference:
    - Ultrasonic sensors: 5× HC-SR04
    - Motor driver: L298N (dual H-bridge)
    - Status LEDs: Optional diagnostic indicators
    - Emergency stop: Physical kill switch input

IMPORTANT: Never hardcode GPIO numbers elsewhere in the codebase.
Always import from this module.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class UltrasonicPins:
    """
    GPIO pin assignments for a single HC-SR04 ultrasonic sensor.

    Attributes:
        trigger: GPIO pin for trigger pulse (OUTPUT)
        echo: GPIO pin for echo reception (INPUT)
    """
    trigger: int
    echo: int


@dataclass(frozen=True)
class MotorPins:
    """
    GPIO pin assignments for a motor channel on L298N driver.

    Attributes:
        enable: PWM pin for speed control (ENA or ENB)
        in1: Direction control pin 1
        in2: Direction control pin 2
    """
    enable: int
    in1: int
    in2: int


@dataclass(frozen=True)
class GPIOMap:
    """
    Complete GPIO pin mapping for the autonomous vehicle.

    This class provides a centralized, immutable configuration of all
    GPIO pin assignments. Use the class methods to get pin configurations
    for specific components.

    Usage:
        gpio = GPIOMap()
        front_sensor = gpio.ULTRASONIC_FRONT
        left_motor = gpio.MOTOR_LEFT
    """

    # =========================================================================
    # ULTRASONIC SENSORS (HC-SR04)
    # =========================================================================
    # Each sensor requires 2 pins: TRIGGER (output) and ECHO (input)
    # Trigger: Send 10μs HIGH pulse to initiate measurement
    # Echo: Measure HIGH pulse duration to calculate distance

    # Front center sensor - primary obstacle detection
    ULTRASONIC_FRONT: UltrasonicPins = UltrasonicPins(trigger=17, echo=27)

    # Front waterproof sensor - redundant front detection (splash-proof)
    ULTRASONIC_FRONT_WP: UltrasonicPins = UltrasonicPins(trigger=22, echo=23)

    # Right side sensor - right clearance monitoring
    ULTRASONIC_RIGHT: UltrasonicPins = UltrasonicPins(trigger=24, echo=25)

    # Left side sensor - left clearance monitoring
    ULTRASONIC_LEFT: UltrasonicPins = UltrasonicPins(trigger=5, echo=6)

    # Rear center sensor - reverse/backing detection
    ULTRASONIC_REAR: UltrasonicPins = UltrasonicPins(trigger=12, echo=16)

    # =========================================================================
    # MOTOR DRIVER (L298N)
    # =========================================================================
    # Differential steering: Left motors + Right motors controlled independently
    # Speed control via PWM on ENABLE pins
    # Direction control via IN1/IN2 logic:
    #   - IN1=HIGH, IN2=LOW  → Forward
    #   - IN1=LOW,  IN2=HIGH → Reverse
    #   - IN1=LOW,  IN2=LOW  → Coast/Stop
    #   - IN1=HIGH, IN2=HIGH → Brake

    # Left motor channel (controls both left-side motors)
    MOTOR_LEFT: MotorPins = MotorPins(enable=13, in1=19, in2=26)

    # Right motor channel (controls both right-side motors)
    MOTOR_RIGHT: MotorPins = MotorPins(enable=18, in1=20, in2=21)

    # =========================================================================
    # STATUS LEDs (Optional - for diagnostics)
    # =========================================================================
    # These pins can be connected to LEDs for visual system status
    LED_STATUS: int = 4       # General status (heartbeat)
    LED_OBSTACLE: int = 14    # Obstacle detected indicator
    LED_ERROR: int = 15       # Error/fault indicator

    # =========================================================================
    # EMERGENCY STOP
    # =========================================================================
    # Physical emergency stop button (normally closed, active LOW)
    # When pressed, immediately cuts power to motors
    EMERGENCY_STOP: int = 7

    # =========================================================================
    # PWM CONFIGURATION
    # =========================================================================
    # PWM frequency for motor speed control
    PWM_FREQUENCY: int = 1000  # Hz (1kHz is good for DC motors)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @classmethod
    def get_all_ultrasonic_sensors(cls) -> Dict[str, UltrasonicPins]:
        """
        Get all ultrasonic sensor pin configurations.

        Returns:
            Dictionary mapping sensor names to their pin configurations.
        """
        return {
            "US_FRONT": cls.ULTRASONIC_FRONT,
            "US_FRONT_WP": cls.ULTRASONIC_FRONT_WP,
            "US_RIGHT": cls.ULTRASONIC_RIGHT,
            "US_LEFT": cls.ULTRASONIC_LEFT,
            "US_REAR": cls.ULTRASONIC_REAR,
        }

    @classmethod
    def get_all_trigger_pins(cls) -> Tuple[int, ...]:
        """
        Get all ultrasonic trigger pins (for bulk GPIO setup).

        Returns:
            Tuple of all trigger GPIO pin numbers.
        """
        return (
            cls.ULTRASONIC_FRONT.trigger,
            cls.ULTRASONIC_FRONT_WP.trigger,
            cls.ULTRASONIC_RIGHT.trigger,
            cls.ULTRASONIC_LEFT.trigger,
            cls.ULTRASONIC_REAR.trigger,
        )

    @classmethod
    def get_all_echo_pins(cls) -> Tuple[int, ...]:
        """
        Get all ultrasonic echo pins (for bulk GPIO setup).

        Returns:
            Tuple of all echo GPIO pin numbers.
        """
        return (
            cls.ULTRASONIC_FRONT.echo,
            cls.ULTRASONIC_FRONT_WP.echo,
            cls.ULTRASONIC_RIGHT.echo,
            cls.ULTRASONIC_LEFT.echo,
            cls.ULTRASONIC_REAR.echo,
        )

    @classmethod
    def get_all_motor_pins(cls) -> Dict[str, MotorPins]:
        """
        Get all motor pin configurations.

        Returns:
            Dictionary mapping motor names to their pin configurations.
        """
        return {
            "MOTOR_LEFT": cls.MOTOR_LEFT,
            "MOTOR_RIGHT": cls.MOTOR_RIGHT,
        }

    @classmethod
    def get_all_output_pins(cls) -> Tuple[int, ...]:
        """
        Get all GPIO pins configured as outputs.

        Returns:
            Tuple of all output GPIO pin numbers.
        """
        outputs = list(cls.get_all_trigger_pins())
        for motor in cls.get_all_motor_pins().values():
            outputs.extend([motor.enable, motor.in1, motor.in2])
        outputs.extend([cls.LED_STATUS, cls.LED_OBSTACLE, cls.LED_ERROR])
        return tuple(outputs)

    @classmethod
    def get_all_input_pins(cls) -> Tuple[int, ...]:
        """
        Get all GPIO pins configured as inputs.

        Returns:
            Tuple of all input GPIO pin numbers.
        """
        inputs = list(cls.get_all_echo_pins())
        inputs.append(cls.EMERGENCY_STOP)
        return tuple(inputs)

    @classmethod
    def validate_no_conflicts(cls) -> bool:
        """
        Validate that no GPIO pins are used by multiple components.

        Returns:
            True if no conflicts, raises ValueError if conflicts found.

        Raises:
            ValueError: If any GPIO pin is assigned to multiple components.
        """
        all_pins = list(cls.get_all_output_pins()) + list(cls.get_all_input_pins())
        seen = set()
        duplicates = []

        for pin in all_pins:
            if pin in seen:
                duplicates.append(pin)
            seen.add(pin)

        if duplicates:
            raise ValueError(f"GPIO pin conflict detected! Pins used multiple times: {duplicates}")

        return True


# Validate pin configuration at module load time
GPIOMap.validate_no_conflicts()
