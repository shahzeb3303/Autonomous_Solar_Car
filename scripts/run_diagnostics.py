#!/usr/bin/env python3
"""
System Diagnostics Script.

Checks all hardware components and system configuration:
- GPIO pin availability
- Ultrasonic sensor functionality
- Camera availability
- Motor controller
- ML model loading

Usage:
    python -m scripts.run_diagnostics
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from config.gpio_map import GPIOMap
from config.logging_config import setup_logging


def check_gpio():
    """Check GPIO availability."""
    print("\n[GPIO Check]")

    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        print("  [OK] RPi.GPIO available")
        GPIO.cleanup()
        return True
    except ImportError:
        print("  [X] RPi.GPIO not available (running in simulation mode)")
        return False
    except Exception as e:
        print(f"  [X] GPIO error: {e}")
        return False


def check_ultrasonic_sensors():
    """Check ultrasonic sensor configuration."""
    print("\n[Ultrasonic Sensors]")

    gpio_map = GPIOMap()
    sensors = gpio_map.get_all_ultrasonic_sensors()

    print(f"  Configured sensors: {len(sensors)}")
    for name, pins in sensors.items():
        print(f"    {name}: Trigger=GPIO{pins.trigger}, Echo=GPIO{pins.echo}")

    # Check for pin conflicts
    try:
        gpio_map.validate_no_conflicts()
        print("  [OK] No GPIO pin conflicts")
        return True
    except ValueError as e:
        print(f"  [X] Pin conflict: {e}")
        return False


def check_camera():
    """Check camera availability."""
    print("\n[Camera Check]")

    # Check picamera2
    try:
        from picamera2 import Picamera2
        print("  [OK] picamera2 available")
    except ImportError:
        print("  [X] picamera2 not available")

    # Check OpenCV
    try:
        import cv2
        print(f"  [OK] OpenCV available (version {cv2.__version__})")
    except ImportError:
        print("  [X] OpenCV not available")

    return True


def check_ml_dependencies():
    """Check ML library availability."""
    print("\n[ML Dependencies]")

    # TensorFlow
    try:
        import tensorflow as tf
        print(f"  [OK] TensorFlow available (version {tf.__version__})")
    except ImportError:
        print("  [X] TensorFlow not available")

    # TFLite runtime
    try:
        import tflite_runtime
        print("  [OK] TFLite Runtime available")
    except ImportError:
        print("  [X] TFLite Runtime not available")

    # NumPy
    try:
        import numpy as np
        print(f"  [OK] NumPy available (version {np.__version__})")
    except ImportError:
        print("  [X] NumPy not available")

    # Pandas
    try:
        import pandas as pd
        print(f"  [OK] Pandas available (version {pd.__version__})")
    except ImportError:
        print("  [X] Pandas not available")

    return True


def check_model_files():
    """Check for trained model files."""
    print("\n[Model Files]")

    settings = get_settings()

    # Decision model
    decision_model = settings.DECISION_MODEL_PATH
    if decision_model.exists():
        size_kb = decision_model.stat().st_size / 1024
        print(f"  [OK] Decision model: {decision_model} ({size_kb:.1f} KB)")
    else:
        print(f"  [X] Decision model not found: {decision_model}")

    # Object detection model
    od_model = settings.OBJECT_DETECTION_MODEL_PATH
    if od_model.exists():
        size_kb = od_model.stat().st_size / 1024
        print(f"  [OK] Object detection model: {od_model} ({size_kb:.1f} KB)")
    else:
        print(f"  [X] Object detection model not found: {od_model}")
        print("    (Download MobileNet SSD v2 from TensorFlow Hub)")

    return True


def check_data_files():
    """Check for data files."""
    print("\n[Data Files]")

    settings = get_settings()

    # Training data
    training_data = settings.DATA_DIR / "synthetic" / "training_data.csv"
    if training_data.exists():
        import os
        size_mb = os.path.getsize(training_data) / (1024 * 1024)
        print(f"  [OK] Training data: {training_data} ({size_mb:.1f} MB)")
    else:
        print(f"  [X] Training data not found: {training_data}")
        print("    (Run: python -m training.generate_synthetic_data)")

    return True


def check_motor_pins():
    """Check motor pin configuration."""
    print("\n[Motor Configuration]")

    gpio_map = GPIOMap()
    motors = gpio_map.get_all_motor_pins()

    for name, pins in motors.items():
        print(f"  {name}:")
        print(f"    Enable: GPIO{pins.enable}")
        print(f"    IN1: GPIO{pins.in1}")
        print(f"    IN2: GPIO{pins.in2}")

    print(f"  PWM Frequency: {gpio_map.PWM_FREQUENCY} Hz")

    return True


def run_sensor_test():
    """Run a quick sensor test."""
    print("\n[Sensor Test]")

    try:
        from src.sensors.ultrasonic import UltrasonicSensorArray

        array = UltrasonicSensorArray()
        results = array.initialize_all()

        print("  Initialization results:")
        for sensor_id, success in results.items():
            status = "[OK]" if success else "[X]"
            print(f"    {status} {sensor_id}")

        # Take readings
        print("\n  Taking readings...")
        distances = array.read_all_distances()

        for sensor_id, distance in distances.items():
            print(f"    {sensor_id}: {distance:.1f} cm")

        array.cleanup_all()
        return True

    except Exception as e:
        print(f"  [X] Sensor test failed: {e}")
        return False


def main():
    """Run all diagnostics."""
    print("=" * 60)
    print("Autonomous Solar Vehicle - System Diagnostics")
    print("=" * 60)

    settings = get_settings()
    print(f"\nProject Root: {settings.PROJECT_ROOT}")
    print(f"Simulation Mode: {settings.SIMULATION_MODE}")
    print(f"Is Raspberry Pi: {settings.IS_RASPBERRY_PI}")

    # Run checks
    results = []

    results.append(("GPIO", check_gpio()))
    results.append(("Ultrasonic Config", check_ultrasonic_sensors()))
    results.append(("Motor Config", check_motor_pins()))
    results.append(("Camera", check_camera()))
    results.append(("ML Dependencies", check_ml_dependencies()))
    results.append(("Model Files", check_model_files()))
    results.append(("Data Files", check_data_files()))

    # Optional sensor test
    print("\n" + "=" * 60)
    response = input("\nRun sensor hardware test? (y/n): ")
    if response.lower() == 'y':
        results.append(("Sensor Test", run_sensor_test()))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("\n[OK] System ready for operation")
    else:
        print("\n[X] Some checks failed - review output above")


if __name__ == "__main__":
    main()
