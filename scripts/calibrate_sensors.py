#!/usr/bin/env python3
"""
Sensor Calibration Script.

Interactive tool for calibrating ultrasonic sensors and camera.

Usage:
    python -m scripts.calibrate_sensors
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def calibrate_ultrasonic():
    """Calibrate ultrasonic sensors."""
    print("\n" + "=" * 50)
    print("Ultrasonic Sensor Calibration")
    print("=" * 50)

    from src.sensors.ultrasonic import UltrasonicSensorArray

    array = UltrasonicSensorArray()
    array.initialize_all()

    print("\nPlace an object at KNOWN distances and record readings.")
    print("Press Ctrl+C when done.\n")

    try:
        while True:
            print("\n--- Current Readings ---")
            distances = array.read_all_distances()

            for sensor_id, distance in distances.items():
                print(f"  {sensor_id}: {distance:7.1f} cm")

            # Front redundancy check
            front = distances.get("US_FRONT", 0)
            front_wp = distances.get("US_FRONT_WP", 0)
            diff = abs(front - front_wp)
            print(f"\n  Front sensor difference: {diff:.1f} cm")

            if diff > 20:
                print("  ⚠ Large difference - check sensor alignment")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nCalibration session ended.")

    finally:
        array.cleanup_all()


def calibrate_camera_distance():
    """Calibrate camera distance estimation."""
    print("\n" + "=" * 50)
    print("Camera Distance Calibration")
    print("=" * 50)

    from src.sensors.camera import CameraManager
    from src.perception.object_detector import ObjectDetector
    from src.perception.distance_estimator import DistanceEstimator

    camera = CameraManager()
    detector = ObjectDetector()
    estimator = DistanceEstimator()

    camera.start()
    detector.load_model()

    print("\nHold a known object (e.g., person) at known distances.")
    print("Record bbox height and actual distance for calibration.")
    print("Press Ctrl+C when done.\n")

    try:
        while True:
            frame = camera.get_front_frame()
            if frame is None:
                continue

            detections = detector.detect(frame)

            print("\n--- Detected Objects ---")
            for det in detections:
                est = estimator.estimate_distance(det)
                print(f"  {det.class_name}: bbox_h={det.height}px, "
                      f"est_dist={est.distance_cm:.0f}cm")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nCalibration session ended.")

    finally:
        camera.stop()


def main():
    """Main calibration menu."""
    print("=" * 60)
    print("Sensor Calibration Tool")
    print("=" * 60)

    while True:
        print("\nOptions:")
        print("  1. Calibrate ultrasonic sensors")
        print("  2. Calibrate camera distance estimation")
        print("  3. Exit")

        choice = input("\nSelect option: ")

        if choice == "1":
            calibrate_ultrasonic()
        elif choice == "2":
            calibrate_camera_distance()
        elif choice == "3":
            break
        else:
            print("Invalid option")

    print("\nCalibration tool closed.")


if __name__ == "__main__":
    main()
