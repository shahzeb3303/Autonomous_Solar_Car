#!/usr/bin/env python3
"""
Obstacle Monitor
Monitors ultrasonic sensor distances and calculates safe speeds.
5 sensors from Arduino (FL, FR, BC, LS, RS) + 1 waterproof on Pi GPIO (FW) = 6 total.
"""

import time
import threading

from sensor_reader import UltrasonicSensorReader
import config

# Try to import waterproof sensor (only works on Pi with GPIO)
try:
    from waterproof_sensor import WaterproofSensor
    WATERPROOF_AVAILABLE = True
except (ImportError, RuntimeError):
    WATERPROOF_AVAILABLE = False


class ObstacleMonitor:
    """
    Monitors obstacles using 6 ultrasonic sensors and calculates safe speeds.
    """

    def __init__(self):
        # Arduino sensors (5 sensors: FL, FR, BC, LS, RS)
        self.sensor_reader = UltrasonicSensorReader(
            port=config.SERIAL_PORT,
            baudrate=config.SERIAL_BAUD
        )

        # Pi waterproof sensor (FW - front center)
        self.waterproof = None
        if WATERPROOF_AVAILABLE:
            try:
                self.waterproof = WaterproofSensor()
            except Exception:
                self.waterproof = None

        self.is_monitoring = False
        self.lock = threading.Lock()

    def start_monitoring(self):
        """Start sensor monitoring."""
        if self.is_monitoring:
            print("[ObstacleMonitor] Already monitoring")
            return

        # Connect to Arduino
        print("[ObstacleMonitor] Connecting to Arduino...")
        if not self.sensor_reader.connect():
            raise RuntimeError("Failed to connect to Arduino")

        # Start reading Arduino sensors
        self.sensor_reader.start_reading()

        # Start waterproof sensor if available
        if self.waterproof:
            try:
                self.waterproof.start_continuous_reading()
                print("[ObstacleMonitor] Waterproof sensor (FW) started")
            except Exception as e:
                print(f"[ObstacleMonitor] Waterproof sensor failed: {e}")
                self.waterproof = None

        self.is_monitoring = True
        sensor_count = 6 if self.waterproof else 5
        print(f"[ObstacleMonitor] Monitoring started ({sensor_count} sensors)")

    def stop_monitoring(self):
        """Stop sensor monitoring."""
        if not self.is_monitoring:
            return

        print("[ObstacleMonitor] Stopping monitoring...")
        self.sensor_reader.stop_reading()

        if self.waterproof:
            try:
                self.waterproof.stop_reading()
            except Exception:
                pass

        self.is_monitoring = False
        print("[ObstacleMonitor] Monitoring stopped")

    def get_all_distances(self):
        """
        Get distances from all 6 sensors.

        Returns:
            dict: {FL, FR, FW, BC, LS, RS} in cm
        """
        # Get 5 sensors from Arduino
        data = self.sensor_reader.get_latest_data()

        # Add waterproof sensor (FW) from Pi GPIO
        if self.waterproof:
            data['FW'] = self.waterproof.get_distance()
        elif 'FW' not in data:
            data['FW'] = 0.0

        return data

    def get_minimum_distance(self, direction):
        """
        Get minimum distance in specified direction.

        Args:
            direction: 'front', 'back', or 'all'

        Returns:
            float or None
        """
        all_distances = self.get_all_distances()

        if direction == 'front':
            sensors_to_check = ['FL', 'FR', 'FW']
        elif direction == 'back':
            sensors_to_check = ['BC']
        else:
            sensors_to_check = ['FL', 'FR', 'FW', 'BC', 'LS', 'RS']

        valid_distances = []
        for sensor in sensors_to_check:
            dist = all_distances.get(sensor, 0)
            if config.MIN_SENSOR_DISTANCE <= dist <= config.MAX_SENSOR_DISTANCE:
                valid_distances.append(dist)

        return min(valid_distances) if valid_distances else None

    def get_safe_speed(self, direction):
        """
        Calculate safe speed based on obstacle distance.

        Args:
            direction: 'FORWARD' or 'BACKWARD'

        Returns:
            int: Safe speed percentage (0-100)
        """
        if direction == config.CMD_FORWARD:
            sensor_direction = 'front'
            emergency_threshold = config.DISTANCE_EMERGENCY_FRONT
        elif direction == config.CMD_BACKWARD:
            sensor_direction = 'back'
            emergency_threshold = config.DISTANCE_EMERGENCY_BACK
        else:
            return config.SPEED_STOP

        min_distance = self.get_minimum_distance(sensor_direction)

        if min_distance is None or min_distance <= 0:
            return config.SPEED_FULL

        if min_distance >= config.DISTANCE_SAFE:
            return config.SPEED_FULL
        elif min_distance >= config.DISTANCE_WARNING:
            return config.SPEED_MODERATE
        elif min_distance >= config.DISTANCE_CRITICAL:
            return config.SPEED_SLOW
        elif min_distance >= emergency_threshold:
            return config.SPEED_SLOW
        else:
            return config.SPEED_STOP

    def get_alert_status(self, direction):
        """Get alert level based on obstacle distance."""
        if direction == config.CMD_FORWARD:
            sensor_direction = 'front'
            emergency_threshold = config.DISTANCE_EMERGENCY_FRONT
        elif direction == config.CMD_BACKWARD:
            sensor_direction = 'back'
            emergency_threshold = config.DISTANCE_EMERGENCY_BACK
        else:
            return config.ALERT_CLEAR

        min_distance = self.get_minimum_distance(sensor_direction)

        if min_distance is None or min_distance <= 0:
            return config.ALERT_CLEAR

        if min_distance >= config.DISTANCE_SAFE:
            return config.ALERT_CLEAR
        elif min_distance >= config.DISTANCE_WARNING:
            return config.ALERT_WARNING
        elif min_distance >= config.DISTANCE_CRITICAL:
            return config.ALERT_CRITICAL
        else:
            return config.ALERT_EMERGENCY


if __name__ == "__main__":
    print("Testing Obstacle Monitor...")

    om = ObstacleMonitor()

    try:
        om.start_monitoring()

        print("Monitoring for 20 seconds... Move objects near sensors.\n")

        for i in range(40):
            distances = om.get_all_distances()
            min_front = om.get_minimum_distance('front')
            min_back = om.get_minimum_distance('back')

            fl = distances.get('FL', 0)
            fr = distances.get('FR', 0)
            fw = distances.get('FW', 0)
            bc = distances.get('BC', 0)
            ls = distances.get('LS', 0)
            rs = distances.get('RS', 0)

            print(f"[{i+1:2d}] FL={fl:6.1f} FW={fw:6.1f} FR={fr:6.1f} | "
                  f"BC={bc:6.1f} | LS={ls:6.1f} RS={rs:6.1f} | "
                  f"minF={min_front or 0:.1f} minB={min_back or 0:.1f}")

            time.sleep(0.5)

        print("\nTest complete!")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        om.stop_monitoring()
