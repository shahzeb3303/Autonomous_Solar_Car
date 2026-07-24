#!/usr/bin/env python3
"""
Arduino Ultrasonic Sensor Reader
Reads JSON sensor data from Arduino via serial.

Arduino sends lines like:
    {"FL":123.4,"FR":234.5,"BC":345.6,"LS":456.7,"RS":567.8}

at 115200 baud, ~10Hz.
"""

import json
import re
import serial
import threading
import time

# Parser for the plain-text sensor_test / all_6_sensors sketch, e.g. "FL: 63.3 cm"
_TEXT_LINE_RE = re.compile(r'^([A-Z]{2})\s*:\s*(-?\d+\.?\d*)\s*cm', re.IGNORECASE)


class UltrasonicSensorReader:
    """
    Reads ultrasonic sensor data from Arduino over serial.
    Runs a background thread that continuously parses JSON lines.
    """

    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=0.05):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        # Default all sensors to 0
        self.data = {'FL': 0.0, 'FR': 0.0, 'FW': 0.0, 'BC': 0.0, 'LS': 0.0, 'RS': 0.0}
        # IMU heading from Arduino MPU
        self.imu = {'heading': 0.0, 'gyro_z': 0.0, 'valid': False}
        self.imu_last_update = 0.0
        # Per-sensor freshness tracking. If a sensor hasn't been updated in
        # STALE_S seconds we report MAX_CM ("clear / no object detected")
        # instead of letting the last reading persist forever.
        self.last_update = {k: 0.0 for k in self.data}
        self.STALE_S = 1.5
        self.MAX_CM = 400.0

    def connect(self) -> bool:
        """Open serial connection to Arduino."""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            # Clear any stale data in buffer
            time.sleep(0.5)
            self.ser.reset_input_buffer()
            print(f"[SensorReader] Connected to {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"[SensorReader] Failed to connect to {self.port}: {e}")
            return False

    def start_reading(self):
        """Start background thread to read sensor data."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print("[SensorReader] Reading started")

    def _read_loop(self):
        """Background thread: reads lines from serial, parses JSON."""
        while self.running:
            try:
                if self.ser is None or not self.ser.is_open:
                    time.sleep(0.1)
                    continue

                # Read a line from Arduino
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue

                # Try JSON first, then plain-text "XX: N.N cm" format
                if line.startswith('{'):
                    try:
                        parsed = json.loads(line)
                        now = time.time()
                        with self.lock:
                            for key in ['FL', 'FR', 'FW', 'BC', 'LS', 'RS']:
                                if key in parsed:
                                    self.data[key] = float(parsed[key])
                                    self.last_update[key] = now
                            if 'heading' in parsed:
                                self.imu['heading'] = float(parsed['heading'])
                                self.imu['gyro_z'] = float(parsed.get('gyro_z', 0.0))
                                self.imu['valid'] = True
                                self.imu_last_update = now
                    except (json.JSONDecodeError, ValueError):
                        pass
                else:
                    m = _TEXT_LINE_RE.match(line)
                    if m:
                        key = m.group(1).upper()
                        if key in ('FL', 'FR', 'FW', 'BC', 'LS', 'RS'):
                            with self.lock:
                                self.data[key] = float(m.group(2))
                                self.last_update[key] = time.time()

            except serial.SerialException:
                print("[SensorReader] Serial error, attempting reconnect...")
                time.sleep(1)
                try:
                    if self.ser:
                        self.ser.close()
                    self.ser = serial.Serial(
                        port=self.port,
                        baudrate=self.baudrate,
                        timeout=self.timeout
                    )
                    self.ser.reset_input_buffer()
                    print("[SensorReader] Reconnected")
                except Exception:
                    pass
            except Exception:
                time.sleep(0.05)

    def get_imu(self) -> dict:
        """Return latest IMU data (heading degrees, gyro_z dps, valid flag)."""
        now = time.time()
        with self.lock:
            valid = self.imu['valid'] and (now - self.imu_last_update) < 2.0
            return {'heading': self.imu['heading'],
                    'gyro_z':  self.imu['gyro_z'],
                    'valid':   valid}

    def get_latest_data(self) -> dict:
        """
        Get latest sensor readings.

        Any sensor whose value hasn't been refreshed in the last STALE_S
        seconds is reported as MAX_CM (treated as "clear / no object"),
        so a sensor that goes silent — e.g. because the Arduino dropped
        the field on a pulseIn timeout — does NOT keep returning its
        last value forever.

        Returns:
            dict with keys: FL, FR, FW, BC, LS, RS (float values in cm)
        """
        now = time.time()
        with self.lock:
            return {k: (self.data[k] if now - self.last_update[k] < self.STALE_S
                        else self.MAX_CM)
                    for k in self.data}

    def stop_reading(self):
        """Stop the background reading thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("[SensorReader] Stopped")


# Test: run directly to see live sensor data
if __name__ == "__main__":
    import sys
    import glob

    # Auto-detect port
    ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    port = ports[0] if ports else '/dev/ttyUSB0'
    print(f"Using port: {port}")

    reader = UltrasonicSensorReader(port=port, baudrate=115200)
    if not reader.connect():
        sys.exit(1)

    reader.start_reading()
    print("Reading sensors... (Ctrl+C to stop)\n")

    try:
        while True:
            data = reader.get_latest_data()
            print(f"\rFL={data['FL']:6.1f}  FR={data['FR']:6.1f}  FW={data['FW']:6.1f}  "
                  f"BC={data['BC']:6.1f}  LS={data['LS']:6.1f}  RS={data['RS']:6.1f}", end='', flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        reader.stop_reading()
