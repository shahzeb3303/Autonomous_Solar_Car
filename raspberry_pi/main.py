#!/usr/bin/env python3
"""
Main Integration Script (Pi side)

Architecture:
    - Laptop sends commands via TCP (manual or ML-generated)
    - Pi reads ultrasonic sensors (via Arduino) + GPS
    - Pi runs safety_governor as a hard-override layer
    - Pi sends sensor + GPS status back to laptop (for ML inference)
"""

import time
import signal
from datetime import datetime

import config
from motor_controller import MotorController
from steering_controller import SteeringController
from obstacle_monitor import ObstacleMonitor
from remote_server import RemoteServer
from safety_governor import SafetyGovernor, SafetyViolation

try:
    from gps_reader import GPSReader
    GPS_ENABLED = True
except ImportError:
    GPS_ENABLED = False




class IMUAnchor:
    """Anchors the gyro-only IMU heading to absolute compass via GPS COG.

    Defensive rules (GPS COG is unreliable at low speed):
      - Only anchor when speed >= GPS_MIN_SPEED
      - Reject COG == 0.0 (default value when NMEA course field is empty)
      - Require COG differs from previous reading
      - Require N consistent COG readings before locking first anchor
    """

    def __init__(self):
        self.offset_deg = 0.0
        self.has_anchor = False
        self.GPS_MIN_SPEED = 0.4  # m/s
        self.SMOOTH_ALPHA = 0.05
        self._prev_cog = None
        self._cog_history = []             # rolling list of recent valid COGs
        self.MIN_CONSISTENT_READINGS = 3
        self.MAX_COG_VARIATION = 30.0      # degrees

    def _cog_is_trustworthy(self, gps_cog, gps_speed):
        if gps_cog is None or gps_speed is None:
            return False
        if gps_speed < self.GPS_MIN_SPEED:
            return False
        # 0.0 is the gps_reader default when NMEA course field is empty.
        if gps_cog == 0.0:
            return False
        # Reject duplicates (stuck reading).
        if self._prev_cog is not None and abs(gps_cog - self._prev_cog) < 0.001:
            return False
        return True

    def update(self, imu_raw_deg, gps_cog, gps_valid, gps_speed):
        if imu_raw_deg is None:
            return None
        if gps_valid and self._cog_is_trustworthy(gps_cog, gps_speed):
            self._prev_cog = gps_cog
            self._cog_history.append(gps_cog)
            if len(self._cog_history) > 10:
                self._cog_history.pop(0)
            desired = (gps_cog - imu_raw_deg + 540.0) % 360.0 - 180.0
            if not self.has_anchor:
                recent = self._cog_history[-self.MIN_CONSISTENT_READINGS:]
                if len(recent) >= self.MIN_CONSISTENT_READINGS:
                    spread = max(recent) - min(recent)
                    if spread <= self.MAX_COG_VARIATION:
                        self.offset_deg = desired
                        self.has_anchor = True
            else:
                diff = (desired - self.offset_deg + 540.0) % 360.0 - 180.0
                self.offset_deg += self.SMOOTH_ALPHA * diff
        return (imu_raw_deg + self.offset_deg) % 360.0


class VehicleController:
    def __init__(self):
        self.motor = MotorController()
        self.steering = SteeringController()
        self.sensors = ObstacleMonitor()
        self.server = RemoteServer()
        self.safety = SafetyGovernor()
        self.gps = GPSReader() if GPS_ENABLED else None
        self.imu_anchor = IMUAnchor()

        self.running = False
        self.current_drive = config.CMD_STOP
        self.current_steer = config.CMD_STEER_STOP
        self.current_speed = 0
        self.last_violation = SafetyViolation.NONE

    def initialize(self):
        print("=" * 60)
        print("VEHICLE CONTROL SYSTEM - Pi Side")
        print("=" * 60)
        try:
            print("[1/5] Motor controller...")
            self.motor.setup()
            print("[2/5] Steering controller...")
            self.steering.setup()
            print("[3/5] Obstacle monitor (Arduino)...")
            self.sensors.start_monitoring()
            print("[4/5] Remote server...")
            self.server.start_server()
            print("[5/5] GPS reader...")
            if self.gps:
                self.gps.start()
            print("=" * 60)
            print(f"READY. Waiting for laptop on port {config.SERVER_PORT}")
            print("=" * 60)
            return True
        except Exception as e:
            print(f"Init failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start(self):
        if not self.initialize():
            return
        self.running = True
        signal.signal(signal.SIGINT, self._sig_handler)
        signal.signal(signal.SIGTERM, self._sig_handler)
        self._control_loop()

    def _control_loop(self):
        loop_count = 0
        while self.running:
            loop_start = time.time()
            try:
                # 1. Read commands from laptop
                drive_cmd = self.server.get_latest_command()
                steer_cmd = self.server.get_latest_steer()
                requested_speed = self.server.get_latest_speed()
                cmd_timestamp = self.server.get_command_timestamp()

                # 2. Read sensors
                distances = self.sensors.get_all_distances()

                # 3. Safety governor (HARD override)
                decision = self.safety.check(distances, drive_cmd, cmd_timestamp)
                self.last_violation = decision.violation

                if decision.is_safe:
                    self.current_drive = drive_cmd
                    self.current_steer = steer_cmd
                    self.current_speed = requested_speed
                else:
                    self.current_drive = decision.override_drive
                    self.current_steer = decision.override_steer
                    self.current_speed = decision.override_speed

                # 4. Apply to motors
                self.motor.set_speed(self.current_drive, self.current_speed)
                self.steering.set_direction(self.current_steer)

                # 5. Build and send status (includes everything ML needs)
                status = self._build_status(distances)
                if self.server.is_connected():
                    self.server.send_status(status)

                # 6. Periodic console log
                loop_count += 1
                if loop_count % 10 == 0:
                    self._log(status)

                # 7. Maintain loop rate
                elapsed = time.time() - loop_start
                time.sleep(max(0, config.CONTROL_LOOP_INTERVAL - elapsed))

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[loop error] {e}")
                time.sleep(0.1)

    def _build_status(self, distances):
        status = {
            'current_command': self.current_drive,
            'current_steer': self.current_steer,
            'actual_speed': self.current_speed,
            'distances': distances,
            'min_distance_front': self._min(distances, ['FL', 'FR', 'FW']),
            'min_distance_back': self._min(distances, ['BC']),
            'min_distance_left': self._min(distances, ['LS']),
            'min_distance_right': self._min(distances, ['RS']),
            'safety_violation': self.last_violation.value,
            'connected': self.server.is_connected(),
        }
        if self.gps:
            fix = self.gps.get_fix()
            if fix and fix.valid:
                status['gps'] = {
                    'lat': fix.latitude, 'lon': fix.longitude,
                    'alt': fix.altitude, 'speed_mps': fix.speed_mps,
                    'heading_deg': fix.heading_deg,
                    'satellites': fix.satellites, 'hdop': fix.hdop,
                    'valid': True,
                }
            else:
                status['gps'] = {'valid': False}

        # IMU heading from Arduino MPU, anchored to absolute compass via GPS COG.
        try:
            imu = self.sensors.sensor_reader.get_imu()
            raw_imu = imu["heading"] if imu["valid"] else None
            gps_dict = status.get("gps", {}) or {}
            gps_cog = gps_dict.get("heading_deg") if gps_dict.get("valid") else None
            gps_speed = gps_dict.get("speed_mps")
            absolute = self.imu_anchor.update(
                raw_imu, gps_cog, gps_dict.get("valid", False), gps_speed
            )
            status["imu"] = {
                "heading_raw": raw_imu,
                "heading_anchored": absolute,
                "offset_deg": self.imu_anchor.offset_deg,
                "has_anchor": self.imu_anchor.has_anchor,
                "gyro_z": imu["gyro_z"] if imu["valid"] else 0.0,
                "valid": imu["valid"],
            }
            # Only override gps.heading_deg AFTER the IMU anchor is confirmed
            # by multiple consistent GPS COG readings. Before that, keep the
            # original GPS COG (or none) so the controller does not steer by
            # an unanchored gyro.
            if absolute is not None and self.imu_anchor.has_anchor:
                if "gps" not in status:
                    status["gps"] = {"valid": False}
                status["gps"]["heading_deg"] = absolute
                status["gps"]["imu_heading"] = absolute
                status["gps"]["cog"] = gps_cog
                status["gps"]["gyro_z"] = imu["gyro_z"]
        except Exception as e:
            import traceback
            print(f"[IMU patch error] {e}\n{traceback.format_exc()}")
        return status

    @staticmethod
    def _min(distances, sensors):
        vals = [distances.get(s, 0) for s in sensors
                if config.MIN_SENSOR_DISTANCE <= distances.get(s, 0) <= config.MAX_SENSOR_DISTANCE]
        return min(vals) if vals else 0.0

    def _log(self, status):
        ts = datetime.now().strftime("%H:%M:%S")
        conn = "C" if status['connected'] else "D"
        v = self.last_violation.value
        d = status.get('distances', {})
        print(f"[{ts}] {conn} | drive={self.current_drive:8s} "
              f"steer={self.current_steer:10s} spd={self.current_speed:3d}% | "
              f"FL={d.get('FL',0):5.1f} FR={d.get('FR',0):5.1f} FW={d.get('FW',0):5.1f} "
              f"BC={d.get('BC',0):5.1f} LS={d.get('LS',0):5.1f} RS={d.get('RS',0):5.1f} | "
              f"safety={v}")

    def _sig_handler(self, signum, frame):
        print("\nShutdown...")
        self.stop()

    def stop(self):
        if not self.running:
            return
        self.running = False
        print("Stopping motors...")
        try: self.motor.stop()
        except Exception: pass
        try: self.steering.stop()
        except Exception: pass
        try: self.server.stop_server()
        except Exception: pass
        try: self.sensors.stop_monitoring()
        except Exception: pass
        if self.gps:
            try: self.gps.stop()
            except Exception: pass
        try:
            self.motor.cleanup()
            self.steering.cleanup()
        except Exception: pass
        print("Shutdown complete.")


def main():
    c = VehicleController()
    try:
        c.start()
    finally:
        c.stop()


if __name__ == "__main__":
    main()
