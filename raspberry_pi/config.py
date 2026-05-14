#!/usr/bin/env python3
"""
Configuration file for Laptop-Controlled Vehicle System
All configurable parameters in one place
"""

# ==============================================================================
# GPIO Pin Configuration (IBT_2 Drive Motor Driver)
# ==============================================================================
MOTOR_RPWM = 4       # GPIO 4 (Physical pin 7) - Forward PWM
MOTOR_LPWM = 17      # GPIO 17 (Physical pin 11) - Backward PWM
MOTOR_R_EN = 18      # GPIO 18 (Physical pin 12) - Right Enable
MOTOR_L_EN = 27      # GPIO 27 (Physical pin 13) - Left Enable
PWM_FREQUENCY = 1000 # Hz

# Set True if your motor is wired such that activating RPWM moves the car
# physically backward (i.e. wiring is reversed). The motor controller swaps
# RPWM/LPWM internally so callers ALWAYS use semantic FORWARD/BACKWARD.
# This is the single source of truth for direction inversion — no other file
# should compensate for wiring.
MOTOR_INVERTED = True

# ==============================================================================
# GPIO Pin Configuration (IBT_2 Steering Motor Driver)
# ==============================================================================
STEER_RPWM = 22      # GPIO 22 (Physical pin 15) - Steer Right PWM
STEER_LPWM = 23      # GPIO 23 (Physical pin 16) - Steer Left PWM
STEER_R_EN = 24      # GPIO 24 (Physical pin 18) - Right Enable
STEER_L_EN = 25      # GPIO 25 (Physical pin 22) - Left Enable
STEER_SPEED = 70     # Default steering speed percentage

# ==============================================================================
# Serial Configuration (Arduino Communication)
# ==============================================================================
import glob as _glob
def _find_arduino():
    """Auto-detect Arduino serial port"""
    for pattern in ['/dev/ttyUSB*', '/dev/ttyACM*']:
        ports = _glob.glob(pattern)
        if ports:
            return sorted(ports)[0]
    return '/dev/ttyUSB0'

SERIAL_PORT = _find_arduino()  # Auto-detect Arduino serial port
SERIAL_BAUD = 115200          # Baud (matches sensors_with_imu.ino)
                              # Change to 115200 if you upload sensor_production.ino (JSON)
SERIAL_TIMEOUT = 0.05         # Timeout in seconds (50ms for fast response)

# ==============================================================================
# Network Configuration (Laptop Communication)
# ==============================================================================
SERVER_HOST = '0.0.0.0'  # Listen on all network interfaces
SERVER_PORT = 5555       # TCP port for laptop connection

# ==============================================================================
# Distance Thresholds (centimeters)
# ==============================================================================
DISTANCE_SAFE = 300            # >= 300cm: Full speed (100%)
DISTANCE_WARNING = 200         # 200-299cm: Reduced speed (60%)
DISTANCE_CRITICAL = 100        # 100-199cm: Very slow (30%)
DISTANCE_EMERGENCY_FRONT = 50  # < 50cm (front): STOP (0%)
DISTANCE_EMERGENCY_BACK = 80   # < 80cm (back): STOP (0%)

# ==============================================================================
# Speed Settings (percentage 0-100)
# ==============================================================================
SPEED_FULL = 100         # Full speed (no obstacles)
SPEED_MODERATE = 60      # Moderate speed (warning level)
SPEED_SLOW = 30          # Slow speed (critical level)
SPEED_STOP = 0           # Stopped

# ==============================================================================
# Control Loop Timing
# ==============================================================================
CONTROL_LOOP_HZ = 20     # Main control loop frequency (Hz) - Optimal balance: fast response, stable
STATUS_UPDATE_HZ = 10    # Status update frequency (Hz)
CONTROL_LOOP_INTERVAL = 1.0 / CONTROL_LOOP_HZ  # Loop interval in seconds

# ==============================================================================
# Safety Settings
# ==============================================================================
WATCHDOG_TIMEOUT = 0.5   # Seconds - stop motor if no command received from laptop
                         # At indoor speed ~30 cm/s, 0.5s = 15cm of "blind" travel.
CONNECTION_TIMEOUT = 5.0 # Seconds - reconnect attempt interval
MAX_SENSOR_DISTANCE = 400.0  # Maximum valid sensor reading (cm)
MIN_SENSOR_DISTANCE = 2.0    # Minimum valid sensor reading (cm)
SENSOR_FAILURE_TIME = 1.0    # Seconds of all-zero sensors before declaring failure

# ==============================================================================
# Sensor Configuration
# ==============================================================================
# All sensors connected to Arduino (6 sensors total):
#   FL = Front Left
#   FR = Front Right
#   FW = Front Waterproof (center)
#   BC = Back Center
#   LS = Left Side
#   RS = Right Side

# Sensors by position
FRONT_SENSORS = ['FL', 'FR', 'FW']          # 3 front sensors
BACK_SENSORS = ['BC']                        # 1 back sensor
SIDE_SENSORS = ['LS', 'RS']                  # 2 side sensors

ALL_SENSORS = ['FL', 'FR', 'FW', 'BC', 'LS', 'RS']  # All 6 sensors from Arduino

# ==============================================================================
# Alert Levels
# ==============================================================================
ALERT_CLEAR = 'CLEAR'        # No obstacles (>= 300cm)
ALERT_WARNING = 'WARNING'    # Obstacle detected (100-299cm)
ALERT_CRITICAL = 'CRITICAL'  # Very close obstacle (30-99cm)
ALERT_EMERGENCY = 'EMERGENCY'  # Immediate danger (< 30cm)

# ==============================================================================
# Command Types
# ==============================================================================
CMD_FORWARD = 'FORWARD'
CMD_BACKWARD = 'BACKWARD'
CMD_LEFT = 'LEFT'
CMD_RIGHT = 'RIGHT'
CMD_STEER_STOP = 'STEER_STOP'
CMD_STOP = 'STOP'
CMD_EMERGENCY = 'EMERGENCY'
CMD_STATUS = 'STATUS'

CMD_AUTONOMOUS = 'AUTONOMOUS'

VALID_COMMANDS = [CMD_FORWARD, CMD_BACKWARD, CMD_LEFT, CMD_RIGHT, CMD_STEER_STOP, CMD_STOP, CMD_EMERGENCY, CMD_STATUS, CMD_AUTONOMOUS]

# ==============================================================================
# Autonomous Mode Settings
# ==============================================================================
AUTO_CRUISE_SPEED = 50           # Normal cruising speed (%)
AUTO_SLOW_SPEED = 30             # Reduced speed when obstacle detected (%)
AUTO_REVERSE_SPEED = 40          # Reverse speed (%)
AUTO_STEER_SPEED = 70            # Steering speed during avoidance (%)

# Distance thresholds for autonomous decisions (cm)
AUTO_DISTANCE_CRUISE = 200       # >= 200cm: cruise at full auto speed
AUTO_DISTANCE_SLOW = 100         # 100-200cm: slow down and prepare to avoid
AUTO_DISTANCE_AVOID = 70         # < 70cm: start avoidance maneuver (turn)
AUTO_DISTANCE_EMERGENCY = 40     # < 40cm: emergency stop (sudden obstacle)
AUTO_DISTANCE_REAR_SAFE = 80     # > 80cm behind: safe to reverse
AUTO_DISTANCE_SIDE_MIN = 50      # Minimum side clearance to turn that direction

# Timing for autonomous maneuvers (seconds)
AUTO_REVERSE_TIME = 1.5          # How long to reverse after emergency stop
AUTO_TURN_TIME = 1.5             # How long to turn after reversing
AUTO_WAIT_TIMEOUT = 10.0         # Max seconds to wait before retrying

# Autonomous states
AUTO_STATE_CRUISING = 'CRUISING'
AUTO_STATE_SLOWING = 'SLOWING'
AUTO_STATE_AVOIDING = 'AVOIDING'
AUTO_STATE_EMERGENCY = 'EMERGENCY_STOP'
AUTO_STATE_REVERSING = 'REVERSING'
AUTO_STATE_TURNING = 'TURNING'
AUTO_STATE_WAITING = 'WAITING'
