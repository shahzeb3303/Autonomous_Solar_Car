"""
Main Loop with Smart Navigation - Real-world driving behavior.

This is how a REAL autonomous car should behave:

1. Obstacle ahead, side clear → OVERTAKE (go around it)
2. Slow vehicle ahead → OVERTAKE (pass it)
3. Pedestrian crossing → WAIT (let them pass, then continue)
4. Path completely blocked → REPLAN (find new route)
5. Emergency (too close) → STOP then decide

Usage:
    python -m src.core.main_loop_smart --destination 300,400 --simulation
"""

import time
import signal
import atexit
import threading
import argparse
from typing import Optional, Dict, Any
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import get_settings
from config.logging_config import setup_logging

from src.sensors.ultrasonic import UltrasonicSensorArray
from src.sensors.camera import CameraManager
from src.sensors.sensor_fusion import SensorFusion
from src.perception.object_detector import ObjectDetector
from src.perception.distance_estimator import DistanceEstimator
from src.decision.safety_governor import SafetyGovernor
from src.decision.action_executor import ActionExecutor, Action
from src.control.motor_controller import MotorController
from src.navigation.smart_navigator import SmartNavigator, DrivingAction
from src.navigation.odometry import Odometry

logger = logging.getLogger(__name__)


class SmartAutonomousVehicle:
    """
    Autonomous vehicle with human-like driving behavior.

    Key behaviors:
    - OVERTAKES obstacles instead of just stopping
    - WAITS for pedestrians then continues
    - Only REPLANS when truly stuck
    - Smooth driving towards destination
    """

    def __init__(self, simulation_mode: bool = True):
        self._settings = get_settings()
        self._simulation_mode = simulation_mode

        # Sensors
        self._sensor_array = UltrasonicSensorArray()
        self._camera_manager = CameraManager()
        self._object_detector = ObjectDetector()
        self._distance_estimator = DistanceEstimator()

        self._sensor_fusion = SensorFusion(
            sensor_array=self._sensor_array,
            camera_manager=self._camera_manager,
            object_detector=self._object_detector,
            distance_estimator=self._distance_estimator,
        )

        # Safety (still highest priority)
        self._safety_governor = SafetyGovernor()

        # Motor control
        self._motor_controller = MotorController(enable_cleanup_handlers=True)
        self._action_executor = ActionExecutor(self._motor_controller)

        # Smart Navigation
        self._navigator = SmartNavigator(
            map_width_cm=3000,
            map_height_cm=3000,
            goal_threshold_cm=30
        )
        self._odometry = Odometry()

        # Runtime
        self._running = False
        self._loop_count = 0
        self._destination = None

        # Cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info("SmartAutonomousVehicle initialized")

    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received")
        self.stop()

    def _cleanup(self):
        if self._running:
            self.stop()

    def initialize(self) -> bool:
        """Initialize all systems."""
        logger.info("Initializing...")

        try:
            self._sensor_fusion.initialize()
            self._motor_controller.initialize()
            logger.info("Initialization complete")
            return True
        except Exception as e:
            logger.error(f"Init failed: {e}")
            return False

    def set_destination(self, x: float, y: float) -> bool:
        """Set destination and plan route."""
        self._destination = (x, y)
        success = self._navigator.set_destination(x, y)

        if success:
            print(f"\n[NAV] Destination set: ({x}, {y}) cm")
            print(f"[NAV] Route planned successfully")
        else:
            print(f"\n[NAV] ERROR: Cannot find route to ({x}, {y})")

        return success

    def start(self, blocking: bool = True):
        """Start autonomous driving."""
        if not self._motor_controller.is_initialized:
            self.initialize()

        self._running = True
        print("\n[DRIVING] Starting autonomous navigation...")
        print("[DRIVING] Press Ctrl+C to stop\n")

        if blocking:
            self._run_loop()

    def stop(self):
        """Stop the vehicle."""
        self._running = False
        self._motor_controller.emergency_stop()
        self._sensor_fusion.cleanup()
        logger.info("Vehicle stopped")

    def _run_loop(self):
        """Main control loop."""
        loop_period = 1.0 / 10  # 10 Hz

        while self._running:
            start_time = time.time()

            try:
                result = self._loop_iteration()

                if result == "ARRIVED":
                    print("\n" + "="*50)
                    print("[SUCCESS] DESTINATION REACHED!")
                    print("="*50)
                    self._print_statistics()
                    break

                if result == "FAILED":
                    print("\n" + "="*50)
                    print("[FAILED] Cannot reach destination")
                    print("="*50)
                    break

            except Exception as e:
                logger.error(f"Loop error: {e}")
                self._motor_controller.emergency_stop()
                break

            # Maintain loop rate
            elapsed = time.time() - start_time
            sleep_time = loop_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            self._loop_count += 1

    def _loop_iteration(self) -> str:
        """Single iteration - sense, think, act."""

        # ========================================
        # SENSE: Get sensor readings
        # ========================================
        fused_data = self._sensor_fusion.get_fused_data(
            current_speed=self._action_executor.current_speed,
            heading_error=0
        )

        # ========================================
        # UPDATE: Position tracking
        # ========================================
        current_action = self._action_executor.current_action
        action_name = current_action.name if current_action else "STOP"
        current_speed = self._action_executor.current_speed

        position = self._odometry.update(action_name, current_speed)
        self._navigator.update_position(position.x, position.y, position.heading)

        # ========================================
        # THINK: Smart navigator decides
        # ========================================
        # Check for person
        person_detected = fused_data.camera_class == 1
        person_distance = fused_data.camera_distance if person_detected else 999

        decision = self._navigator.decide(
            front_dist=fused_data.front_min_distance,
            front_left_dist=fused_data.front_min_distance * 0.9,  # Approximate
            front_right_dist=fused_data.front_min_distance * 0.9,
            left_dist=fused_data.left_distance,
            right_dist=fused_data.right_distance,
            rear_dist=fused_data.rear_distance,
            person_detected=person_detected,
            person_distance=person_distance
        )

        # ========================================
        # ACT: Execute the decision
        # ========================================
        motor_action = self._convert_to_motor_action(decision)
        self._action_executor.execute(motor_action.value)

        # ========================================
        # LOG: Show what's happening
        # ========================================
        if self._loop_count % 10 == 0:
            self._print_status(decision, position, fused_data)

        # Check for completion
        if decision.action == DrivingAction.ARRIVED:
            return "ARRIVED"

        if decision.action == DrivingAction.STOP and "no route" in decision.reason.lower():
            return "FAILED"

        return "CONTINUE"

    def _convert_to_motor_action(self, decision) -> Action:
        """Convert smart decision to motor action."""

        if decision.action == DrivingAction.CONTINUE:
            # Determine turn direction from angle
            if decision.turn_angle < -20:
                return Action.TURN_LEFT
            elif decision.turn_angle > 20:
                return Action.TURN_RIGHT
            else:
                return Action.FORWARD

        elif decision.action == DrivingAction.OVERTAKE_LEFT:
            return Action.TURN_LEFT

        elif decision.action == DrivingAction.OVERTAKE_RIGHT:
            return Action.TURN_RIGHT

        elif decision.action == DrivingAction.SLOW_DOWN:
            return Action.SLOW_DOWN

        elif decision.action in (DrivingAction.WAIT, DrivingAction.STOP, DrivingAction.ARRIVED):
            return Action.STOP

        elif decision.action == DrivingAction.REPLAN:
            # After replan, check turn angle
            if decision.turn_angle < -20:
                return Action.TURN_LEFT
            elif decision.turn_angle > 20:
                return Action.TURN_RIGHT
            return Action.FORWARD

        return Action.STOP

    def _print_status(self, decision, position, sensor_data):
        """Print current status."""
        stats = self._navigator.get_statistics()

        # Color codes for terminal
        action_colors = {
            DrivingAction.CONTINUE: "\033[92m",      # Green
            DrivingAction.OVERTAKE_LEFT: "\033[93m", # Yellow
            DrivingAction.OVERTAKE_RIGHT: "\033[93m",# Yellow
            DrivingAction.SLOW_DOWN: "\033[94m",     # Blue
            DrivingAction.WAIT: "\033[95m",          # Magenta
            DrivingAction.STOP: "\033[91m",          # Red
            DrivingAction.REPLAN: "\033[96m",        # Cyan
        }
        reset = "\033[0m"
        color = action_colors.get(decision.action, "")

        print(f"  Pos: ({position.x:6.0f}, {position.y:6.0f}) cm | "
              f"Head: {position.heading:5.0f} deg | "
              f"Front: {sensor_data.front_min_distance:4.0f} cm | "
              f"{color}{decision.action.name:15}{reset} | "
              f"Goal: {stats['distance_to_goal']:5.0f} cm | "
              f"OT: {stats['overtakes_completed']}")

    def _print_statistics(self):
        """Print final statistics."""
        stats = self._navigator.get_statistics()
        odom = self._odometry.get_statistics()

        print(f"\nFinal Statistics:")
        print(f"  Final Position:    ({stats['position'][0]:.0f}, {stats['position'][1]:.0f}) cm")
        print(f"  Total Distance:    {odom['total_distance_cm']:.0f} cm")
        print(f"  Overtakes Done:    {stats['overtakes_completed']}")
        print(f"  Route Replans:     {stats['replans_done']}")
        print(f"  Pedestrian Waits:  {stats['pedestrians_waited']}")


def main():
    parser = argparse.ArgumentParser(description="Smart Autonomous Vehicle")
    parser.add_argument("--destination", "-d", required=True,
                       help="Destination as 'x,y' in cm")
    parser.add_argument("--simulation", "-s", action="store_true",
                       help="Run in simulation mode")
    args = parser.parse_args()

    # Parse destination
    try:
        dest_x, dest_y = map(float, args.destination.split(','))
    except:
        print("ERROR: Use format --destination 'x,y' (e.g., '200,300')")
        sys.exit(1)

    setup_logging()

    print("="*60)
    print("Smart Autonomous Solar Vehicle")
    print("Capital University of Science & Technology")
    print("="*60)
    print(f"\nStart:       (0, 0)")
    print(f"Destination: ({dest_x}, {dest_y})")
    print(f"Mode:        {'Simulation' if args.simulation else 'Real Hardware'}")
    print()

    # Create and run
    vehicle = SmartAutonomousVehicle(simulation_mode=args.simulation)

    if not vehicle.initialize():
        print("ERROR: Initialization failed")
        sys.exit(1)

    if not vehicle.set_destination(dest_x, dest_y):
        print("ERROR: Cannot plan route")
        sys.exit(1)

    try:
        vehicle.start(blocking=True)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        vehicle.stop()


if __name__ == "__main__":
    main()
