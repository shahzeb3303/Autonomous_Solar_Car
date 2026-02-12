"""
Main Autonomy Loop with Path Planning Navigation.

This version adds point-to-point navigation:
1. Set destination (Point B)
2. A* path planning to find route
3. Follow path while avoiding obstacles
4. Re-plan when obstacles block the path
5. Reach destination

Usage:
    python -m src.core.main_loop_nav --destination 500,300

    # Or in simulation:
    python -m src.core.main_loop_nav --simulation --destination 200,200
"""

import time
import signal
import atexit
import threading
import argparse
from typing import Optional, Dict, Any, Tuple
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import get_settings
from config.logging_config import setup_logging, get_logger

from src.core.state_machine import StateMachine, VehicleState
from src.sensors.ultrasonic import UltrasonicSensorArray
from src.sensors.camera import CameraManager
from src.sensors.sensor_fusion import SensorFusion, FusedSensorData
from src.perception.object_detector import ObjectDetector
from src.perception.distance_estimator import DistanceEstimator
from src.decision.safety_governor import SafetyGovernor
from src.decision.decision_engine import DecisionEngine
from src.decision.rule_engine import RuleEngine, HybridDecisionMaker
from src.decision.action_executor import ActionExecutor, Action
from src.control.motor_controller import MotorController
from src.navigation.navigator import Navigator, NavigationState, NavigationCommand
from src.navigation.odometry import Odometry
from src.navigation.grid_map import Position

logger = logging.getLogger(__name__)


class AutonomousVehicleNav:
    """
    Autonomous vehicle with point-to-point navigation.

    Enhanced version that supports:
    - Setting destination coordinates
    - A* path planning
    - Dynamic obstacle avoidance with re-planning
    - Destination arrival detection

    The system hierarchy is:
    1. SAFETY GOVERNOR: Can always override everything (STOP)
    2. NAVIGATOR: Decides direction to reach goal
    3. ML ENGINE: Fine-grained obstacle avoidance
    4. RULE ENGINE: Fallback when ML is uncertain
    """

    def __init__(self, simulation_mode: Optional[bool] = None):
        """Initialize the autonomous vehicle with navigation."""
        self._settings = get_settings()

        # Determine simulation mode
        if simulation_mode is None:
            simulation_mode = self._settings.SIMULATION_MODE
        self._simulation_mode = simulation_mode

        # State machine
        self._state_machine = StateMachine()

        # Sensors
        self._sensor_array = UltrasonicSensorArray()
        self._camera_manager = CameraManager()
        self._object_detector = ObjectDetector()
        self._distance_estimator = DistanceEstimator()

        # Sensor fusion
        self._sensor_fusion = SensorFusion(
            sensor_array=self._sensor_array,
            camera_manager=self._camera_manager,
            object_detector=self._object_detector,
            distance_estimator=self._distance_estimator,
        )

        # Decision making
        self._safety_governor = SafetyGovernor()
        self._decision_engine = DecisionEngine()
        self._rule_engine = RuleEngine()
        self._hybrid_decision = HybridDecisionMaker(
            decision_engine=self._decision_engine,
            rule_engine=self._rule_engine,
            confidence_threshold=self._settings.ML_CONFIDENCE_THRESHOLD,
        )

        # Motor control
        self._motor_controller = MotorController(enable_cleanup_handlers=True)
        self._action_executor = ActionExecutor(self._motor_controller)

        # Navigation (NEW)
        self._navigator = Navigator(
            map_width_cm=2000,   # 20m x 20m map
            map_height_cm=2000,
            cell_size_cm=10,     # 10cm grid resolution
            vehicle_width_cm=30, # 30cm safety margin
            goal_threshold_cm=30 # Within 30cm = arrived
        )
        self._odometry = Odometry()

        # Runtime state
        self._running = False
        self._main_thread: Optional[threading.Thread] = None
        self._loop_count = 0
        self._last_loop_time = 0.0
        self._destination: Optional[Tuple[float, float]] = None

        # Target loop frequency
        self._loop_period = 1.0 / self._settings.MAIN_LOOP_FREQUENCY_HZ

        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(
            f"AutonomousVehicleNav initialized: "
            f"simulation={simulation_mode}, "
            f"loop_freq={self._settings.MAIN_LOOP_FREQUENCY_HZ}Hz"
        )

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.stop()

    def _cleanup(self) -> None:
        """Clean up resources on exit."""
        if self._running:
            self.stop()

    def initialize(self) -> bool:
        """Initialize all subsystems."""
        logger.info("Initializing vehicle subsystems...")

        try:
            # Initialize sensors
            if not self._sensor_fusion.initialize():
                logger.error("Sensor fusion initialization failed")
                return False

            # Load ML model
            if not self._decision_engine.load_model():
                logger.warning("ML model load failed, using rule-based fallback")

            # Initialize motor controller
            if not self._motor_controller.initialize():
                logger.error("Motor controller initialization failed")
                return False

            logger.info("All subsystems initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False

    def set_destination(self, x_cm: float, y_cm: float) -> bool:
        """
        Set navigation destination.

        Args:
            x_cm: X coordinate in cm (relative to start position)
            y_cm: Y coordinate in cm (relative to start position)

        Returns:
            True if path planning succeeded
        """
        self._destination = (x_cm, y_cm)
        success = self._navigator.set_destination(x_cm, y_cm)

        if success:
            logger.info(f"Destination set: ({x_cm}, {y_cm}) cm")
            print(f"\nDestination set: ({x_cm}, {y_cm}) cm")
            print(f"Path planned with {len(self._navigator.get_path_for_visualization())} waypoints")
        else:
            logger.warning(f"Failed to plan path to ({x_cm}, {y_cm})")
            print(f"\nERROR: Cannot find path to ({x_cm}, {y_cm})")

        return success

    def start(self, blocking: bool = False) -> bool:
        """Start autonomous navigation."""
        if self._running:
            logger.warning("Vehicle already running")
            return True

        # Initialize if not already done
        if not self._motor_controller.is_initialized:
            if not self.initialize():
                return False

        # Transition to navigating state
        if not self._state_machine.start():
            logger.error("Failed to enter NAVIGATING state")
            return False

        self._running = True

        if blocking:
            self._run_main_loop()
        else:
            self._main_thread = threading.Thread(
                target=self._run_main_loop,
                name="MainLoop",
                daemon=True,
            )
            self._main_thread.start()

        logger.info("Autonomous navigation started")
        return True

    def stop(self) -> None:
        """Stop the autonomous navigation."""
        logger.info("Stopping autonomous navigation...")

        self._running = False
        self._state_machine.stop("User requested stop")

        # Wait for main thread to finish
        if self._main_thread and self._main_thread.is_alive():
            self._main_thread.join(timeout=2.0)

        # Stop motors
        self._motor_controller.emergency_stop()

        # Clean up sensors
        self._sensor_fusion.cleanup()

        logger.info("Autonomous navigation stopped")

    def _run_main_loop(self) -> None:
        """Main control loop with navigation."""
        logger.info("Main loop started")

        while self._running and self._state_machine.state.is_operational:
            loop_start = time.time()

            try:
                self._loop_iteration()

                # Check if destination reached
                if self._navigator.state == NavigationState.REACHED:
                    print("\n*** DESTINATION REACHED! ***")
                    logger.info("Destination reached, stopping")
                    self._running = False
                    break

                # Check if navigation failed
                if self._navigator.state == NavigationState.FAILED:
                    print("\n*** NAVIGATION FAILED - No path to destination ***")
                    logger.warning("Navigation failed")
                    self._running = False
                    break

            except Exception as e:
                logger.error(f"Loop error: {e}")
                self._state_machine.enter_error(str(e))
                self._motor_controller.emergency_stop()
                break

            # Maintain loop frequency
            elapsed = time.time() - loop_start
            self._last_loop_time = elapsed

            sleep_time = self._loop_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            self._loop_count += 1

        logger.info("Main loop exited")

    def _loop_iteration(self) -> None:
        """Single iteration of the control loop."""

        # ================================================================
        # STEP 1: SENSE - Read all sensors
        # ================================================================
        fused_data = self._sensor_fusion.get_fused_data(
            current_speed=self._action_executor.current_speed,
            heading_error=0.0,
        )

        # ================================================================
        # STEP 2: UPDATE POSITION (Odometry)
        # ================================================================
        current_action = self._action_executor.current_action
        action_name = current_action.name if current_action else "STOP"
        current_speed = self._action_executor.current_speed

        position = self._odometry.update(action_name, current_speed)
        self._navigator.update_position(position.x, position.y, position.heading)

        # ================================================================
        # STEP 3: UPDATE MAP with sensor readings
        # ================================================================
        self._navigator.update_sensors(
            front_dist=fused_data.front_min_distance,
            left_dist=fused_data.left_distance,
            right_dist=fused_data.right_distance,
            rear_dist=fused_data.rear_distance,
            person_detected=(fused_data.camera_class == 1)
        )

        # ================================================================
        # STEP 4: SAFETY CHECK - Layer 3 (highest priority)
        # ================================================================
        safety_status = self._safety_governor.check(fused_data)

        if safety_status.requires_stop:
            # Safety override - immediate stop
            self._action_executor.emergency_stop()
            logger.warning(f"Safety stop: {safety_status.reason}")

            # If obstacle detected, navigator will replan
            if "obstacle" in safety_status.reason.lower():
                # Navigator will detect blocked path in next iteration
                pass
            return

        # ================================================================
        # STEP 5: GET NAVIGATION COMMAND
        # ================================================================
        nav_command = self._navigator.get_command()

        # ================================================================
        # STEP 6: COMBINE WITH ML for fine-grained control
        # ================================================================
        # If navigating, use navigator's direction but ML's obstacle avoidance
        if nav_command.action == "STOP":
            final_action = Action.STOP
        elif nav_command.action == "FORWARD":
            # ML can still suggest slowing down or minor turns
            ml_action, confidence, _ = self._hybrid_decision.decide(fused_data)

            if ml_action == Action.STOP.value:
                final_action = Action.STOP
            elif ml_action == Action.SLOW_DOWN.value:
                final_action = Action.SLOW_DOWN
            else:
                final_action = Action.FORWARD

        elif nav_command.action == "TURN_LEFT":
            final_action = Action.TURN_LEFT

        elif nav_command.action == "TURN_RIGHT":
            final_action = Action.TURN_RIGHT

        else:
            final_action = Action.STOP

        # ================================================================
        # STEP 7: Safety override check
        # ================================================================
        final_action_id, was_overridden, _ = self._safety_governor.check_and_override(
            fused_data, final_action.value
        )

        if was_overridden:
            final_action = Action(final_action_id)

        # ================================================================
        # STEP 8: EXECUTE ACTION
        # ================================================================
        self._action_executor.execute(final_action.value)

        # ================================================================
        # STEP 9: Logging
        # ================================================================
        if self._loop_count % 20 == 0:  # Log every 20th iteration
            nav_status = self._navigator.get_status()
            logger.info(
                f"Loop {self._loop_count}: "
                f"pos=({position.x:.0f},{position.y:.0f}), "
                f"heading={position.heading:.0f}, "
                f"action={final_action.name}, "
                f"nav_state={nav_status['state']}, "
                f"dist_to_goal={nav_status['distance_to_goal']:.0f}cm"
            )
            print(
                f"  Position: ({position.x:.0f}, {position.y:.0f}) cm | "
                f"Heading: {position.heading:.0f} deg | "
                f"Action: {final_action.name} | "
                f"Distance to goal: {nav_status['distance_to_goal']:.0f} cm"
            )

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive vehicle status."""
        nav_status = self._navigator.get_status()
        odom_stats = self._odometry.get_statistics()

        return {
            "state": self._state_machine.state.name,
            "running": self._running,
            "loop_count": self._loop_count,
            "last_loop_ms": self._last_loop_time * 1000,
            "position": odom_stats['position'],
            "heading": odom_stats['heading'],
            "destination": self._destination,
            "navigation_state": nav_status['state'],
            "distance_to_goal": nav_status['distance_to_goal'],
            "path_length": nav_status['path_length'],
            "obstacles_avoided": nav_status['obstacles_avoided'],
            "replan_attempts": nav_status['replan_attempts'],
            "total_distance": odom_stats['total_distance_cm'],
            "simulation_mode": self._simulation_mode,
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Autonomous Solar Vehicle with Navigation"
    )
    parser.add_argument(
        "--destination", "-d",
        type=str,
        required=True,
        help="Destination coordinates as 'x,y' in cm (e.g., '200,300')"
    )
    parser.add_argument(
        "--simulation", "-s",
        action="store_true",
        help="Run in simulation mode"
    )
    parser.add_argument(
        "--start-position",
        type=str,
        default="0,0",
        help="Starting position as 'x,y' in cm (default: 0,0)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Parse destination
    try:
        dest_x, dest_y = map(float, args.destination.split(','))
    except ValueError:
        print("ERROR: Invalid destination format. Use 'x,y' (e.g., '200,300')")
        sys.exit(1)

    # Parse start position
    try:
        start_x, start_y = map(float, args.start_position.split(','))
    except ValueError:
        print("ERROR: Invalid start position format. Use 'x,y' (e.g., '0,0')")
        sys.exit(1)

    # Setup logging
    setup_logging()

    print("=" * 60)
    print("Autonomous Solar Vehicle - Navigation Mode")
    print("Capital University of Science & Technology")
    print("=" * 60)
    print()
    print(f"Start Position: ({start_x}, {start_y}) cm")
    print(f"Destination:    ({dest_x}, {dest_y}) cm")
    print(f"Simulation:     {args.simulation}")
    print()

    # Create vehicle
    vehicle = AutonomousVehicleNav(simulation_mode=args.simulation)

    # Set starting position
    vehicle._odometry.set_position(start_x, start_y, heading=0)

    # Initialize
    print("Initializing...")
    if not vehicle.initialize():
        print("ERROR: Initialization failed!")
        sys.exit(1)

    print("Initialization complete.")
    print()

    # Set destination
    if not vehicle.set_destination(dest_x, dest_y):
        print("ERROR: Path planning failed!")
        sys.exit(1)

    print()
    print("Starting autonomous navigation...")
    print("Press Ctrl+C to stop")
    print("-" * 60)

    try:
        vehicle.start(blocking=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        vehicle.stop()

    # Print final statistics
    print()
    print("=" * 60)
    print("Navigation Complete")
    print("=" * 60)

    status = vehicle.get_status()
    print(f"Final Position:     ({status['position'][0]:.1f}, {status['position'][1]:.1f}) cm")
    print(f"Destination:        ({dest_x}, {dest_y}) cm")
    print(f"Distance to Goal:   {status['distance_to_goal']:.1f} cm")
    print(f"Total Distance:     {status['total_distance']:.1f} cm")
    print(f"Obstacles Avoided:  {status['obstacles_avoided']}")
    print(f"Replan Attempts:    {status['replan_attempts']}")
    print(f"Navigation State:   {status['navigation_state']}")


if __name__ == "__main__":
    main()
