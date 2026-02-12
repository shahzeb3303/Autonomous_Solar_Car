"""
Main Autonomy Loop for Autonomous Solar Vehicle.

This is the primary control loop implementing the sense-perceive-decide-act
cycle for autonomous navigation.

Loop Structure:
    1. SENSE: Read all sensors (ultrasonic + camera)
    2. PERCEIVE: Fuse sensor data, detect objects
    3. DECIDE: Safety check → ML inference → Action selection
    4. ACT: Execute motor commands

Threading Model:
    Thread 1 (Main): Decision loop @ 5-10 Hz
    Thread 2 (Camera): Object detection loop @ 3-5 FPS
    Thread 3 (Logger): Async logging

Usage:
    python -m src.core.main_loop

    # Or programmatically:
    from src.core.main_loop import AutonomousVehicle

    vehicle = AutonomousVehicle()
    vehicle.start()  # Non-blocking
    # ... do other things ...
    vehicle.stop()
"""

import time
import signal
import atexit
import threading
from typing import Optional, Dict, Any
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import get_settings
from config.logging_config import setup_logging, get_logger, get_decision_logger

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

logger = logging.getLogger(__name__)


class AutonomousVehicle:
    """
    Main autonomous vehicle controller.

    Orchestrates all subsystems to achieve autonomous navigation:
    - Sensor reading and fusion
    - Object detection
    - Safety checks
    - Decision making (ML + rules)
    - Motor control

    Args:
        simulation_mode: If True, use mock sensors/motors
    """

    def __init__(self, simulation_mode: Optional[bool] = None):
        """Initialize the autonomous vehicle."""
        self._settings = get_settings()
        self._decision_logger = get_decision_logger()

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

        # Runtime state
        self._running = False
        self._main_thread: Optional[threading.Thread] = None
        self._loop_count = 0
        self._last_loop_time = 0.0

        # Target loop frequency
        self._loop_period = 1.0 / self._settings.MAIN_LOOP_FREQUENCY_HZ

        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(
            f"AutonomousVehicle initialized: "
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
        """
        Initialize all subsystems.

        Returns:
            True if all subsystems initialized successfully.
        """
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

    def start(self, blocking: bool = False) -> bool:
        """
        Start the autonomous navigation.

        Args:
            blocking: If True, block until stopped. If False, start in background.

        Returns:
            True if started successfully.
        """
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
        """Main control loop."""
        logger.info("Main loop started")

        while self._running and self._state_machine.state.is_operational:
            loop_start = time.time()

            try:
                self._loop_iteration()
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
            else:
                logger.warning(f"Loop overrun: {elapsed*1000:.1f}ms")

            self._loop_count += 1

        logger.info("Main loop exited")

    def _loop_iteration(self) -> None:
        """Single iteration of the control loop."""

        # ================================================================
        # STEP 1: SENSE - Read all sensors
        # ================================================================
        fused_data = self._sensor_fusion.get_fused_data(
            current_speed=self._action_executor.current_speed,
            heading_error=0.0,  # TODO: Get from GPS/compass
        )

        # ================================================================
        # STEP 2: SAFETY CHECK - Layer 3 (highest priority)
        # ================================================================
        safety_status = self._safety_governor.check(fused_data)

        if safety_status.requires_stop:
            # Safety override - immediate stop
            self._action_executor.emergency_stop()
            self._state_machine.stop(safety_status.reason)
            logger.warning(f"Safety stop: {safety_status.reason}")
            return

        # ================================================================
        # STEP 3: DECIDE - ML inference with rule fallback
        # ================================================================
        action_id, confidence, source = self._hybrid_decision.decide(fused_data)

        # Additional safety check - override if needed
        final_action, was_overridden, override_reason = \
            self._safety_governor.check_and_override(
                fused_data, action_id
            )

        if was_overridden:
            action_id = final_action
            source = "SAFETY"

        # ================================================================
        # STEP 4: ACT - Execute the decision
        # ================================================================
        result = self._action_executor.execute(action_id)

        # ================================================================
        # STEP 5: Update state machine
        # ================================================================
        self._update_state_from_action(Action(action_id))

        # Log decision
        if self._loop_count % 10 == 0:  # Log every 10th iteration
            logger.debug(
                f"Loop {self._loop_count}: action={Action(action_id).name}, "
                f"conf={confidence:.2f}, src={source}, "
                f"front={fused_data.front_min_distance:.0f}cm"
            )

    def _update_state_from_action(self, action: Action) -> None:
        """Update state machine based on executed action."""
        current_state = self._state_machine.state

        if action == Action.STOP:
            if current_state != VehicleState.STOPPED:
                self._state_machine.stop("Action STOP executed")

        elif action in (Action.TURN_LEFT, Action.TURN_RIGHT):
            if current_state == VehicleState.NAVIGATING:
                self._state_machine.avoid_obstacle()

        elif action in (Action.REVERSE_LEFT, Action.REVERSE_RIGHT):
            if current_state != VehicleState.REVERSING:
                self._state_machine.enter_reverse()

        elif action == Action.FORWARD:
            if current_state in (VehicleState.AVOIDING, VehicleState.REVERSING):
                self._state_machine.resume_navigation()

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive vehicle status."""
        return {
            "state": self._state_machine.state.name,
            "running": self._running,
            "loop_count": self._loop_count,
            "last_loop_ms": self._last_loop_time * 1000,
            "current_action": (
                self._action_executor.current_action.name
                if self._action_executor.current_action else None
            ),
            "current_speed": self._action_executor.current_speed,
            "simulation_mode": self._simulation_mode,
            "ml_stats": self._decision_engine.get_statistics(),
            "rule_stats": self._rule_engine.get_statistics(),
            "safety_stats": self._safety_governor.get_statistics(),
        }

    @property
    def state(self) -> VehicleState:
        """Get current vehicle state."""
        return self._state_machine.state

    @property
    def is_running(self) -> bool:
        """Check if vehicle is currently running."""
        return self._running


def main():
    """Main entry point."""
    # Setup logging
    setup_logging()

    print("=" * 60)
    print("Autonomous Solar Vehicle")
    print("Capital University of Science & Technology")
    print("=" * 60)
    print()

    # Create vehicle
    vehicle = AutonomousVehicle()

    # Initialize
    print("Initializing...")
    if not vehicle.initialize():
        print("ERROR: Initialization failed!")
        sys.exit(1)

    print("Initialization complete.")
    print()

    # Start autonomous navigation
    print("Starting autonomous navigation...")
    print("Press Ctrl+C to stop")
    print()

    try:
        vehicle.start(blocking=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        vehicle.stop()

    print("\nVehicle stopped.")

    # Print final statistics
    status = vehicle.get_status()
    print("\nFinal Statistics:")
    print(f"  Loop count: {status['loop_count']}")
    print(f"  Final state: {status['state']}")
    print(f"  ML predictions: {status['ml_stats']['predictions']}")
    print(f"  Safety interventions: {status['safety_stats']['stop_interventions']}")


if __name__ == "__main__":
    main()
