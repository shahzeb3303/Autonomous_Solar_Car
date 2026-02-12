"""
Action Executor - Translates Actions to Motor Commands.

This module bridges the decision layer with motor control,
translating high-level action decisions into low-level
motor commands.

Actions:
    0: FORWARD      - Continue straight at current speed
    1: SLOW_DOWN    - Reduce speed by 30-50%
    2: TURN_LEFT    - Steer left (differential steering)
    3: TURN_RIGHT   - Steer right (differential steering)
    4: STOP         - Full stop
    5: REVERSE_LEFT - Back up while turning left
    6: REVERSE_RIGHT- Back up while turning right

Usage:
    from src.decision.action_executor import ActionExecutor
    from src.control.motor_controller import MotorController

    motor = MotorController()
    executor = ActionExecutor(motor)

    executor.execute(action_id=0, speed=50)  # Forward at 50%
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum
import logging

from src.control.motor_controller import MotorController
from config.settings import get_settings

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """Action enumeration matching model output."""
    FORWARD = 0
    SLOW_DOWN = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    STOP = 4
    REVERSE_LEFT = 5
    REVERSE_RIGHT = 6

    @property
    def name_str(self) -> str:
        """Get action name as string."""
        return self.name


@dataclass
class ExecutionResult:
    """Result of action execution."""
    action: Action
    success: bool
    speed_applied: int
    error: Optional[str] = None


class ActionExecutor:
    """
    Executes navigation actions via motor control.

    Translates decision actions (0-6) into motor commands,
    handling speed management and turn execution.

    Args:
        motor_controller: Initialized MotorController instance
    """

    def __init__(self, motor_controller: MotorController):
        """Initialize the action executor."""
        self._settings = get_settings()
        self._motor = motor_controller

        # Current state
        self._current_speed = 0
        self._current_action: Optional[Action] = None

        # Statistics
        self._execution_count = 0
        self._action_counts: Dict[Action, int] = {a: 0 for a in Action}

        logger.info("ActionExecutor initialized")

    def execute(
        self,
        action_id: int,
        speed: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute a navigation action.

        Args:
            action_id: Action to execute (0-6)
            speed: Base speed (0-100), None uses default

        Returns:
            ExecutionResult with execution status.
        """
        try:
            action = Action(action_id)
        except ValueError:
            logger.error(f"Invalid action ID: {action_id}")
            return ExecutionResult(
                action=Action.STOP,
                success=False,
                speed_applied=0,
                error=f"Invalid action ID: {action_id}",
            )

        self._execution_count += 1
        self._action_counts[action] += 1

        # Determine speed
        if speed is None:
            speed = self._get_default_speed(action)

        try:
            # Execute action
            if action == Action.FORWARD:
                self._execute_forward(speed)
            elif action == Action.SLOW_DOWN:
                self._execute_slow_down()
            elif action == Action.TURN_LEFT:
                self._execute_turn_left(speed)
            elif action == Action.TURN_RIGHT:
                self._execute_turn_right(speed)
            elif action == Action.STOP:
                self._execute_stop()
            elif action == Action.REVERSE_LEFT:
                self._execute_reverse_left(speed)
            elif action == Action.REVERSE_RIGHT:
                self._execute_reverse_right(speed)

            self._current_action = action
            self._current_speed = speed

            logger.debug(f"Executed {action.name} at speed {speed}")

            return ExecutionResult(
                action=action,
                success=True,
                speed_applied=speed,
            )

        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            # Safety: stop on error
            self._motor.emergency_stop()

            return ExecutionResult(
                action=action,
                success=False,
                speed_applied=0,
                error=str(e),
            )

    def _get_default_speed(self, action: Action) -> int:
        """Get default speed for an action."""
        if action == Action.FORWARD:
            return self._settings.SPEED_NORMAL
        elif action == Action.SLOW_DOWN:
            return self._settings.SPEED_SLOW
        elif action in (Action.TURN_LEFT, Action.TURN_RIGHT):
            return self._settings.TURN_SPEED_OUTER
        elif action in (Action.REVERSE_LEFT, Action.REVERSE_RIGHT):
            return self._settings.SPEED_SLOW
        else:
            return 0

    def _execute_forward(self, speed: int) -> None:
        """Execute forward movement."""
        self._motor.forward(speed)

    def _execute_slow_down(self) -> None:
        """Execute slow down (reduce current speed)."""
        # Reduce speed by configured factor
        new_speed = int(self._current_speed * (1 - self._settings.SLOW_DOWN_FACTOR))
        new_speed = max(self._settings.SPEED_SLOW, new_speed)
        self._motor.forward(new_speed)
        self._current_speed = new_speed

    def _execute_turn_left(self, speed: int) -> None:
        """Execute left turn."""
        self._motor.turn_left(speed)

    def _execute_turn_right(self, speed: int) -> None:
        """Execute right turn."""
        self._motor.turn_right(speed)

    def _execute_stop(self) -> None:
        """Execute stop."""
        self._motor.stop()
        self._current_speed = 0

    def _execute_reverse_left(self, speed: int) -> None:
        """Execute reverse while turning left."""
        self._motor.reverse_left(speed)

    def _execute_reverse_right(self, speed: int) -> None:
        """Execute reverse while turning right."""
        self._motor.reverse_right(speed)

    def emergency_stop(self) -> None:
        """Execute immediate emergency stop."""
        self._motor.emergency_stop()
        self._current_speed = 0
        self._current_action = Action.STOP
        logger.warning("Emergency stop executed by ActionExecutor")

    def set_speed(self, speed: int) -> None:
        """Set motor speed while maintaining current action."""
        if self._current_action == Action.FORWARD:
            self._motor.forward(speed)
        self._current_speed = speed

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "total_executions": self._execution_count,
            "action_distribution": {
                a.name: self._action_counts[a] for a in Action
            },
            "current_action": (
                self._current_action.name if self._current_action else None
            ),
            "current_speed": self._current_speed,
        }

    @property
    def current_speed(self) -> int:
        """Get current speed setting."""
        return self._current_speed

    @property
    def current_action(self) -> Optional[Action]:
        """Get current action."""
        return self._current_action

    @property
    def is_moving(self) -> bool:
        """Check if vehicle is currently moving."""
        return self._current_speed > 0 and self._current_action != Action.STOP
