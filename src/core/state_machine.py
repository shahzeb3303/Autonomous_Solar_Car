"""
Vehicle State Machine.

This module manages the overall vehicle state and transitions.

States:
    IDLE        - Vehicle is stationary, not navigating
    NAVIGATING  - Normal forward navigation
    AVOIDING    - Obstacle avoidance maneuver in progress
    STOPPED     - Emergency or safety stop
    REVERSING   - Backing up after obstacle
    ERROR       - System error state

State Transitions:
    IDLE → NAVIGATING: Start command received
    NAVIGATING → AVOIDING: Obstacle detected in path
    NAVIGATING → STOPPED: Safety stop triggered
    AVOIDING → NAVIGATING: Obstacle cleared
    AVOIDING → STOPPED: Cannot avoid, must stop
    STOPPED → IDLE: Manual reset
    ANY → ERROR: System fault detected

Usage:
    from src.core.state_machine import StateMachine, VehicleState

    sm = StateMachine()
    sm.start()  # Transition to NAVIGATING

    if obstacle_detected:
        sm.transition_to(VehicleState.AVOIDING)
"""

from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Any
import logging
import time

logger = logging.getLogger(__name__)


class VehicleState(Enum):
    """Vehicle operational states."""
    IDLE = auto()
    NAVIGATING = auto()
    AVOIDING = auto()
    STOPPED = auto()
    REVERSING = auto()
    ERROR = auto()

    @property
    def is_moving(self) -> bool:
        """Check if this state involves movement."""
        return self in (
            VehicleState.NAVIGATING,
            VehicleState.AVOIDING,
            VehicleState.REVERSING,
        )

    @property
    def is_operational(self) -> bool:
        """Check if vehicle is operational (not error or stopped)."""
        return self not in (VehicleState.STOPPED, VehicleState.ERROR)


class StateTransition:
    """Represents a state transition."""

    def __init__(
        self,
        from_state: VehicleState,
        to_state: VehicleState,
        timestamp: float,
        reason: str,
    ):
        self.from_state = from_state
        self.to_state = to_state
        self.timestamp = timestamp
        self.reason = reason


class StateMachine:
    """
    Vehicle state machine managing operational states.

    Handles state transitions, validates allowed transitions,
    and notifies subscribers of state changes.
    """

    # Valid state transitions
    VALID_TRANSITIONS: Dict[VehicleState, List[VehicleState]] = {
        VehicleState.IDLE: [
            VehicleState.NAVIGATING,
            VehicleState.ERROR,
        ],
        VehicleState.NAVIGATING: [
            VehicleState.IDLE,
            VehicleState.AVOIDING,
            VehicleState.STOPPED,
            VehicleState.REVERSING,
            VehicleState.ERROR,
        ],
        VehicleState.AVOIDING: [
            VehicleState.NAVIGATING,
            VehicleState.STOPPED,
            VehicleState.REVERSING,
            VehicleState.ERROR,
        ],
        VehicleState.REVERSING: [
            VehicleState.NAVIGATING,
            VehicleState.AVOIDING,
            VehicleState.STOPPED,
            VehicleState.ERROR,
        ],
        VehicleState.STOPPED: [
            VehicleState.IDLE,
            VehicleState.NAVIGATING,
            VehicleState.ERROR,
        ],
        VehicleState.ERROR: [
            VehicleState.IDLE,
        ],
    }

    def __init__(self, initial_state: VehicleState = VehicleState.IDLE):
        """Initialize the state machine."""
        self._current_state = initial_state
        self._previous_state = initial_state
        self._state_entered_time = time.time()

        # State change callbacks
        self._listeners: List[Callable[[VehicleState, VehicleState], None]] = []

        # Transition history
        self._history: List[StateTransition] = []
        self._max_history = 100

        logger.info(f"StateMachine initialized: state={initial_state.name}")

    @property
    def state(self) -> VehicleState:
        """Get current state."""
        return self._current_state

    @property
    def previous_state(self) -> VehicleState:
        """Get previous state."""
        return self._previous_state

    @property
    def time_in_state(self) -> float:
        """Get time spent in current state (seconds)."""
        return time.time() - self._state_entered_time

    def can_transition_to(self, new_state: VehicleState) -> bool:
        """Check if transition to new state is valid."""
        valid_targets = self.VALID_TRANSITIONS.get(self._current_state, [])
        return new_state in valid_targets

    def transition_to(self, new_state: VehicleState, reason: str = "") -> bool:
        """
        Transition to a new state.

        Args:
            new_state: Target state
            reason: Reason for transition (for logging)

        Returns:
            True if transition successful, False otherwise.
        """
        if new_state == self._current_state:
            return True  # Already in this state

        if not self.can_transition_to(new_state):
            logger.warning(
                f"Invalid transition: {self._current_state.name} → {new_state.name}"
            )
            return False

        # Record transition
        transition = StateTransition(
            from_state=self._current_state,
            to_state=new_state,
            timestamp=time.time(),
            reason=reason,
        )
        self._history.append(transition)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Update state
        self._previous_state = self._current_state
        self._current_state = new_state
        self._state_entered_time = time.time()

        logger.info(
            f"State transition: {self._previous_state.name} → {new_state.name} "
            f"({reason})"
        )

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(self._previous_state, new_state)
            except Exception as e:
                logger.error(f"State listener error: {e}")

        return True

    def force_transition(self, new_state: VehicleState, reason: str = "") -> None:
        """
        Force transition to a new state (bypasses validation).

        Use sparingly - for emergency situations only.
        """
        logger.warning(
            f"FORCED transition: {self._current_state.name} → {new_state.name} "
            f"({reason})"
        )

        self._previous_state = self._current_state
        self._current_state = new_state
        self._state_entered_time = time.time()

        # Record
        self._history.append(StateTransition(
            from_state=self._previous_state,
            to_state=new_state,
            timestamp=time.time(),
            reason=f"FORCED: {reason}",
        ))

    def add_listener(
        self,
        callback: Callable[[VehicleState, VehicleState], None],
    ) -> None:
        """Add a state change listener."""
        self._listeners.append(callback)

    def remove_listener(
        self,
        callback: Callable[[VehicleState, VehicleState], None],
    ) -> None:
        """Remove a state change listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    # Convenience methods for common transitions

    def start(self) -> bool:
        """Start navigation from IDLE."""
        return self.transition_to(VehicleState.NAVIGATING, "Start command")

    def stop(self, reason: str = "Stop requested") -> bool:
        """Stop vehicle."""
        return self.transition_to(VehicleState.STOPPED, reason)

    def avoid_obstacle(self) -> bool:
        """Enter obstacle avoidance mode."""
        return self.transition_to(VehicleState.AVOIDING, "Obstacle detected")

    def resume_navigation(self) -> bool:
        """Resume normal navigation after avoidance."""
        return self.transition_to(VehicleState.NAVIGATING, "Obstacle cleared")

    def enter_reverse(self, reason: str = "Reversing") -> bool:
        """Enter reverse mode."""
        return self.transition_to(VehicleState.REVERSING, reason)

    def enter_error(self, error_msg: str) -> bool:
        """Enter error state."""
        return self.transition_to(VehicleState.ERROR, error_msg)

    def reset(self) -> bool:
        """Reset to IDLE state (from STOPPED or ERROR)."""
        return self.transition_to(VehicleState.IDLE, "Manual reset")

    def get_history(self, limit: int = 10) -> List[StateTransition]:
        """Get recent state transition history."""
        return self._history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get state machine statistics."""
        return {
            "current_state": self._current_state.name,
            "previous_state": self._previous_state.name,
            "time_in_state": self.time_in_state,
            "total_transitions": len(self._history),
            "is_moving": self._current_state.is_moving,
            "is_operational": self._current_state.is_operational,
        }
