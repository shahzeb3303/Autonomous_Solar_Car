"""
Navigator - High-level navigation controller.

Integrates path planning with obstacle avoidance and ML decision making.
Handles the complete flow:
1. Set destination (Point B)
2. Plan initial path
3. Follow path while avoiding obstacles
4. Re-plan when obstacles block the path
5. Reach destination
"""

import logging
import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable

from .grid_map import GridMap, Position
from .path_planner import PathPlanner

logger = logging.getLogger(__name__)


class NavigationState(Enum):
    """Navigation state machine states."""
    IDLE = auto()           # No destination set
    PLANNING = auto()       # Computing path
    FOLLOWING = auto()      # Following computed path
    AVOIDING = auto()       # Avoiding obstacle, will replan
    REPLANNING = auto()     # Replanning after obstacle
    REACHED = auto()        # Reached destination
    FAILED = auto()         # Cannot reach destination


@dataclass
class NavigationCommand:
    """Command output from navigator to vehicle controller."""
    action: str             # FORWARD, TURN_LEFT, TURN_RIGHT, STOP, REVERSE
    speed_percent: float    # 0-100
    turn_angle: float       # Degrees to turn (-180 to 180)
    reason: str             # Why this command


class Navigator:
    """
    High-level navigator that combines path planning with reactive obstacle avoidance.

    This is the main interface for point-to-point navigation:
    1. Call set_destination() to set goal
    2. Call update() every control loop cycle
    3. Navigator returns commands (FORWARD, TURN, STOP)
    4. When obstacle detected, it automatically replans

    The navigator works WITH the existing ML safety system:
    - Navigator decides DIRECTION (where to go)
    - Safety Governor can still override with STOP if needed
    - ML Engine handles fine-grained obstacle avoidance
    """

    def __init__(
        self,
        map_width_cm: float = 1000,
        map_height_cm: float = 1000,
        cell_size_cm: float = 10,
        vehicle_width_cm: float = 30,
        goal_threshold_cm: float = 20,
        replan_cooldown_seconds: float = 1.0
    ):
        """
        Initialize the navigator.

        Args:
            map_width_cm: Map width in cm
            map_height_cm: Map height in cm
            cell_size_cm: Grid cell size in cm
            vehicle_width_cm: Vehicle width for obstacle inflation
            goal_threshold_cm: Distance to consider goal reached
            replan_cooldown_seconds: Minimum time between replans
        """
        # Create grid map with obstacle inflation based on vehicle size
        self.grid_map = GridMap(
            width_cm=map_width_cm,
            height_cm=map_height_cm,
            cell_size_cm=cell_size_cm,
            obstacle_inflation_cm=vehicle_width_cm
        )

        # Create path planner
        self.path_planner = PathPlanner(self.grid_map)

        # Configuration
        self.goal_threshold = goal_threshold_cm
        self.replan_cooldown = replan_cooldown_seconds

        # State
        self.state = NavigationState.IDLE
        self.current_position = Position(0, 0, 0)
        self.destination: Optional[Position] = None
        self.last_replan_time = 0
        self.replan_attempts = 0
        self.max_replan_attempts = 5

        # Stats
        self.total_distance_traveled = 0
        self.obstacles_avoided = 0

        logger.info("Navigator initialized")

    def set_destination(self, x_cm: float, y_cm: float) -> bool:
        """
        Set navigation destination (Point B).

        Args:
            x_cm: Destination X coordinate in cm
            y_cm: Destination Y coordinate in cm

        Returns:
            True if path planning started successfully
        """
        self.destination = Position(x_cm, y_cm)
        self.state = NavigationState.PLANNING
        self.replan_attempts = 0

        logger.info(f"Destination set: ({x_cm}, {y_cm})")

        # Plan initial path
        path = self.path_planner.find_path(self.current_position, self.destination)

        if path:
            self.state = NavigationState.FOLLOWING
            logger.info(f"Path planned with {len(path)} waypoints")
            return True
        else:
            self.state = NavigationState.FAILED
            logger.warning("Failed to plan initial path")
            return False

    def update_position(self, x_cm: float, y_cm: float, heading_degrees: float):
        """
        Update current vehicle position.

        This should be called every control loop with odometry data.

        Args:
            x_cm: Current X position
            y_cm: Current Y position
            heading_degrees: Current heading (0 = north)
        """
        old_pos = self.current_position
        self.current_position = Position(x_cm, y_cm, heading_degrees)

        # Track distance traveled
        dist = old_pos.distance_to(self.current_position)
        self.total_distance_traveled += dist

    def update_sensors(
        self,
        front_dist: float,
        left_dist: float,
        right_dist: float,
        rear_dist: float,
        person_detected: bool = False
    ):
        """
        Update map with sensor readings.

        Args:
            front_dist: Front ultrasonic reading (cm)
            left_dist: Left ultrasonic reading (cm)
            right_dist: Right ultrasonic reading (cm)
            rear_dist: Rear ultrasonic reading (cm)
            person_detected: Whether camera detected a person
        """
        # Update grid map with sensor data
        self.grid_map.update_from_sensors(
            self.current_position,
            front_dist,
            left_dist,
            right_dist,
            rear_dist
        )

        # Check if path is now blocked
        if self.state == NavigationState.FOLLOWING:
            if self.path_planner.is_path_blocked(self.current_position):
                self.state = NavigationState.AVOIDING
                self.obstacles_avoided += 1
                logger.info("Obstacle detected in path, switching to avoidance")

    def get_command(self) -> NavigationCommand:
        """
        Get the next navigation command.

        This is the main function called every control loop.
        It returns what action the vehicle should take.

        Returns:
            NavigationCommand with action and parameters
        """
        # State machine
        if self.state == NavigationState.IDLE:
            return NavigationCommand(
                action="STOP",
                speed_percent=0,
                turn_angle=0,
                reason="No destination set"
            )

        elif self.state == NavigationState.REACHED:
            return NavigationCommand(
                action="STOP",
                speed_percent=0,
                turn_angle=0,
                reason="Destination reached!"
            )

        elif self.state == NavigationState.FAILED:
            return NavigationCommand(
                action="STOP",
                speed_percent=0,
                turn_angle=0,
                reason="Navigation failed - no path to destination"
            )

        elif self.state == NavigationState.AVOIDING:
            # Obstacle in path - try to replan
            return self._handle_avoidance()

        elif self.state == NavigationState.FOLLOWING:
            # Check if reached goal
            if self.path_planner.has_reached_goal(self.current_position, self.goal_threshold):
                self.state = NavigationState.REACHED
                logger.info("Destination reached!")
                return NavigationCommand(
                    action="STOP",
                    speed_percent=0,
                    turn_angle=0,
                    reason="Destination reached!"
                )

            # Follow path
            return self._follow_path()

        # Default
        return NavigationCommand(
            action="STOP",
            speed_percent=0,
            turn_angle=0,
            reason="Unknown state"
        )

    def _handle_avoidance(self) -> NavigationCommand:
        """Handle obstacle avoidance and replanning."""
        current_time = time.time()

        # Check replan cooldown
        if current_time - self.last_replan_time < self.replan_cooldown:
            # Still in cooldown, just stop
            return NavigationCommand(
                action="STOP",
                speed_percent=0,
                turn_angle=0,
                reason="Waiting to replan..."
            )

        # Check max replan attempts
        if self.replan_attempts >= self.max_replan_attempts:
            self.state = NavigationState.FAILED
            logger.warning("Max replan attempts reached, navigation failed")
            return NavigationCommand(
                action="STOP",
                speed_percent=0,
                turn_angle=0,
                reason="Cannot find path around obstacles"
            )

        # Try to replan
        self.state = NavigationState.REPLANNING
        self.last_replan_time = current_time
        self.replan_attempts += 1

        logger.info(f"Replanning attempt {self.replan_attempts}/{self.max_replan_attempts}")

        new_path = self.path_planner.replan(self.current_position)

        if new_path:
            self.state = NavigationState.FOLLOWING
            logger.info(f"Replan successful! New path has {len(new_path)} waypoints")
            return self._follow_path()
        else:
            # Replan failed, stay in avoidance to try again
            self.state = NavigationState.AVOIDING
            return NavigationCommand(
                action="STOP",
                speed_percent=0,
                turn_angle=0,
                reason="Replanning failed, will retry..."
            )

    def _follow_path(self) -> NavigationCommand:
        """Generate command to follow the planned path."""
        # Get next waypoint
        waypoint = self.path_planner.get_next_waypoint(
            self.current_position,
            lookahead_distance=30.0
        )

        if waypoint is None:
            return NavigationCommand(
                action="FORWARD",
                speed_percent=30,
                turn_angle=0,
                reason="No waypoint, moving forward slowly"
            )

        # Calculate heading to waypoint
        target_heading = self.path_planner.get_heading_to_waypoint(
            self.current_position,
            waypoint
        )

        # Determine turn direction
        turn_direction = self.path_planner.get_turn_direction(
            self.current_position.heading,
            target_heading
        )

        # Calculate turn angle
        turn_angle = target_heading - self.current_position.heading
        while turn_angle > 180:
            turn_angle -= 360
        while turn_angle < -180:
            turn_angle += 360

        # Calculate speed based on turn angle (slow down for sharp turns)
        abs_turn = abs(turn_angle)
        if abs_turn < 10:
            speed = 70  # Fast when going straight
        elif abs_turn < 30:
            speed = 50  # Medium for gentle turns
        elif abs_turn < 60:
            speed = 35  # Slower for sharper turns
        else:
            speed = 20  # Very slow for sharp turns

        # Distance to goal affects speed too
        dist_to_goal = self.path_planner.distance_to_goal(self.current_position)
        if dist_to_goal < 50:  # Slow down near goal
            speed = min(speed, 30)

        # Determine action
        if turn_direction == 'STRAIGHT':
            action = "FORWARD"
        elif turn_direction == 'LEFT':
            action = "TURN_LEFT"
        else:
            action = "TURN_RIGHT"

        return NavigationCommand(
            action=action,
            speed_percent=speed,
            turn_angle=turn_angle,
            reason=f"Following path to waypoint ({waypoint[0]:.0f}, {waypoint[1]:.0f})"
        )

    def cancel_navigation(self):
        """Cancel current navigation."""
        self.state = NavigationState.IDLE
        self.destination = None
        self.path_planner.clear_path()
        logger.info("Navigation cancelled")

    def get_status(self) -> dict:
        """Get current navigation status."""
        return {
            'state': self.state.name,
            'position': (self.current_position.x, self.current_position.y),
            'heading': self.current_position.heading,
            'destination': (self.destination.x, self.destination.y) if self.destination else None,
            'distance_to_goal': self.path_planner.distance_to_goal(self.current_position),
            'path_length': len(self.path_planner.current_path),
            'replan_attempts': self.replan_attempts,
            'obstacles_avoided': self.obstacles_avoided,
            'total_distance': self.total_distance_traveled
        }

    def get_path_for_visualization(self) -> List[Tuple[float, float]]:
        """Get current path for visualization."""
        return self.path_planner.current_path_world.copy()

    def reset(self):
        """Reset navigator to initial state."""
        self.state = NavigationState.IDLE
        self.destination = None
        self.path_planner.clear_path()
        self.grid_map.reset()
        self.total_distance_traveled = 0
        self.obstacles_avoided = 0
        self.replan_attempts = 0
        logger.info("Navigator reset")
