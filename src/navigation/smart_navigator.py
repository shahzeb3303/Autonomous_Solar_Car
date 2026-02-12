"""
Smart Navigator - Real-world driving behavior.

This navigator thinks like a human driver:
1. Obstacle ahead but sides clear? → Overtake (turn around it)
2. Path blocked completely? → Find alternate route
3. Temporary obstacle (person crossing)? → Wait, then continue
4. Moving obstacle (slow car)? → Overtake if safe

Decision Priority:
1. SAFETY: Always stop for imminent collision (<20cm) or pedestrians
2. OVERTAKE: If obstacle ahead but side is clear, go around
3. SLOW DOWN: If obstacle approaching, reduce speed
4. REPLAN: Only if no immediate way around obstacle
5. CONTINUE: Keep going to destination
"""

import time
import math
import logging
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple, List

from .grid_map import GridMap, Position
from .path_planner import PathPlanner

logger = logging.getLogger(__name__)


class DrivingAction(Enum):
    """Driving actions like a real driver."""
    CONTINUE = auto()       # Keep going on path
    OVERTAKE_LEFT = auto()  # Go around obstacle from left
    OVERTAKE_RIGHT = auto() # Go around obstacle from right
    SLOW_DOWN = auto()      # Reduce speed, obstacle ahead
    WAIT = auto()           # Temporary stop (pedestrian crossing)
    STOP = auto()           # Full stop (emergency)
    REPLAN = auto()         # Need new route (path blocked)
    ARRIVED = auto()        # Reached destination


@dataclass
class DrivingDecision:
    """Decision output with reasoning."""
    action: DrivingAction
    speed_percent: float      # 0-100
    turn_angle: float         # -90 to +90 (negative=left)
    reason: str
    confidence: float         # 0-1


class SmartNavigator:
    """
    Navigator that drives like a human.

    Real-world driving logic:

    SCENARIO 1: Clear path
    → Just drive forward to destination

    SCENARIO 2: Obstacle ahead, left side clear
    → Overtake from left (like overtaking slow car)

    SCENARIO 3: Obstacle ahead, right side clear
    → Overtake from right

    SCENARIO 4: Obstacle ahead, both sides partially clear
    → Choose the clearer side to overtake

    SCENARIO 5: Obstacle very close (<20cm)
    → Emergency stop first, then decide

    SCENARIO 6: Pedestrian detected
    → Stop and wait for them to pass

    SCENARIO 7: All directions blocked
    → Stop and replan entire route

    SCENARIO 8: Reached destination
    → Stop and report success
    """

    # Distance thresholds (cm)
    EMERGENCY_STOP_DIST = 20      # Immediate stop
    OVERTAKE_TRIGGER_DIST = 80    # Start considering overtake
    SLOW_DOWN_DIST = 120          # Start slowing down
    CLEAR_PATH_DIST = 150         # Consider path clear

    # Side clearance for overtaking (cm)
    MIN_SIDE_CLEARANCE = 50       # Minimum space needed to overtake
    SAFE_SIDE_CLEARANCE = 80      # Comfortable overtaking space

    def __init__(
        self,
        map_width_cm: float = 2000,
        map_height_cm: float = 2000,
        cell_size_cm: float = 10,
        goal_threshold_cm: float = 30
    ):
        """Initialize smart navigator."""
        self.grid_map = GridMap(
            width_cm=map_width_cm,
            height_cm=map_height_cm,
            cell_size_cm=cell_size_cm,
            obstacle_inflation_cm=25
        )
        self.path_planner = PathPlanner(self.grid_map)

        self.goal_threshold = goal_threshold_cm
        self.current_position = Position(0, 0, 0)
        self.destination: Optional[Position] = None

        # State tracking
        self.is_overtaking = False
        self.overtake_direction: Optional[str] = None  # 'LEFT' or 'RIGHT'
        self.overtake_start_time = 0
        self.waiting_for_pedestrian = False
        self.pedestrian_wait_start = 0

        # Statistics
        self.overtakes_completed = 0
        self.replans_done = 0
        self.pedestrians_waited = 0

        logger.info("SmartNavigator initialized with real-world driving logic")

    def set_destination(self, x_cm: float, y_cm: float) -> bool:
        """Set navigation destination."""
        self.destination = Position(x_cm, y_cm)

        # Plan initial path
        path = self.path_planner.find_path(self.current_position, self.destination)

        if path:
            logger.info(f"Route planned to ({x_cm}, {y_cm}) - {len(path)} waypoints")
            return True
        else:
            logger.warning(f"Cannot find route to ({x_cm}, {y_cm})")
            return False

    def update_position(self, x: float, y: float, heading: float):
        """Update current vehicle position."""
        self.current_position = Position(x, y, heading)

    def decide(
        self,
        front_dist: float,
        front_left_dist: float,
        front_right_dist: float,
        left_dist: float,
        right_dist: float,
        rear_dist: float,
        person_detected: bool = False,
        person_distance: float = 999
    ) -> DrivingDecision:
        """
        Make driving decision based on sensor readings.

        This is the main brain - thinks like a human driver.

        Args:
            front_dist: Distance to front obstacle (cm)
            front_left_dist: Front-left diagonal sensor (cm)
            front_right_dist: Front-right diagonal sensor (cm)
            left_dist: Left side distance (cm)
            right_dist: Right side distance (cm)
            rear_dist: Rear distance (cm)
            person_detected: Is a person detected by camera
            person_distance: Distance to detected person (cm)

        Returns:
            DrivingDecision with action and parameters
        """

        # ============================================================
        # PRIORITY 1: Check if arrived at destination
        # ============================================================
        if self._has_reached_destination():
            return DrivingDecision(
                action=DrivingAction.ARRIVED,
                speed_percent=0,
                turn_angle=0,
                reason="Destination reached!",
                confidence=1.0
            )

        # ============================================================
        # PRIORITY 2: Pedestrian safety (always stop for people)
        # ============================================================
        if person_detected and person_distance < 150:
            self.waiting_for_pedestrian = True
            self.pedestrian_wait_start = time.time()
            self.pedestrians_waited += 1

            return DrivingDecision(
                action=DrivingAction.WAIT,
                speed_percent=0,
                turn_angle=0,
                reason=f"Pedestrian detected at {person_distance:.0f}cm - waiting",
                confidence=1.0
            )

        # Check if pedestrian has passed (waited > 3 seconds and no longer detected)
        if self.waiting_for_pedestrian:
            if not person_detected or person_distance > 200:
                wait_time = time.time() - self.pedestrian_wait_start
                if wait_time > 2.0:  # Wait at least 2 seconds
                    self.waiting_for_pedestrian = False
                    logger.info("Pedestrian passed, resuming")

        if self.waiting_for_pedestrian:
            return DrivingDecision(
                action=DrivingAction.WAIT,
                speed_percent=0,
                turn_angle=0,
                reason="Waiting for pedestrian to pass",
                confidence=1.0
            )

        # ============================================================
        # PRIORITY 3: Emergency stop (too close to obstacle)
        # ============================================================
        if front_dist < self.EMERGENCY_STOP_DIST:
            # Check if we can immediately go around
            can_go_left = left_dist > self.MIN_SIDE_CLEARANCE
            can_go_right = right_dist > self.MIN_SIDE_CLEARANCE

            if can_go_left or can_go_right:
                # Don't fully stop, start overtaking immediately
                return self._start_overtake(
                    left_dist, right_dist,
                    front_left_dist, front_right_dist,
                    "Obstacle very close"
                )
            else:
                # Truly blocked, must stop
                return DrivingDecision(
                    action=DrivingAction.STOP,
                    speed_percent=0,
                    turn_angle=0,
                    reason=f"Emergency stop - obstacle at {front_dist:.0f}cm, sides blocked",
                    confidence=1.0
                )

        # ============================================================
        # PRIORITY 4: Currently overtaking - continue maneuver
        # ============================================================
        if self.is_overtaking:
            return self._continue_overtake(
                front_dist, left_dist, right_dist,
                front_left_dist, front_right_dist
            )

        # ============================================================
        # PRIORITY 5: Obstacle ahead - decide to overtake or slow down
        # ============================================================
        if front_dist < self.OVERTAKE_TRIGGER_DIST:
            # Obstacle in the way - can we go around?
            can_go_left = left_dist > self.MIN_SIDE_CLEARANCE
            can_go_right = right_dist > self.MIN_SIDE_CLEARANCE

            if can_go_left or can_go_right:
                # Yes! Overtake like a normal driver
                return self._start_overtake(
                    left_dist, right_dist,
                    front_left_dist, front_right_dist,
                    f"Obstacle at {front_dist:.0f}cm"
                )
            else:
                # Both sides blocked - need to replan
                return self._handle_blocked_path(front_dist, left_dist, right_dist)

        # ============================================================
        # PRIORITY 6: Obstacle approaching - slow down
        # ============================================================
        if front_dist < self.SLOW_DOWN_DIST:
            # Something ahead, slow down but keep going
            speed = self._calculate_approach_speed(front_dist)

            return DrivingDecision(
                action=DrivingAction.SLOW_DOWN,
                speed_percent=speed,
                turn_angle=self._get_heading_to_goal(),
                reason=f"Slowing down - obstacle at {front_dist:.0f}cm",
                confidence=0.8
            )

        # ============================================================
        # PRIORITY 7: Path clear - drive towards destination
        # ============================================================
        return self._drive_to_destination()

    def _start_overtake(
        self,
        left_dist: float,
        right_dist: float,
        front_left_dist: float,
        front_right_dist: float,
        reason: str
    ) -> DrivingDecision:
        """Start overtaking maneuver."""

        # Decide which side to overtake from
        # Prefer the side with more space
        left_score = left_dist + front_left_dist * 0.5
        right_score = right_dist + front_right_dist * 0.5

        if left_dist < self.MIN_SIDE_CLEARANCE:
            left_score = 0
        if right_dist < self.MIN_SIDE_CLEARANCE:
            right_score = 0

        if left_score > right_score and left_score > 0:
            self.is_overtaking = True
            self.overtake_direction = 'LEFT'
            self.overtake_start_time = time.time()

            turn_angle = -45  # Turn left
            speed = 40  # Moderate speed while turning

            logger.info(f"Starting LEFT overtake - {reason}")

            return DrivingDecision(
                action=DrivingAction.OVERTAKE_LEFT,
                speed_percent=speed,
                turn_angle=turn_angle,
                reason=f"Overtaking from LEFT - {reason}",
                confidence=0.85
            )

        elif right_score > 0:
            self.is_overtaking = True
            self.overtake_direction = 'RIGHT'
            self.overtake_start_time = time.time()

            turn_angle = 45  # Turn right
            speed = 40

            logger.info(f"Starting RIGHT overtake - {reason}")

            return DrivingDecision(
                action=DrivingAction.OVERTAKE_RIGHT,
                speed_percent=speed,
                turn_angle=turn_angle,
                reason=f"Overtaking from RIGHT - {reason}",
                confidence=0.85
            )

        else:
            # Cannot overtake, must stop
            return DrivingDecision(
                action=DrivingAction.STOP,
                speed_percent=0,
                turn_angle=0,
                reason="Cannot overtake - both sides blocked",
                confidence=1.0
            )

    def _continue_overtake(
        self,
        front_dist: float,
        left_dist: float,
        right_dist: float,
        front_left_dist: float,
        front_right_dist: float
    ) -> DrivingDecision:
        """Continue ongoing overtake maneuver."""

        overtake_duration = time.time() - self.overtake_start_time

        # Phase 1: Turn away from obstacle (0-1 second)
        if overtake_duration < 1.0:
            if self.overtake_direction == 'LEFT':
                return DrivingDecision(
                    action=DrivingAction.OVERTAKE_LEFT,
                    speed_percent=40,
                    turn_angle=-35,
                    reason="Overtaking - moving left",
                    confidence=0.85
                )
            else:
                return DrivingDecision(
                    action=DrivingAction.OVERTAKE_RIGHT,
                    speed_percent=40,
                    turn_angle=35,
                    reason="Overtaking - moving right",
                    confidence=0.85
                )

        # Phase 2: Go past obstacle (1-2 seconds)
        elif overtake_duration < 2.5:
            # Check if front is now clear
            if front_dist > self.CLEAR_PATH_DIST:
                # Obstacle passed, start returning to path
                pass

            # Continue forward with slight angle
            angle = -15 if self.overtake_direction == 'LEFT' else 15

            return DrivingDecision(
                action=DrivingAction.CONTINUE,
                speed_percent=50,
                turn_angle=angle,
                reason="Overtaking - passing obstacle",
                confidence=0.8
            )

        # Phase 3: Return to original path (2+ seconds)
        else:
            # Check if we've cleared the obstacle
            if front_dist > self.OVERTAKE_TRIGGER_DIST:
                # Done overtaking, return to normal driving
                self.is_overtaking = False
                self.overtake_direction = None
                self.overtakes_completed += 1

                logger.info(f"Overtake complete! Total overtakes: {self.overtakes_completed}")

                return self._drive_to_destination()

            # Still need to continue overtake
            angle = 20 if self.overtake_direction == 'LEFT' else -20  # Turn back

            return DrivingDecision(
                action=DrivingAction.CONTINUE,
                speed_percent=45,
                turn_angle=angle,
                reason="Overtaking - returning to path",
                confidence=0.75
            )

    def _handle_blocked_path(
        self,
        front_dist: float,
        left_dist: float,
        right_dist: float
    ) -> DrivingDecision:
        """Handle completely blocked path - need to replan."""

        logger.info("Path blocked on all sides, replanning...")
        self.replans_done += 1

        # Update map with obstacle
        heading_rad = math.radians(self.current_position.heading)
        obs_x = self.current_position.x + front_dist * math.sin(heading_rad)
        obs_y = self.current_position.y + front_dist * math.cos(heading_rad)
        self.grid_map.set_obstacle(obs_x, obs_y)

        # Try to find new path
        if self.destination:
            new_path = self.path_planner.replan(self.current_position)

            if new_path:
                logger.info(f"New path found with {len(new_path)} waypoints")
                return DrivingDecision(
                    action=DrivingAction.REPLAN,
                    speed_percent=0,
                    turn_angle=0,
                    reason="Path blocked - new route calculated",
                    confidence=0.7
                )

        # Cannot find any path
        return DrivingDecision(
            action=DrivingAction.STOP,
            speed_percent=0,
            turn_angle=0,
            reason="Path completely blocked - no route available",
            confidence=1.0
        )

    def _drive_to_destination(self) -> DrivingDecision:
        """Normal driving towards destination."""

        if self.destination is None:
            return DrivingDecision(
                action=DrivingAction.STOP,
                speed_percent=0,
                turn_angle=0,
                reason="No destination set",
                confidence=1.0
            )

        # Get heading to goal
        heading_to_goal = self._get_heading_to_goal()

        # Calculate speed based on turn angle
        abs_turn = abs(heading_to_goal)
        if abs_turn < 10:
            speed = 70
        elif abs_turn < 30:
            speed = 55
        elif abs_turn < 60:
            speed = 40
        else:
            speed = 30

        # Slow down when approaching destination
        dist_to_goal = self._distance_to_destination()
        if dist_to_goal < 100:
            speed = min(speed, 35)
        if dist_to_goal < 50:
            speed = min(speed, 25)

        return DrivingDecision(
            action=DrivingAction.CONTINUE,
            speed_percent=speed,
            turn_angle=heading_to_goal,
            reason=f"Driving to destination - {dist_to_goal:.0f}cm away",
            confidence=0.9
        )

    def _calculate_approach_speed(self, obstacle_dist: float) -> float:
        """Calculate safe approach speed based on obstacle distance."""
        # Linear interpolation:
        # At SLOW_DOWN_DIST (120cm) → 50% speed
        # At OVERTAKE_TRIGGER (80cm) → 30% speed
        # At EMERGENCY (20cm) → 10% speed

        if obstacle_dist >= self.SLOW_DOWN_DIST:
            return 60
        elif obstacle_dist >= self.OVERTAKE_TRIGGER_DIST:
            ratio = (obstacle_dist - self.OVERTAKE_TRIGGER_DIST) / (self.SLOW_DOWN_DIST - self.OVERTAKE_TRIGGER_DIST)
            return 30 + ratio * 20  # 30-50%
        else:
            ratio = (obstacle_dist - self.EMERGENCY_STOP_DIST) / (self.OVERTAKE_TRIGGER_DIST - self.EMERGENCY_STOP_DIST)
            return 10 + ratio * 20  # 10-30%

    def _get_heading_to_goal(self) -> float:
        """Get turn angle needed to face destination."""
        if self.destination is None:
            return 0

        dx = self.destination.x - self.current_position.x
        dy = self.destination.y - self.current_position.y

        # Target heading (0 = north)
        target_heading = math.degrees(math.atan2(dx, dy))
        if target_heading < 0:
            target_heading += 360

        # Current heading
        current_heading = self.current_position.heading % 360

        # Calculate turn angle
        diff = target_heading - current_heading

        # Normalize to -180 to 180
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        return diff

    def _distance_to_destination(self) -> float:
        """Calculate distance to destination."""
        if self.destination is None:
            return float('inf')

        dx = self.destination.x - self.current_position.x
        dy = self.destination.y - self.current_position.y
        return math.sqrt(dx*dx + dy*dy)

    def _has_reached_destination(self) -> bool:
        """Check if destination reached."""
        return self._distance_to_destination() < self.goal_threshold

    def get_statistics(self) -> dict:
        """Get navigation statistics."""
        return {
            'position': (self.current_position.x, self.current_position.y),
            'heading': self.current_position.heading,
            'destination': (self.destination.x, self.destination.y) if self.destination else None,
            'distance_to_goal': self._distance_to_destination(),
            'overtakes_completed': self.overtakes_completed,
            'replans_done': self.replans_done,
            'pedestrians_waited': self.pedestrians_waited,
            'currently_overtaking': self.is_overtaking,
            'overtake_direction': self.overtake_direction
        }
