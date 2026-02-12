"""
A* Path Planning Algorithm.

Finds the optimal path from point A to point B while avoiding obstacles.
Supports dynamic re-planning when new obstacles are detected.
"""

import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import logging
import time

from .grid_map import GridMap, Position

logger = logging.getLogger(__name__)


@dataclass(order=True)
class Node:
    """
    A* search node.

    Attributes:
        f: Total cost (g + h) - used for priority queue ordering
        g: Cost from start to this node
        h: Heuristic cost from this node to goal
        position: (x, y) grid coordinates
        parent: Parent node for path reconstruction
    """
    f: float
    g: float = field(compare=False)
    h: float = field(compare=False)
    position: Tuple[int, int] = field(compare=False)
    parent: Optional['Node'] = field(default=None, compare=False)


class PathPlanner:
    """
    A* Path Planner for autonomous vehicle navigation.

    Features:
    - Finds shortest path from start to goal
    - Avoids obstacles marked on the grid map
    - Supports 8-directional movement
    - Can re-plan when obstacles are detected
    - Optimized with early termination and path smoothing
    """

    def __init__(self, grid_map: GridMap):
        """
        Initialize the path planner.

        Args:
            grid_map: The occupancy grid map to plan on
        """
        self.grid_map = grid_map
        self.current_path: List[Tuple[int, int]] = []
        self.current_path_world: List[Tuple[float, float]] = []
        self.goal_position: Optional[Position] = None
        self.start_position: Optional[Position] = None

        # Movement costs (diagonal costs more)
        self.STRAIGHT_COST = 1.0
        self.DIAGONAL_COST = 1.414  # sqrt(2)

        logger.info("PathPlanner initialized with A* algorithm")

    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Calculate heuristic (estimated cost to goal).

        Uses Euclidean distance for better path quality.

        Args:
            pos: Current position (grid coordinates)
            goal: Goal position (grid coordinates)

        Returns:
            Estimated cost to reach goal
        """
        dx = abs(pos[0] - goal[0])
        dy = abs(pos[1] - goal[1])
        # Euclidean distance
        return np.sqrt(dx*dx + dy*dy)

    def get_movement_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """Get the cost of moving from one cell to another."""
        dx = abs(from_pos[0] - to_pos[0])
        dy = abs(from_pos[1] - to_pos[1])
        if dx + dy == 2:  # Diagonal
            return self.DIAGONAL_COST
        return self.STRAIGHT_COST

    def find_path(
        self,
        start: Position,
        goal: Position,
        timeout_seconds: float = 1.0
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Find path from start to goal using A* algorithm.

        Args:
            start: Starting position (world coordinates)
            goal: Goal position (world coordinates)
            timeout_seconds: Maximum time to search

        Returns:
            List of waypoints (world coordinates) or None if no path found
        """
        self.start_position = start
        self.goal_position = goal

        # Convert to grid coordinates
        start_grid = self.grid_map.world_to_grid(start.x, start.y)
        goal_grid = self.grid_map.world_to_grid(goal.x, goal.y)

        logger.info(f"Planning path from {start_grid} to {goal_grid}")

        # Check if start or goal is invalid
        if self.grid_map.is_obstacle(*start_grid):
            logger.warning("Start position is inside an obstacle!")
            return None

        if self.grid_map.is_obstacle(*goal_grid):
            logger.warning("Goal position is inside an obstacle!")
            return None

        # A* search
        start_time = time.time()

        # Priority queue: (f_cost, node)
        open_set: List[Node] = []
        start_node = Node(
            f=self.heuristic(start_grid, goal_grid),
            g=0,
            h=self.heuristic(start_grid, goal_grid),
            position=start_grid,
            parent=None
        )
        heapq.heappush(open_set, start_node)

        # Track visited nodes and their best g-costs
        g_costs: Dict[Tuple[int, int], float] = {start_grid: 0}
        visited: set = set()

        while open_set:
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                logger.warning(f"Path planning timeout after {timeout_seconds}s")
                return None

            # Get node with lowest f-cost
            current = heapq.heappop(open_set)

            # Skip if already visited with better cost
            if current.position in visited:
                continue
            visited.add(current.position)

            # Check if reached goal
            if current.position == goal_grid:
                path = self._reconstruct_path(current)
                self.current_path = path
                self.current_path_world = self._path_to_world(path)

                elapsed = time.time() - start_time
                logger.info(f"Path found! {len(path)} waypoints in {elapsed:.3f}s")

                # Smooth the path
                smoothed = self._smooth_path(self.current_path_world)
                return smoothed

            # Explore neighbors
            for neighbor_pos in self.grid_map.get_neighbors(*current.position):
                if neighbor_pos in visited:
                    continue

                # Calculate costs
                move_cost = self.get_movement_cost(current.position, neighbor_pos)
                new_g = current.g + move_cost

                # Skip if we've found a better path to this neighbor
                if neighbor_pos in g_costs and new_g >= g_costs[neighbor_pos]:
                    continue

                g_costs[neighbor_pos] = new_g
                h = self.heuristic(neighbor_pos, goal_grid)

                neighbor_node = Node(
                    f=new_g + h,
                    g=new_g,
                    h=h,
                    position=neighbor_pos,
                    parent=current
                )
                heapq.heappush(open_set, neighbor_node)

        logger.warning("No path found to goal!")
        return None

    def _reconstruct_path(self, goal_node: Node) -> List[Tuple[int, int]]:
        """Reconstruct path from goal node back to start."""
        path = []
        current = goal_node
        while current is not None:
            path.append(current.position)
            current = current.parent
        path.reverse()
        return path

    def _path_to_world(self, path: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Convert grid path to world coordinates."""
        return [self.grid_map.grid_to_world(x, y) for x, y in path]

    def _smooth_path(
        self,
        path: List[Tuple[float, float]],
        weight_data: float = 0.5,
        weight_smooth: float = 0.1,
        tolerance: float = 0.001
    ) -> List[Tuple[float, float]]:
        """
        Smooth the path using gradient descent.

        This removes unnecessary zigzags while keeping the path
        close to the original waypoints.

        Args:
            path: Original path
            weight_data: How much to stay close to original path
            weight_smooth: How much to smooth
            tolerance: Convergence threshold

        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path

        # Make a copy
        smoothed = [list(p) for p in path]

        change = tolerance + 1
        while change >= tolerance:
            change = 0
            for i in range(1, len(path) - 1):  # Skip first and last
                for j in range(2):  # x and y
                    old = smoothed[i][j]

                    # Gradient descent update
                    smoothed[i][j] += weight_data * (path[i][j] - smoothed[i][j])
                    smoothed[i][j] += weight_smooth * (
                        smoothed[i-1][j] + smoothed[i+1][j] - 2 * smoothed[i][j]
                    )

                    change += abs(old - smoothed[i][j])

        return [tuple(p) for p in smoothed]

    def replan(self, current_position: Position) -> Optional[List[Tuple[float, float]]]:
        """
        Re-plan path from current position to goal.

        Called when an obstacle is detected in the current path.

        Args:
            current_position: Current vehicle position

        Returns:
            New path or None if no path found
        """
        if self.goal_position is None:
            logger.warning("No goal set, cannot replan")
            return None

        logger.info("Re-planning path due to obstacle...")
        return self.find_path(current_position, self.goal_position)

    def is_path_blocked(self, current_position: Position) -> bool:
        """
        Check if the current path is blocked by an obstacle.

        Args:
            current_position: Current vehicle position

        Returns:
            True if path is blocked and needs re-planning
        """
        if not self.current_path:
            return False

        # Find the closest waypoint to current position
        current_grid = self.grid_map.world_to_grid(current_position.x, current_position.y)

        # Check remaining path for obstacles
        found_current = False
        for waypoint in self.current_path:
            if waypoint == current_grid:
                found_current = True
            if found_current:
                if self.grid_map.is_obstacle(*waypoint):
                    logger.info(f"Path blocked at {waypoint}")
                    return True

        return False

    def get_next_waypoint(
        self,
        current_position: Position,
        lookahead_distance: float = 30.0  # cm
    ) -> Optional[Tuple[float, float]]:
        """
        Get the next waypoint to navigate towards.

        Uses pure pursuit lookahead to find a waypoint ahead
        of the vehicle on the path.

        Args:
            current_position: Current vehicle position
            lookahead_distance: How far ahead to look for waypoint

        Returns:
            Next waypoint (world coordinates) or None
        """
        if not self.current_path_world:
            return None

        # Find closest point on path
        min_dist = float('inf')
        closest_idx = 0

        for i, waypoint in enumerate(self.current_path_world):
            dist = np.sqrt(
                (waypoint[0] - current_position.x)**2 +
                (waypoint[1] - current_position.y)**2
            )
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Find waypoint at lookahead distance
        accumulated_dist = 0
        for i in range(closest_idx, len(self.current_path_world) - 1):
            wp1 = self.current_path_world[i]
            wp2 = self.current_path_world[i + 1]
            segment_dist = np.sqrt(
                (wp2[0] - wp1[0])**2 + (wp2[1] - wp1[1])**2
            )
            accumulated_dist += segment_dist

            if accumulated_dist >= lookahead_distance:
                return wp2

        # Return final waypoint if lookahead exceeds path
        return self.current_path_world[-1]

    def get_heading_to_waypoint(
        self,
        current_position: Position,
        waypoint: Tuple[float, float]
    ) -> float:
        """
        Calculate heading angle to reach waypoint.

        Args:
            current_position: Current vehicle position
            waypoint: Target waypoint (x, y)

        Returns:
            Heading in degrees (0 = north, 90 = east)
        """
        dx = waypoint[0] - current_position.x
        dy = waypoint[1] - current_position.y

        # atan2 gives angle from positive X axis, we want from positive Y
        heading_rad = np.arctan2(dx, dy)
        heading_deg = np.degrees(heading_rad)

        # Normalize to 0-360
        if heading_deg < 0:
            heading_deg += 360

        return heading_deg

    def get_turn_direction(
        self,
        current_heading: float,
        target_heading: float
    ) -> str:
        """
        Determine which way to turn.

        Args:
            current_heading: Current heading (degrees)
            target_heading: Target heading (degrees)

        Returns:
            'LEFT', 'RIGHT', or 'STRAIGHT'
        """
        diff = target_heading - current_heading

        # Normalize to -180 to 180
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        if abs(diff) < 10:  # Within 10 degrees
            return 'STRAIGHT'
        elif diff > 0:
            return 'RIGHT'
        else:
            return 'LEFT'

    def distance_to_goal(self, current_position: Position) -> float:
        """Calculate distance from current position to goal."""
        if self.goal_position is None:
            return float('inf')
        return current_position.distance_to(self.goal_position)

    def has_reached_goal(self, current_position: Position, threshold: float = 20.0) -> bool:
        """
        Check if vehicle has reached the goal.

        Args:
            current_position: Current vehicle position
            threshold: Distance threshold to consider goal reached (cm)

        Returns:
            True if within threshold of goal
        """
        return self.distance_to_goal(current_position) < threshold

    def clear_path(self):
        """Clear the current path."""
        self.current_path = []
        self.current_path_world = []
        self.goal_position = None
