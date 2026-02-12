"""
Grid Map for path planning.

Represents the environment as a 2D grid where:
- 0 = free space (can drive through)
- 1 = obstacle (blocked)
- 2 = unknown (not yet explored)

The grid is dynamically updated based on sensor readings.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a position in the real world (cm)."""
    x: float  # cm
    y: float  # cm
    heading: float = 0.0  # degrees, 0 = facing positive Y (north)

    def to_grid(self, cell_size: float) -> Tuple[int, int]:
        """Convert real-world position to grid coordinates."""
        grid_x = int(self.x / cell_size)
        grid_y = int(self.y / cell_size)
        return (grid_x, grid_y)

    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class GridMap:
    """
    2D occupancy grid map for navigation.

    The map uses a coordinate system where:
    - Origin (0,0) is at the center of the grid
    - X increases to the right
    - Y increases upward (north)
    - Each cell represents CELL_SIZE cm x CELL_SIZE cm
    """

    # Cell states
    FREE = 0
    OBSTACLE = 1
    UNKNOWN = 2
    PATH = 3  # For visualization

    def __init__(
        self,
        width_cm: float = 1000,  # 10 meters wide
        height_cm: float = 1000,  # 10 meters tall
        cell_size_cm: float = 10,  # 10cm per cell
        obstacle_inflation_cm: float = 20  # Safety margin around obstacles
    ):
        """
        Initialize the grid map.

        Args:
            width_cm: Map width in centimeters
            height_cm: Map height in centimeters
            cell_size_cm: Size of each grid cell in centimeters
            obstacle_inflation_cm: Safety margin to add around obstacles
        """
        self.width_cm = width_cm
        self.height_cm = height_cm
        self.cell_size = cell_size_cm
        self.obstacle_inflation = obstacle_inflation_cm

        # Calculate grid dimensions
        self.grid_width = int(width_cm / cell_size_cm)
        self.grid_height = int(height_cm / cell_size_cm)

        # Initialize grid with unknown cells
        self.grid = np.full((self.grid_height, self.grid_width), self.UNKNOWN, dtype=np.int8)

        # Offset to convert world coordinates to grid (center origin)
        self.offset_x = self.grid_width // 2
        self.offset_y = self.grid_height // 2

        logger.info(f"GridMap initialized: {self.grid_width}x{self.grid_height} cells, "
                   f"cell_size={cell_size_cm}cm")

    def world_to_grid(self, x_cm: float, y_cm: float) -> Tuple[int, int]:
        """Convert world coordinates (cm) to grid coordinates."""
        grid_x = int(x_cm / self.cell_size) + self.offset_x
        grid_y = int(y_cm / self.cell_size) + self.offset_y
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates (cm)."""
        x_cm = (grid_x - self.offset_x) * self.cell_size
        y_cm = (grid_y - self.offset_y) * self.cell_size
        return (x_cm, y_cm)

    def is_valid(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid coordinates are within bounds."""
        return 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height

    def is_free(self, grid_x: int, grid_y: int) -> bool:
        """Check if a cell is free (not obstacle)."""
        if not self.is_valid(grid_x, grid_y):
            return False
        return self.grid[grid_y, grid_x] == self.FREE

    def is_obstacle(self, grid_x: int, grid_y: int) -> bool:
        """Check if a cell is an obstacle."""
        if not self.is_valid(grid_x, grid_y):
            return True  # Out of bounds = obstacle
        return self.grid[grid_y, grid_x] == self.OBSTACLE

    def set_obstacle(self, x_cm: float, y_cm: float, inflate: bool = True):
        """
        Mark a position as obstacle.

        Args:
            x_cm: X position in world coordinates
            y_cm: Y position in world coordinates
            inflate: Whether to add safety margin around obstacle
        """
        grid_x, grid_y = self.world_to_grid(x_cm, y_cm)

        if inflate:
            # Inflate obstacle by safety margin
            inflation_cells = int(self.obstacle_inflation / self.cell_size)
            for dx in range(-inflation_cells, inflation_cells + 1):
                for dy in range(-inflation_cells, inflation_cells + 1):
                    nx, ny = grid_x + dx, grid_y + dy
                    if self.is_valid(nx, ny):
                        self.grid[ny, nx] = self.OBSTACLE
        else:
            if self.is_valid(grid_x, grid_y):
                self.grid[grid_y, grid_x] = self.OBSTACLE

    def set_free(self, x_cm: float, y_cm: float):
        """Mark a position as free."""
        grid_x, grid_y = self.world_to_grid(x_cm, y_cm)
        if self.is_valid(grid_x, grid_y):
            self.grid[grid_y, grid_x] = self.FREE

    def clear_area(self, center_x: float, center_y: float, radius_cm: float):
        """Mark a circular area as free."""
        radius_cells = int(radius_cm / self.cell_size)
        grid_cx, grid_cy = self.world_to_grid(center_x, center_y)

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx*dx + dy*dy <= radius_cells*radius_cells:
                    nx, ny = grid_cx + dx, grid_cy + dy
                    if self.is_valid(nx, ny):
                        self.grid[ny, nx] = self.FREE

    def update_from_sensors(
        self,
        position: Position,
        front_dist: float,
        left_dist: float,
        right_dist: float,
        rear_dist: float,
        max_range: float = 400  # Max sensor range in cm
    ):
        """
        Update map based on ultrasonic sensor readings.

        Args:
            position: Current vehicle position and heading
            front_dist: Front sensor reading (cm)
            left_dist: Left sensor reading (cm)
            right_dist: Right sensor reading (cm)
            rear_dist: Rear sensor reading (cm)
            max_range: Maximum sensor range
        """
        heading_rad = np.radians(position.heading)

        # Sensor directions relative to heading (in radians)
        sensors = [
            (front_dist, heading_rad),           # Front: same as heading
            (left_dist, heading_rad + np.pi/2),  # Left: 90 degrees left
            (right_dist, heading_rad - np.pi/2), # Right: 90 degrees right
            (rear_dist, heading_rad + np.pi),    # Rear: opposite direction
        ]

        for distance, angle in sensors:
            if distance < max_range:
                # Obstacle detected - mark it
                obs_x = position.x + distance * np.sin(angle)
                obs_y = position.y + distance * np.cos(angle)
                self.set_obstacle(obs_x, obs_y)

                # Mark space between vehicle and obstacle as free
                steps = int(distance / self.cell_size)
                for i in range(steps):
                    d = i * self.cell_size
                    free_x = position.x + d * np.sin(angle)
                    free_y = position.y + d * np.cos(angle)
                    self.set_free(free_x, free_y)
            else:
                # No obstacle - mark ray as free
                for d in range(0, int(max_range), int(self.cell_size)):
                    free_x = position.x + d * np.sin(angle)
                    free_y = position.y + d * np.cos(angle)
                    self.set_free(free_x, free_y)

    def get_neighbors(self, grid_x: int, grid_y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (8-connected)."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = grid_x + dx, grid_y + dy
                if self.is_valid(nx, ny) and not self.is_obstacle(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors

    def reset(self):
        """Reset the map to unknown."""
        self.grid.fill(self.UNKNOWN)

    def __str__(self) -> str:
        """String representation for debugging."""
        symbols = {self.FREE: '.', self.OBSTACLE: '#', self.UNKNOWN: '?', self.PATH: '*'}
        rows = []
        for y in range(self.grid_height - 1, -1, -1):  # Top to bottom
            row = ''.join(symbols.get(self.grid[y, x], '?') for x in range(self.grid_width))
            rows.append(row)
        return '\n'.join(rows)
