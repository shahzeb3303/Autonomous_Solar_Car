"""
Tests for navigation module.

Tests:
- A* path planning
- Obstacle avoidance
- Path re-planning
- Odometry
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import numpy as np

from src.navigation.grid_map import GridMap, Position
from src.navigation.path_planner import PathPlanner
from src.navigation.navigator import Navigator, NavigationState
from src.navigation.odometry import Odometry


class TestGridMap(unittest.TestCase):
    """Tests for GridMap."""

    def setUp(self):
        self.grid_map = GridMap(
            width_cm=500,
            height_cm=500,
            cell_size_cm=10,
            obstacle_inflation_cm=20
        )

    def test_world_to_grid_conversion(self):
        """Test coordinate conversions."""
        # Center should be at grid center
        gx, gy = self.grid_map.world_to_grid(0, 0)
        self.assertEqual(gx, self.grid_map.grid_width // 2)
        self.assertEqual(gy, self.grid_map.grid_height // 2)

    def test_set_obstacle(self):
        """Test obstacle setting with inflation."""
        self.grid_map.set_obstacle(100, 100)
        gx, gy = self.grid_map.world_to_grid(100, 100)

        # Should be obstacle
        self.assertTrue(self.grid_map.is_obstacle(gx, gy))

        # Nearby cells should also be obstacles (inflation)
        self.assertTrue(self.grid_map.is_obstacle(gx + 1, gy))
        self.assertTrue(self.grid_map.is_obstacle(gx, gy + 1))

    def test_set_free(self):
        """Test marking cells as free."""
        self.grid_map.set_free(50, 50)
        gx, gy = self.grid_map.world_to_grid(50, 50)
        self.assertTrue(self.grid_map.is_free(gx, gy))


class TestPathPlanner(unittest.TestCase):
    """Tests for A* PathPlanner."""

    def setUp(self):
        self.grid_map = GridMap(
            width_cm=500,
            height_cm=500,
            cell_size_cm=10,
            obstacle_inflation_cm=10
        )
        # Mark area as free
        self.grid_map.clear_area(0, 0, 200)
        self.path_planner = PathPlanner(self.grid_map)

    def test_find_path_no_obstacles(self):
        """Test path finding without obstacles."""
        start = Position(0, 0, 0)
        goal = Position(100, 100, 0)

        path = self.path_planner.find_path(start, goal)

        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)

        # First point should be near start
        self.assertAlmostEqual(path[0][0], 0, delta=15)
        self.assertAlmostEqual(path[0][1], 0, delta=15)

        # Last point should be near goal
        self.assertAlmostEqual(path[-1][0], 100, delta=15)
        self.assertAlmostEqual(path[-1][1], 100, delta=15)

    def test_find_path_with_obstacle(self):
        """Test path finding around obstacle."""
        # Add obstacle in the middle
        self.grid_map.set_obstacle(50, 50, inflate=True)

        start = Position(0, 0, 0)
        goal = Position(100, 100, 0)

        path = self.path_planner.find_path(start, goal)

        self.assertIsNotNone(path)

        # Path should go around obstacle
        for wx, wy in path:
            gx, gy = self.grid_map.world_to_grid(wx, wy)
            self.assertFalse(self.grid_map.is_obstacle(gx, gy))

    def test_no_path_to_blocked_goal(self):
        """Test when goal is blocked."""
        # Block entire right side
        for y in range(-200, 200, 10):
            self.grid_map.set_obstacle(80, y, inflate=False)

        start = Position(0, 0, 0)
        goal = Position(150, 0, 0)  # Behind the wall

        path = self.path_planner.find_path(start, goal)

        # Should fail to find path
        self.assertIsNone(path)

    def test_heading_calculation(self):
        """Test heading to waypoint calculation."""
        current = Position(0, 0, 0)

        # North
        heading = self.path_planner.get_heading_to_waypoint(current, (0, 100))
        self.assertAlmostEqual(heading, 0, delta=5)

        # East
        heading = self.path_planner.get_heading_to_waypoint(current, (100, 0))
        self.assertAlmostEqual(heading, 90, delta=5)

        # South
        heading = self.path_planner.get_heading_to_waypoint(current, (0, -100))
        self.assertAlmostEqual(heading, 180, delta=5)

    def test_turn_direction(self):
        """Test turn direction determination."""
        # Should turn right
        direction = self.path_planner.get_turn_direction(0, 45)
        self.assertEqual(direction, 'RIGHT')

        # Should turn left
        direction = self.path_planner.get_turn_direction(0, -45)
        self.assertEqual(direction, 'LEFT')

        # Should go straight
        direction = self.path_planner.get_turn_direction(0, 5)
        self.assertEqual(direction, 'STRAIGHT')


class TestNavigator(unittest.TestCase):
    """Tests for Navigator."""

    def setUp(self):
        self.navigator = Navigator(
            map_width_cm=500,
            map_height_cm=500,
            cell_size_cm=10,
            goal_threshold_cm=20
        )

    def test_initial_state(self):
        """Test navigator starts in IDLE state."""
        self.assertEqual(self.navigator.state, NavigationState.IDLE)

    def test_set_destination(self):
        """Test setting destination."""
        success = self.navigator.set_destination(100, 100)
        self.assertTrue(success)
        self.assertEqual(self.navigator.state, NavigationState.FOLLOWING)

    def test_get_command_idle(self):
        """Test command when idle (no destination)."""
        command = self.navigator.get_command()
        self.assertEqual(command.action, "STOP")

    def test_get_command_following(self):
        """Test command when following path."""
        self.navigator.set_destination(100, 0)  # Straight ahead
        command = self.navigator.get_command()

        # Should be moving (not stopped)
        self.assertIn(command.action, ["FORWARD", "TURN_LEFT", "TURN_RIGHT"])

    def test_cancel_navigation(self):
        """Test cancelling navigation."""
        self.navigator.set_destination(100, 100)
        self.navigator.cancel_navigation()
        self.assertEqual(self.navigator.state, NavigationState.IDLE)


class TestOdometry(unittest.TestCase):
    """Tests for Odometry."""

    def setUp(self):
        self.odometry = Odometry()

    def test_initial_position(self):
        """Test initial position is origin."""
        pos = self.odometry.get_position()
        self.assertEqual(pos.x, 0)
        self.assertEqual(pos.y, 0)
        self.assertEqual(pos.heading, 0)

    def test_forward_movement(self):
        """Test forward movement updates position."""
        # Simulate forward movement for 1 second at 50% speed
        self.odometry.last_update_time -= 1.0  # Fake 1 second elapsed
        pos = self.odometry.update("FORWARD", 50)

        # Should have moved forward (positive Y)
        self.assertGreater(pos.y, 0)
        self.assertAlmostEqual(pos.x, 0, delta=1)

    def test_turn_updates_heading(self):
        """Test turning updates heading."""
        self.odometry.last_update_time -= 1.0
        pos = self.odometry.update("TURN_RIGHT", 50)

        # Heading should have increased
        self.assertGreater(pos.heading, 0)

    def test_set_position(self):
        """Test manual position setting."""
        self.odometry.set_position(100, 200, 90)
        pos = self.odometry.get_position()

        self.assertEqual(pos.x, 100)
        self.assertEqual(pos.y, 200)
        self.assertEqual(pos.heading, 90)

    def test_reset(self):
        """Test reset returns to origin."""
        self.odometry.set_position(100, 200, 90)
        self.odometry.reset()

        pos = self.odometry.get_position()
        self.assertEqual(pos.x, 0)
        self.assertEqual(pos.y, 0)


def run_tests():
    """Run all navigation tests."""
    print("=" * 60)
    print("Navigation Module Tests")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestGridMap))
    suite.addTests(loader.loadTestsFromTestCase(TestPathPlanner))
    suite.addTests(loader.loadTestsFromTestCase(TestNavigator))
    suite.addTests(loader.loadTestsFromTestCase(TestOdometry))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print()
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)

    return len(result.failures) + len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
