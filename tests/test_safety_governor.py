"""
Critical Safety Tests for Safety Governor.

These tests verify that safety-critical functionality works correctly.
The safety governor is the outermost layer that CANNOT be overridden by ML.

All tests must pass before deployment. Failure of any safety test
should block deployment.

Run with: pytest tests/test_safety_governor.py -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.decision.safety_governor import SafetyGovernor, SafetyStatus, SafetyViolation
from src.sensors.sensor_fusion import FusedSensorData


class TestSafetyGovernorCritical:
    """Critical safety tests - must all pass."""

    def test_critical_front_distance_triggers_stop(self):
        """TEST: Front distance < 20cm MUST trigger STOP."""
        governor = SafetyGovernor()

        # Test multiple values under critical threshold
        for distance in [0, 5, 10, 15, 19, 19.9]:
            data = FusedSensorData(
                front_distance=distance,
                front_wp_distance=distance,
                front_min_distance=distance,
                is_valid=True,
            )

            status = governor.check(data)

            assert status.requires_stop, f"Distance {distance}cm should trigger STOP"
            assert status.violation == SafetyViolation.CRITICAL_FRONT_DISTANCE

    def test_person_detected_triggers_stop(self):
        """TEST: Person detected MUST trigger STOP."""
        governor = SafetyGovernor()

        data = FusedSensorData(
            front_distance=100,
            front_wp_distance=100,
            front_min_distance=100,
            camera_object_detected=1,
            camera_object_class=1,  # Person
            camera_object_distance_estimate=200,
            is_valid=True,
        )

        status = governor.check(data)

        assert status.requires_stop, "Person detected should trigger STOP"
        assert status.violation == SafetyViolation.PERSON_DETECTED

    def test_sensor_error_triggers_stop(self):
        """TEST: Both front sensors reading zero MUST trigger STOP."""
        governor = SafetyGovernor()

        data = FusedSensorData(
            front_distance=0,
            front_wp_distance=0,
            front_min_distance=0,
            is_valid=True,  # Valid structure but zero readings
        )

        status = governor.check(data)

        assert status.requires_stop, "Both sensors at zero should trigger STOP"

    def test_invalid_data_triggers_stop(self):
        """TEST: Invalid sensor data MUST trigger STOP."""
        governor = SafetyGovernor()

        data = FusedSensorData(
            front_distance=100,
            front_wp_distance=100,
            front_min_distance=100,
            is_valid=False,  # Invalid data
        )

        status = governor.check(data)

        assert status.requires_stop, "Invalid data should trigger STOP"
        assert status.violation == SafetyViolation.INVALID_DATA

    def test_emergency_stop_flag_persists(self):
        """TEST: Emergency stop flag MUST persist until reset."""
        governor = SafetyGovernor()

        # Trigger emergency stop
        governor.trigger_emergency_stop()

        # Even with safe data, should still require stop
        safe_data = FusedSensorData(
            front_distance=200,
            front_wp_distance=200,
            front_min_distance=200,
            is_valid=True,
        )

        status = governor.check(safe_data)
        assert status.requires_stop, "Emergency stop should persist"
        assert status.violation == SafetyViolation.EMERGENCY_STOP

        # After reset, should allow proceed
        governor.reset_emergency_stop()
        status = governor.check(safe_data)
        assert not status.requires_stop, "After reset, should allow proceed"


class TestSafetyGovernorBoundary:
    """Boundary condition tests."""

    def test_exactly_at_critical_threshold(self):
        """TEST: Distance exactly at 20cm should NOT trigger stop."""
        governor = SafetyGovernor()

        data = FusedSensorData(
            front_distance=20.0,
            front_wp_distance=20.0,
            front_min_distance=20.0,
            is_valid=True,
        )

        status = governor.check(data)
        # At exactly 20cm, should NOT stop (< 20 is the rule)
        assert not status.requires_stop

    def test_just_above_critical_threshold(self):
        """TEST: Distance just above 20cm should allow proceed."""
        governor = SafetyGovernor()

        data = FusedSensorData(
            front_distance=20.1,
            front_wp_distance=20.1,
            front_min_distance=20.1,
            is_valid=True,
        )

        status = governor.check(data)
        assert not status.requires_stop

    def test_front_min_takes_smaller_value(self):
        """TEST: front_min should use the smaller sensor value."""
        governor = SafetyGovernor()

        # US_FRONT is safe (100cm) but US_FRONT_WP is critical (15cm)
        data = FusedSensorData(
            front_distance=100,
            front_wp_distance=15,
            front_min_distance=15,  # Min of the two
            is_valid=True,
        )

        status = governor.check(data)
        assert status.requires_stop, "front_min should use smaller value"


class TestSafetyGovernorOverride:
    """Tests for action override functionality."""

    def test_safe_action_not_overridden(self):
        """TEST: Safe conditions should not override proposed action."""
        governor = SafetyGovernor()

        safe_data = FusedSensorData(
            front_distance=150,
            front_wp_distance=150,
            front_min_distance=150,
            is_valid=True,
        )

        proposed_action = 0  # FORWARD

        final_action, was_overridden, reason = governor.check_and_override(
            safe_data, proposed_action
        )

        assert not was_overridden
        assert final_action == proposed_action

    def test_unsafe_action_overridden_to_stop(self):
        """TEST: Unsafe conditions should override to STOP."""
        governor = SafetyGovernor()

        unsafe_data = FusedSensorData(
            front_distance=10,
            front_wp_distance=10,
            front_min_distance=10,
            is_valid=True,
        )

        proposed_action = 0  # FORWARD

        final_action, was_overridden, reason = governor.check_and_override(
            unsafe_data, proposed_action
        )

        assert was_overridden
        assert final_action == 4  # STOP


class TestSafetyGovernorPriority:
    """Tests for rule priority ordering."""

    def test_emergency_stop_highest_priority(self):
        """TEST: Emergency stop takes priority over all other conditions."""
        governor = SafetyGovernor()

        # Trigger emergency stop
        governor.trigger_emergency_stop()

        # Even with person detected, emergency stop should be the violation
        data = FusedSensorData(
            front_distance=5,  # Also critical
            front_wp_distance=5,
            front_min_distance=5,
            camera_object_detected=1,
            camera_object_class=1,  # Person
            is_valid=True,
        )

        status = governor.check(data)

        assert status.requires_stop
        # Emergency stop should be checked first
        assert status.violation == SafetyViolation.EMERGENCY_STOP

    def test_invalid_data_second_priority(self):
        """TEST: Invalid data checked before distance/person."""
        governor = SafetyGovernor()

        data = FusedSensorData(
            front_distance=5,
            front_wp_distance=5,
            front_min_distance=5,
            camera_object_detected=1,
            camera_object_class=1,
            is_valid=False,  # Invalid data
        )

        status = governor.check(data)

        assert status.requires_stop
        assert status.violation == SafetyViolation.INVALID_DATA


class TestSafetyGovernorStatistics:
    """Tests for statistics tracking."""

    def test_check_count_increments(self):
        """TEST: Check count should increment on each check."""
        governor = SafetyGovernor()

        data = FusedSensorData(front_min_distance=100, is_valid=True)

        for i in range(5):
            governor.check(data)

        stats = governor.get_statistics()
        assert stats["total_checks"] == 5

    def test_stop_count_increments_on_violations(self):
        """TEST: Stop count should increment on violations."""
        governor = SafetyGovernor()

        # Safe check
        safe_data = FusedSensorData(front_min_distance=100, is_valid=True)
        governor.check(safe_data)

        # Unsafe check
        unsafe_data = FusedSensorData(front_min_distance=10, is_valid=True)
        governor.check(unsafe_data)

        stats = governor.get_statistics()
        assert stats["total_checks"] == 2
        assert stats["stop_interventions"] == 1


class TestSafetyGovernorNonPerson:
    """Tests for non-person camera detections."""

    def test_non_person_detection_allows_proceed(self):
        """TEST: Non-person detections should allow proceed (at safe distance)."""
        governor = SafetyGovernor()

        # Detect a car (class 2) at safe distance
        data = FusedSensorData(
            front_distance=100,
            front_wp_distance=100,
            front_min_distance=100,
            camera_object_detected=1,
            camera_object_class=2,  # Car, not person
            camera_object_distance_estimate=200,
            is_valid=True,
        )

        status = governor.check(data)

        assert not status.requires_stop
        assert status.is_safe
