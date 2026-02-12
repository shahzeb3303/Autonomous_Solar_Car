"""
Safety Governor - Hard Safety Overrides (Layer 3).

This is the OUTERMOST safety layer that CANNOT be overridden by ML.
It enforces hard safety rules that take precedence over all other decisions.

Safety Rules:
1. front < 20cm → STOP (always, no exceptions)
2. Person detected by camera → STOP (always)
3. Any sensor timeout/error → STOP (fail-safe)
4. Emergency stop triggered → STOP

These rules are NOT learned by ML. They are hard-coded and immutable.
The ML model operates ONLY when the safety governor permits.

Usage:
    from src.decision.safety_governor import SafetyGovernor
    from src.sensors.sensor_fusion import FusedSensorData

    governor = SafetyGovernor()

    # Check before executing any action
    status = governor.check(fused_data)
    if status.requires_stop:
        motor.emergency_stop()
        return  # Don't execute ML decision

    # Safe to proceed with ML decision
    action = decision_engine.predict(fused_data)
"""

import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

from src.sensors.sensor_fusion import FusedSensorData
from config.settings import get_settings
from config.logging_config import get_decision_logger

logger = logging.getLogger(__name__)


class SafetyViolation(Enum):
    """Types of safety violations."""
    NONE = "none"
    CRITICAL_FRONT_DISTANCE = "critical_front_distance"
    PERSON_DETECTED = "person_detected"
    SENSOR_ERROR = "sensor_error"
    SENSOR_TIMEOUT = "sensor_timeout"
    EMERGENCY_STOP = "emergency_stop"
    INVALID_DATA = "invalid_data"


@dataclass
class SafetyStatus:
    """
    Result of safety check.

    Attributes:
        is_safe: Whether it's safe to proceed with ML decision
        requires_stop: Whether immediate stop is required
        violation: Type of safety violation if any
        reason: Human-readable explanation
        override_action: Action to take if not safe (usually STOP)
    """
    is_safe: bool
    requires_stop: bool
    violation: SafetyViolation
    reason: str
    override_action: int = 4  # STOP

    @property
    def can_proceed(self) -> bool:
        """Check if ML decision can proceed."""
        return self.is_safe and not self.requires_stop


class SafetyGovernor:
    """
    Hard safety layer that overrides all other decisions.

    This class implements non-negotiable safety rules that take
    absolute precedence over ML decisions. It's designed to be
    simple, fast, and foolproof.

    The safety governor CANNOT be disabled or overridden programmatically.
    The only way to bypass it is to modify this source code.
    """

    # Class ID for person (matches training data encoding)
    PERSON_CLASS_ID = 1

    # Action IDs
    ACTION_STOP = 4

    def __init__(self):
        """Initialize the safety governor."""
        self._settings = get_settings()
        self._decision_logger = get_decision_logger()

        # Safety thresholds - these are HARD LIMITS
        self.critical_distance_cm = self._settings.SAFETY_STOP_DISTANCE_CM
        self.person_stop_enabled = self._settings.SAFETY_PERSON_STOP
        self.sensor_error_stop = self._settings.SAFETY_SENSOR_ERROR_STOP

        # Emergency stop flag (can be triggered externally)
        self._emergency_stop_triggered = False

        # Statistics
        self._check_count = 0
        self._stop_count = 0

        logger.info(
            f"SafetyGovernor initialized: "
            f"critical_distance={self.critical_distance_cm}cm, "
            f"person_stop={self.person_stop_enabled}"
        )

    def check(self, sensor_data: FusedSensorData) -> SafetyStatus:
        """
        Check sensor data against safety rules.

        This is the main entry point. Call this before every
        decision cycle to ensure safety rules are enforced.

        Args:
            sensor_data: Current fused sensor readings

        Returns:
            SafetyStatus indicating whether it's safe to proceed.
        """
        self._check_count += 1

        # Check in order of priority (highest to lowest)

        # 1. Emergency stop flag
        if self._emergency_stop_triggered:
            return self._create_stop_status(
                SafetyViolation.EMERGENCY_STOP,
                "Emergency stop triggered"
            )

        # 2. Invalid sensor data
        if not sensor_data.is_valid:
            return self._create_stop_status(
                SafetyViolation.INVALID_DATA,
                "Sensor data invalid - possible sensor failure"
            )

        # 3. Critical front distance (< 20cm)
        if sensor_data.front_min_distance < self.critical_distance_cm:
            return self._create_stop_status(
                SafetyViolation.CRITICAL_FRONT_DISTANCE,
                f"Front obstacle at {sensor_data.front_min_distance:.0f}cm "
                f"(threshold: {self.critical_distance_cm}cm)"
            )

        # 4. Person detected
        if self.person_stop_enabled and sensor_data.camera_object_detected == 1:
            if sensor_data.camera_object_class == self.PERSON_CLASS_ID:
                return self._create_stop_status(
                    SafetyViolation.PERSON_DETECTED,
                    f"Person detected at ~{sensor_data.camera_object_distance_estimate:.0f}cm"
                )

        # 5. Sensor reading zero (possible sensor failure)
        if self.sensor_error_stop:
            if sensor_data.front_distance == 0 and sensor_data.front_wp_distance == 0:
                return self._create_stop_status(
                    SafetyViolation.SENSOR_ERROR,
                    "Both front sensors reading zero - possible failure"
                )

        # All checks passed - safe to proceed
        return SafetyStatus(
            is_safe=True,
            requires_stop=False,
            violation=SafetyViolation.NONE,
            reason="All safety checks passed",
            override_action=None,
        )

    def _create_stop_status(
        self,
        violation: SafetyViolation,
        reason: str,
    ) -> SafetyStatus:
        """Create a stop status and log the safety override."""
        self._stop_count += 1

        logger.warning(f"SAFETY OVERRIDE: {violation.value} - {reason}")

        return SafetyStatus(
            is_safe=False,
            requires_stop=True,
            violation=violation,
            reason=reason,
            override_action=self.ACTION_STOP,
        )

    def trigger_emergency_stop(self) -> None:
        """
        Trigger emergency stop mode.

        Once triggered, the safety governor will force STOP
        on all subsequent checks until reset.
        """
        self._emergency_stop_triggered = True
        self._stop_count += 1
        logger.critical("EMERGENCY STOP TRIGGERED")

    def reset_emergency_stop(self) -> None:
        """
        Reset emergency stop mode.

        Should only be called after the emergency has been addressed
        and it's safe to resume operation.
        """
        self._emergency_stop_triggered = False
        logger.info("Emergency stop reset")

    def check_and_override(
        self,
        sensor_data: FusedSensorData,
        proposed_action: int,
    ) -> Tuple[int, bool, str]:
        """
        Check safety and potentially override proposed action.

        This is a convenience method that combines safety check
        with action override logic.

        Args:
            sensor_data: Current sensor readings
            proposed_action: Action proposed by ML/rules (0-6)

        Returns:
            Tuple of (final_action, was_overridden, reason)
        """
        status = self.check(sensor_data)

        if status.requires_stop:
            # Log the override
            self._decision_logger.log_safety_override(
                original_action=str(proposed_action),
                override_action="STOP",
                reason=status.reason,
                sensor_data=sensor_data.to_dict(),
            )
            return (self.ACTION_STOP, True, status.reason)

        return (proposed_action, False, "No override")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get safety governor statistics.

        Returns:
            Dictionary with check counts and stop counts.
        """
        return {
            "total_checks": self._check_count,
            "stop_interventions": self._stop_count,
            "stop_rate": self._stop_count / max(1, self._check_count),
            "emergency_stop_active": self._emergency_stop_triggered,
        }

    @property
    def is_emergency_stop_active(self) -> bool:
        """Check if emergency stop is currently active."""
        return self._emergency_stop_triggered
