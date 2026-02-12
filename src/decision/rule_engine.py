"""
Rule Engine - Fallback Rule-Based Decisions (Layer 1).

This module provides rule-based decision making as a fallback
when the ML model is unavailable or has low confidence.

The rules mirror those used for training data generation,
ensuring consistent behavior when ML cannot be used.

Use Cases:
1. ML model fails to load
2. ML prediction confidence < threshold
3. Debugging/validation mode

Usage:
    from src.decision.rule_engine import RuleEngine
    from src.sensors.sensor_fusion import FusedSensorData

    engine = RuleEngine()
    action = engine.decide(fused_data)
"""

from typing import Tuple, Dict, Any
import logging

from src.sensors.sensor_fusion import FusedSensorData
from config.settings import get_settings
from config.logging_config import get_decision_logger

logger = logging.getLogger(__name__)


class Actions:
    """Action constants matching the ML model."""
    FORWARD = 0
    SLOW_DOWN = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    STOP = 4
    REVERSE_LEFT = 5
    REVERSE_RIGHT = 6

    NAMES = [
        "FORWARD",
        "SLOW_DOWN",
        "TURN_LEFT",
        "TURN_RIGHT",
        "STOP",
        "REVERSE_LEFT",
        "REVERSE_RIGHT",
    ]


class RuleEngine:
    """
    Rule-based decision engine implementing expert driving rules.

    These rules are deterministic and match the training data
    generation rules. They provide a reliable fallback when
    the ML model cannot be used.

    Thresholds are configurable via settings.
    """

    # Distance thresholds (cm)
    CRITICAL_DISTANCE = 20
    DANGER_DISTANCE = 50
    CAUTION_DISTANCE = 100
    CLEAR_DISTANCE = 150

    SIDE_CRITICAL = 30
    SIDE_SAFE = 60

    # Heading correction threshold (degrees)
    HEADING_CORRECTION_THRESHOLD = 15

    def __init__(self):
        """Initialize the rule engine."""
        self._settings = get_settings()
        self._decision_logger = get_decision_logger()

        # Override thresholds from settings if available
        self.critical_distance = self._settings.CRITICAL_DISTANCE_CM
        self.danger_distance = self._settings.DANGER_DISTANCE_CM
        self.caution_distance = self._settings.CAUTION_DISTANCE_CM
        self.clear_distance = self._settings.CLEAR_DISTANCE_CM
        self.side_critical = self._settings.SIDE_CRITICAL_CM
        self.side_safe = self._settings.SIDE_SAFE_CM

        # Statistics
        self._decision_count = 0
        self._action_counts: Dict[int, int] = {i: 0 for i in range(7)}

        logger.info("RuleEngine initialized")

    def decide(self, sensor_data: FusedSensorData) -> Tuple[int, float]:
        """
        Decide action based on rules.

        Args:
            sensor_data: Fused sensor readings

        Returns:
            Tuple of (action_id, confidence)
            Confidence is always 1.0 for rule-based decisions.
        """
        self._decision_count += 1

        # Extract sensor values
        front = sensor_data.front_min_distance
        left = sensor_data.left_distance
        right = sensor_data.right_distance
        rear = sensor_data.rear_distance

        camera_detected = sensor_data.camera_object_detected
        camera_class = sensor_data.camera_object_class
        camera_position = sensor_data.camera_object_position
        camera_distance = sensor_data.camera_object_distance_estimate

        heading_error = sensor_data.heading_error

        # Apply rules in priority order

        # RULE 1: Person detected - STOP
        if camera_detected == 1 and camera_class == 1:
            return self._decide(Actions.STOP, "Person detected")

        # RULE 2: CRITICAL ZONE (front < 20cm)
        if front < self.critical_distance:
            return self._decide(Actions.STOP, f"Critical: front={front:.0f}cm")

        # RULE 3: DANGER ZONE (front 20-50cm)
        if front < self.danger_distance:
            # Try to turn away
            if left > self.side_safe and left > right:
                return self._decide(Actions.TURN_LEFT, f"Danger zone, left clear")
            elif right > self.side_safe and right >= left:
                return self._decide(Actions.TURN_RIGHT, f"Danger zone, right clear")
            else:
                # Both sides blocked - try reverse
                if rear > self.critical_distance:
                    if right > left:
                        return self._decide(Actions.REVERSE_RIGHT, "Danger, reversing right")
                    else:
                        return self._decide(Actions.REVERSE_LEFT, "Danger, reversing left")
                else:
                    return self._decide(Actions.STOP, "All directions blocked")

        # RULE 4: CAUTION ZONE (front 50-100cm)
        if front < self.caution_distance:
            # Check camera detection for directional hint
            if camera_detected:
                if camera_position == 0:  # left
                    return self._decide(Actions.TURN_RIGHT, "Caution, obstacle left")
                elif camera_position == 2:  # right
                    return self._decide(Actions.TURN_LEFT, "Caution, obstacle right")
                else:  # center
                    return self._decide(Actions.SLOW_DOWN, "Caution, obstacle center")
            return self._decide(Actions.SLOW_DOWN, f"Caution zone: front={front:.0f}cm")

        # RULE 5: CLEAR ZONE - Check side proximity
        if left < self.side_critical:
            return self._decide(Actions.TURN_RIGHT, f"Left side close: {left:.0f}cm")

        if right < self.side_critical:
            return self._decide(Actions.TURN_LEFT, f"Right side close: {right:.0f}cm")

        # RULE 6: Check heading error for correction
        if abs(heading_error) > self.HEADING_CORRECTION_THRESHOLD:
            if heading_error > 0:
                return self._decide(Actions.TURN_RIGHT, f"Correcting heading +{heading_error:.0f}°")
            else:
                return self._decide(Actions.TURN_LEFT, f"Correcting heading {heading_error:.0f}°")

        # RULE 7: Camera object in center path
        if camera_detected and camera_position == 1:  # center
            if camera_distance < self.caution_distance:
                return self._decide(Actions.SLOW_DOWN, "Camera: obstacle ahead")

        # DEFAULT: All clear - forward
        return self._decide(Actions.FORWARD, "Clear path")

    def _decide(self, action: int, reason: str) -> Tuple[int, float]:
        """
        Record decision and return result.

        Args:
            action: Action ID (0-6)
            reason: Explanation for the decision

        Returns:
            Tuple of (action_id, confidence=1.0)
        """
        self._action_counts[action] += 1

        logger.debug(f"Rule decision: {Actions.NAMES[action]} - {reason}")

        return (action, 1.0)

    def get_action_name(self, action_id: int) -> str:
        """Get action name from ID."""
        if 0 <= action_id < len(Actions.NAMES):
            return Actions.NAMES[action_id]
        return f"UNKNOWN({action_id})"

    def get_statistics(self) -> Dict[str, Any]:
        """Get rule engine statistics."""
        return {
            "total_decisions": self._decision_count,
            "action_distribution": {
                Actions.NAMES[k]: v for k, v in self._action_counts.items()
            },
        }

    def validate_against_ml(
        self,
        sensor_data: FusedSensorData,
        ml_action: int,
        ml_confidence: float,
    ) -> Dict[str, Any]:
        """
        Validate ML decision against rules.

        Useful for debugging and monitoring ML behavior.

        Args:
            sensor_data: Sensor readings
            ml_action: ML predicted action
            ml_confidence: ML confidence

        Returns:
            Dictionary with validation results.
        """
        rule_action, _ = self.decide(sensor_data)

        agreement = ml_action == rule_action

        return {
            "ml_action": Actions.NAMES[ml_action],
            "ml_confidence": ml_confidence,
            "rule_action": Actions.NAMES[rule_action],
            "agreement": agreement,
            "front_distance": sensor_data.front_min_distance,
        }


class HybridDecisionMaker:
    """
    Combines ML and rule-based decisions.

    Uses ML when confidence is high, falls back to rules otherwise.
    """

    def __init__(
        self,
        decision_engine,
        rule_engine: RuleEngine,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize hybrid decision maker.

        Args:
            decision_engine: DecisionEngine instance
            rule_engine: RuleEngine instance
            confidence_threshold: Threshold to trust ML
        """
        self._ml_engine = decision_engine
        self._rule_engine = rule_engine
        self._threshold = confidence_threshold

        self._ml_decisions = 0
        self._rule_decisions = 0

        logger.info(f"HybridDecisionMaker initialized, threshold={confidence_threshold}")

    def decide(self, sensor_data: FusedSensorData) -> Tuple[int, float, str]:
        """
        Make decision using ML with rule fallback.

        Args:
            sensor_data: Fused sensor readings

        Returns:
            Tuple of (action_id, confidence, source)
        """
        # Try ML first
        ml_action, ml_confidence = self._ml_engine.predict(sensor_data)

        if ml_confidence >= self._threshold:
            self._ml_decisions += 1
            return (ml_action, ml_confidence, "ML")
        else:
            # Fall back to rules
            rule_action, _ = self._rule_engine.decide(sensor_data)
            self._rule_decisions += 1

            logger.debug(
                f"ML confidence low ({ml_confidence:.2f}), using rules: "
                f"{self._rule_engine.get_action_name(rule_action)}"
            )

            return (rule_action, 1.0, "RULES")

    def get_statistics(self) -> Dict[str, Any]:
        """Get decision statistics."""
        total = self._ml_decisions + self._rule_decisions
        return {
            "ml_decisions": self._ml_decisions,
            "rule_decisions": self._rule_decisions,
            "ml_usage_rate": self._ml_decisions / max(1, total),
            "rule_fallback_rate": self._rule_decisions / max(1, total),
        }
