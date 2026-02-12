"""
Threat Assessor for Detected Objects.

This module classifies detected objects by their threat level
to vehicle navigation, combining object class, distance, and position.

Threat Levels:
- CRITICAL: Requires immediate stop (person, very close obstacle)
- HIGH: Requires immediate action (close obstacle, moving vehicle)
- MEDIUM: Requires caution (moderate distance obstacle)
- LOW: Awareness only (far obstacle, static object)
- NONE: No threat (very far or irrelevant objects)

Usage:
    from src.perception.threat_assessor import ThreatAssessor
    from src.perception.object_detector import ObjectDetector, Detection
    from src.perception.distance_estimator import DistanceEstimator

    detector = ObjectDetector()
    estimator = DistanceEstimator()
    assessor = ThreatAssessor(estimator)

    detections = detector.detect(frame)
    threats = assessor.assess_all(detections)

    for threat in threats:
        if threat.level == ThreatLevel.CRITICAL:
            print("STOP!")
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

from src.perception.object_detector import Detection
from src.perception.distance_estimator import DistanceEstimator, DistanceEstimate
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat level classification."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other):
        if isinstance(other, ThreatLevel):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, ThreatLevel):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, ThreatLevel):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, ThreatLevel):
            return self.value >= other.value
        return NotImplemented


@dataclass
class ThreatAssessment:
    """
    Container for threat assessment result.

    Attributes:
        detection: Original detection
        level: Assessed threat level
        distance_estimate: Distance estimation
        recommended_action: Suggested action to take
        reasoning: Explanation for the assessment
    """
    detection: Detection
    level: ThreatLevel
    distance_estimate: DistanceEstimate
    recommended_action: str
    reasoning: str


class ThreatAssessor:
    """
    Assess threat level of detected objects.

    Combines multiple factors to determine threat:
    - Object class (person > vehicle > animal > static)
    - Distance (closer = higher threat)
    - Position (center > sides)
    - Object size (larger apparent size = closer = higher threat)

    Args:
        distance_estimator: DistanceEstimator instance
    """

    # Class threat categories
    # Higher number = higher base threat
    CLASS_THREAT_WEIGHTS: Dict[str, int] = {
        "person": 5,      # Always highest priority
        "bicycle": 4,
        "motorcycle": 4,
        "car": 3,
        "bus": 3,
        "truck": 3,
        "dog": 3,
        "cat": 2,
        "bird": 1,
        "chair": 2,
        "potted_plant": 1,
        "couch": 2,
        "dining_table": 2,
    }

    # Position threat multipliers
    POSITION_MULTIPLIERS: Dict[str, float] = {
        "center": 1.0,    # Full threat
        "left": 0.7,      # Partial threat
        "right": 0.7,     # Partial threat
    }

    # Distance thresholds (cm)
    CRITICAL_DISTANCE = 50
    HIGH_DISTANCE = 100
    MEDIUM_DISTANCE = 200
    LOW_DISTANCE = 400

    def __init__(self, distance_estimator: Optional[DistanceEstimator] = None):
        """Initialize the threat assessor."""
        self._settings = get_settings()
        self._distance_estimator = distance_estimator or DistanceEstimator()

        logger.info("ThreatAssessor initialized")

    def assess(self, detection: Detection) -> ThreatAssessment:
        """
        Assess threat level of a single detection.

        Args:
            detection: Detection object to assess

        Returns:
            ThreatAssessment with level and recommendations.
        """
        # Estimate distance
        distance_estimate = self._distance_estimator.estimate_distance(detection)
        distance_cm = distance_estimate.distance_cm

        # Get base threat from class
        class_weight = self.CLASS_THREAT_WEIGHTS.get(
            detection.class_name, 1
        )

        # Apply position multiplier
        position_mult = self.POSITION_MULTIPLIERS.get(
            detection.position, 0.7
        )

        # Calculate threat score (0-10 scale)
        # Closer = higher threat, scaled inversely with distance
        if distance_cm <= 0:
            distance_score = 10
        elif distance_cm < self.CRITICAL_DISTANCE:
            distance_score = 10
        elif distance_cm < self.HIGH_DISTANCE:
            distance_score = 8
        elif distance_cm < self.MEDIUM_DISTANCE:
            distance_score = 5
        elif distance_cm < self.LOW_DISTANCE:
            distance_score = 3
        else:
            distance_score = 1

        # Combine factors
        total_score = class_weight * position_mult * (distance_score / 10)

        # Special case: Person always critical if close
        if detection.class_name == "person" and distance_cm < self.HIGH_DISTANCE:
            level = ThreatLevel.CRITICAL
            recommended_action = "STOP"
            reasoning = f"Person detected at {distance_cm:.0f}cm - safety stop required"

        # Determine threat level from score
        elif total_score >= 4.0 or distance_cm < self.CRITICAL_DISTANCE:
            level = ThreatLevel.CRITICAL
            recommended_action = "STOP"
            reasoning = f"{detection.class_name} at critical distance ({distance_cm:.0f}cm)"

        elif total_score >= 3.0 or distance_cm < self.HIGH_DISTANCE:
            level = ThreatLevel.HIGH
            if detection.position == "left":
                recommended_action = "TURN_RIGHT"
            elif detection.position == "right":
                recommended_action = "TURN_LEFT"
            else:
                recommended_action = "SLOW_DOWN"
            reasoning = f"{detection.class_name} at high threat distance ({distance_cm:.0f}cm)"

        elif total_score >= 2.0 or distance_cm < self.MEDIUM_DISTANCE:
            level = ThreatLevel.MEDIUM
            recommended_action = "SLOW_DOWN"
            reasoning = f"{detection.class_name} at medium distance ({distance_cm:.0f}cm)"

        elif total_score >= 1.0 or distance_cm < self.LOW_DISTANCE:
            level = ThreatLevel.LOW
            recommended_action = "MONITOR"
            reasoning = f"{detection.class_name} detected but far ({distance_cm:.0f}cm)"

        else:
            level = ThreatLevel.NONE
            recommended_action = "NONE"
            reasoning = f"{detection.class_name} too far to be a threat ({distance_cm:.0f}cm)"

        return ThreatAssessment(
            detection=detection,
            level=level,
            distance_estimate=distance_estimate,
            recommended_action=recommended_action,
            reasoning=reasoning,
        )

    def assess_all(
        self,
        detections: List[Detection],
    ) -> List[ThreatAssessment]:
        """
        Assess all detections and sort by threat level.

        Args:
            detections: List of Detection objects

        Returns:
            List of ThreatAssessments sorted by threat level (highest first).
        """
        assessments = [self.assess(d) for d in detections]

        # Sort by threat level (highest first)
        assessments.sort(key=lambda a: a.level.value, reverse=True)

        return assessments

    def get_highest_threat(
        self,
        detections: List[Detection],
    ) -> Optional[ThreatAssessment]:
        """
        Get the highest threat from detections.

        Args:
            detections: List of Detection objects

        Returns:
            Highest ThreatAssessment, or None if no detections.
        """
        if not detections:
            return None

        assessments = self.assess_all(detections)
        return assessments[0] if assessments else None

    def requires_stop(self, detections: List[Detection]) -> bool:
        """
        Check if any detection requires immediate stop.

        Args:
            detections: List of Detection objects

        Returns:
            True if any threat is CRITICAL level.
        """
        for det in detections:
            assessment = self.assess(det)
            if assessment.level == ThreatLevel.CRITICAL:
                return True
        return False

    def requires_action(self, detections: List[Detection]) -> bool:
        """
        Check if any detection requires navigation action.

        Args:
            detections: List of Detection objects

        Returns:
            True if any threat is HIGH level or above.
        """
        for det in detections:
            assessment = self.assess(det)
            if assessment.level >= ThreatLevel.HIGH:
                return True
        return False

    def get_recommended_action(
        self,
        detections: List[Detection],
    ) -> str:
        """
        Get overall recommended action based on all detections.

        Args:
            detections: List of Detection objects

        Returns:
            Recommended action string.
        """
        if not detections:
            return "FORWARD"

        highest_threat = self.get_highest_threat(detections)
        if highest_threat is None:
            return "FORWARD"

        return highest_threat.recommended_action

    def summarize_threats(
        self,
        detections: List[Detection],
    ) -> Dict[str, Any]:
        """
        Generate a summary of all threats.

        Args:
            detections: List of Detection objects

        Returns:
            Dictionary with threat summary.
        """
        assessments = self.assess_all(detections)

        summary = {
            "total_objects": len(detections),
            "critical_count": sum(1 for a in assessments if a.level == ThreatLevel.CRITICAL),
            "high_count": sum(1 for a in assessments if a.level == ThreatLevel.HIGH),
            "medium_count": sum(1 for a in assessments if a.level == ThreatLevel.MEDIUM),
            "low_count": sum(1 for a in assessments if a.level == ThreatLevel.LOW),
            "requires_stop": any(a.level == ThreatLevel.CRITICAL for a in assessments),
            "requires_action": any(a.level >= ThreatLevel.HIGH for a in assessments),
            "recommended_action": self.get_recommended_action(detections),
        }

        if assessments:
            highest = assessments[0]
            summary["highest_threat"] = {
                "class": highest.detection.class_name,
                "level": highest.level.name,
                "distance_cm": highest.distance_estimate.distance_cm,
                "action": highest.recommended_action,
            }

        return summary
