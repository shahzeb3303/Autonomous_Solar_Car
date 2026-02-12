"""
Distance Estimator from Bounding Box Size.

This module estimates the distance to detected objects based on
their bounding box size and known reference dimensions.

Method: Pinhole camera model
    distance = (known_height * focal_length) / bbox_height

This is an approximation and works best for objects of known size
at moderate distances (1-10 meters).

Usage:
    from src.perception.distance_estimator import DistanceEstimator
    from src.perception.object_detector import Detection

    estimator = DistanceEstimator()
    distance = estimator.estimate_distance(detection)
"""

from typing import Optional, Dict
from dataclasses import dataclass
import logging
import math

from src.perception.object_detector import Detection
from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class DistanceEstimate:
    """
    Container for distance estimation result.

    Attributes:
        distance_cm: Estimated distance in centimeters
        confidence: Confidence in estimate (0-1)
        method: Estimation method used
        is_reliable: Whether estimate is considered reliable
    """
    distance_cm: float
    confidence: float
    method: str
    is_reliable: bool


class DistanceEstimator:
    """
    Estimate distance to detected objects using camera geometry.

    Uses the pinhole camera model to estimate distance based on
    the apparent size of detected objects.

    Args:
        focal_length_px: Camera focal length in pixels
        frame_height: Camera frame height in pixels
    """

    # Reference heights for common objects (in cm)
    # These are approximate average heights used for estimation
    DEFAULT_REFERENCE_HEIGHTS: Dict[str, float] = {
        "person": 170.0,
        "bicycle": 100.0,
        "car": 150.0,
        "motorcycle": 120.0,
        "bus": 300.0,
        "truck": 250.0,
        "dog": 60.0,
        "cat": 30.0,
        "chair": 80.0,
        "potted_plant": 50.0,
        "couch": 90.0,
        "dining_table": 75.0,
    }

    # Minimum reliable bbox height (below this, estimate is unreliable)
    MIN_RELIABLE_BBOX_HEIGHT = 30  # pixels

    # Maximum reliable distance (beyond this, estimate is unreliable)
    MAX_RELIABLE_DISTANCE_CM = 1000  # 10 meters

    def __init__(
        self,
        focal_length_px: Optional[float] = None,
        frame_height: int = 480,
    ):
        """Initialize the distance estimator."""
        self._settings = get_settings()

        self.focal_length_px = focal_length_px or self._settings.CAMERA_FOCAL_LENGTH_PX
        self.frame_height = frame_height

        # Use settings reference heights if available, else defaults
        self.reference_heights = (
            self._settings.REFERENCE_HEIGHTS_CM
            if hasattr(self._settings, 'REFERENCE_HEIGHTS_CM')
            else self.DEFAULT_REFERENCE_HEIGHTS
        )

        logger.info(
            f"DistanceEstimator initialized: focal_length={self.focal_length_px}px"
        )

    def estimate_distance(
        self,
        detection: Detection,
        reference_height_cm: Optional[float] = None,
    ) -> DistanceEstimate:
        """
        Estimate distance to a detected object.

        Uses pinhole camera model:
            distance = (real_height * focal_length) / image_height

        Args:
            detection: Detection object with bounding box
            reference_height_cm: Override default reference height

        Returns:
            DistanceEstimate with distance and confidence.
        """
        # Get bounding box height
        bbox_height = detection.height

        if bbox_height <= 0:
            return DistanceEstimate(
                distance_cm=0.0,
                confidence=0.0,
                method="bbox_height",
                is_reliable=False,
            )

        # Get reference height for this object class
        if reference_height_cm is None:
            reference_height_cm = self.reference_heights.get(
                detection.class_name,
                100.0  # Default fallback height
            )

        # Calculate distance using pinhole model
        distance_cm = (reference_height_cm * self.focal_length_px) / bbox_height

        # Calculate confidence based on bbox size and distance
        confidence = self._calculate_confidence(bbox_height, distance_cm)

        # Determine reliability
        is_reliable = (
            bbox_height >= self.MIN_RELIABLE_BBOX_HEIGHT and
            distance_cm <= self.MAX_RELIABLE_DISTANCE_CM and
            confidence > 0.5
        )

        return DistanceEstimate(
            distance_cm=distance_cm,
            confidence=confidence,
            method="pinhole_model",
            is_reliable=is_reliable,
        )

    def _calculate_confidence(
        self,
        bbox_height: int,
        distance_cm: float,
    ) -> float:
        """
        Calculate confidence in distance estimate.

        Confidence decreases with:
        - Very small bounding boxes (far objects)
        - Very large distances
        - Very close objects (may be partially visible)

        Args:
            bbox_height: Bounding box height in pixels
            distance_cm: Estimated distance in cm

        Returns:
            Confidence score (0-1)
        """
        # Penalize very small bboxes
        if bbox_height < 20:
            size_factor = bbox_height / 20.0
        elif bbox_height > self.frame_height * 0.8:
            # Object filling most of frame - may be partially visible
            size_factor = 0.7
        else:
            size_factor = 1.0

        # Penalize very far distances
        if distance_cm > 500:
            distance_factor = max(0.3, 1.0 - (distance_cm - 500) / 1000)
        else:
            distance_factor = 1.0

        confidence = size_factor * distance_factor * 0.9  # Max 90% confidence

        return min(1.0, max(0.0, confidence))

    def estimate_distance_simple(
        self,
        bbox_height: int,
        class_name: str = "person",
    ) -> float:
        """
        Simple distance estimation without full Detection object.

        Args:
            bbox_height: Bounding box height in pixels
            class_name: Object class name for reference height

        Returns:
            Estimated distance in cm (0 if invalid).
        """
        if bbox_height <= 0:
            return 0.0

        reference_height = self.reference_heights.get(class_name, 100.0)
        return (reference_height * self.focal_length_px) / bbox_height

    def calibrate_focal_length(
        self,
        known_distance_cm: float,
        bbox_height: int,
        reference_height_cm: float,
    ) -> float:
        """
        Calibrate focal length using known distance measurement.

        Use this to improve accuracy by calibrating with a real
        measurement.

        Args:
            known_distance_cm: Actual measured distance to object
            bbox_height: Bounding box height at that distance
            reference_height_cm: Actual height of the object

        Returns:
            Calibrated focal length in pixels.
        """
        # focal_length = (distance * bbox_height) / real_height
        new_focal_length = (known_distance_cm * bbox_height) / reference_height_cm

        logger.info(
            f"Focal length calibrated: {self.focal_length_px:.1f} -> {new_focal_length:.1f}px"
        )

        self.focal_length_px = new_focal_length
        return new_focal_length

    def estimate_relative_distance(
        self,
        detection: Detection,
    ) -> str:
        """
        Get a qualitative distance estimate.

        Useful when precise distance isn't critical.

        Args:
            detection: Detection object

        Returns:
            Qualitative distance: "very_close", "close", "medium", "far"
        """
        estimate = self.estimate_distance(detection)

        if estimate.distance_cm < 100:
            return "very_close"
        elif estimate.distance_cm < 200:
            return "close"
        elif estimate.distance_cm < 400:
            return "medium"
        else:
            return "far"

    def get_closest_object(
        self,
        detections: list,
    ) -> Optional[tuple]:
        """
        Find the closest detected object.

        Args:
            detections: List of Detection objects

        Returns:
            Tuple of (detection, distance_estimate), or None if empty.
        """
        if not detections:
            return None

        closest = None
        min_distance = float('inf')

        for det in detections:
            estimate = self.estimate_distance(det)
            if estimate.distance_cm < min_distance:
                min_distance = estimate.distance_cm
                closest = (det, estimate)

        return closest

    def update_reference_height(
        self,
        class_name: str,
        height_cm: float,
    ) -> None:
        """
        Update reference height for a class.

        Args:
            class_name: Object class name
            height_cm: New reference height in cm
        """
        self.reference_heights[class_name] = height_cm
        logger.info(f"Updated reference height: {class_name} = {height_cm}cm")
