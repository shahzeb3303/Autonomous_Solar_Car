"""
Sensor Fusion Module for Autonomous Solar Vehicle.

This module combines data from multiple sensors into a unified
feature vector for the ML decision model:
- 5 ultrasonic sensors (distance measurements)
- Camera object detection (class, distance, position)
- Vehicle state (current speed, heading error)

The fused data format matches the ML model's expected input format.

Feature Vector (12 elements):
    0. front_distance - US_FRONT reading (cm)
    1. front_wp_distance - US_FRONT_WP reading (cm)
    2. right_distance - US_RIGHT reading (cm)
    3. left_distance - US_LEFT reading (cm)
    4. rear_distance - US_REAR reading (cm)
    5. front_min_distance - min(front, front_wp)
    6. camera_object_detected - boolean (0/1)
    7. camera_object_class - encoded class ID
    8. camera_object_distance_estimate - estimated distance (cm)
    9. camera_object_position - position encoding (0=left, 1=center, 2=right)
    10. current_speed - PWM duty cycle percentage
    11. heading_error - deviation from planned path (degrees)

Usage:
    from src.sensors.sensor_fusion import SensorFusion

    fusion = SensorFusion(sensor_array, camera_manager, object_detector)
    fusion.initialize()

    feature_vector = fusion.get_feature_vector(current_speed=50)
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import numpy as np

from src.sensors.ultrasonic import UltrasonicSensorArray
from src.sensors.camera import CameraManager
from src.perception.object_detector import ObjectDetector, Detection
from src.perception.distance_estimator import DistanceEstimator
from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class FusedSensorData:
    """
    Container for fused sensor data.

    All distance measurements are in centimeters.
    All speeds are in percentage (0-100).
    All angles are in degrees.
    """
    # Ultrasonic distances
    front_distance: float = 0.0
    front_wp_distance: float = 0.0
    right_distance: float = 0.0
    left_distance: float = 0.0
    rear_distance: float = 0.0
    front_min_distance: float = 0.0

    # Camera detection
    camera_object_detected: int = 0
    camera_object_class: int = 0
    camera_object_distance_estimate: float = 0.0
    camera_object_position: int = 1  # Default center

    # Vehicle state
    current_speed: int = 0
    heading_error: float = 0.0

    # Metadata
    timestamp: float = field(default_factory=time.time)
    is_valid: bool = True

    def to_feature_vector(self) -> np.ndarray:
        """Convert to numpy array for ML model input."""
        return np.array([
            self.front_distance,
            self.front_wp_distance,
            self.right_distance,
            self.left_distance,
            self.rear_distance,
            self.front_min_distance,
            self.camera_object_detected,
            self.camera_object_class,
            self.camera_object_distance_estimate,
            self.camera_object_position,
            self.current_speed,
            self.heading_error,
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            "front_distance": self.front_distance,
            "front_wp_distance": self.front_wp_distance,
            "right_distance": self.right_distance,
            "left_distance": self.left_distance,
            "rear_distance": self.rear_distance,
            "front_min_distance": self.front_min_distance,
            "camera_object_detected": float(self.camera_object_detected),
            "camera_object_class": float(self.camera_object_class),
            "camera_object_distance_estimate": self.camera_object_distance_estimate,
            "camera_object_position": float(self.camera_object_position),
            "current_speed": float(self.current_speed),
            "heading_error": self.heading_error,
        }

    def to_normalized_vector(self, max_distance: float = 400.0) -> np.ndarray:
        """
        Convert to normalized feature vector (0-1 range).

        This is typically what the ML model expects.

        Args:
            max_distance: Maximum distance for normalization (cm)
        """
        return np.array([
            min(self.front_distance / max_distance, 1.0),
            min(self.front_wp_distance / max_distance, 1.0),
            min(self.right_distance / max_distance, 1.0),
            min(self.left_distance / max_distance, 1.0),
            min(self.rear_distance / max_distance, 1.0),
            min(self.front_min_distance / max_distance, 1.0),
            float(self.camera_object_detected),
            self.camera_object_class / 10.0,  # Normalize class ID
            min(self.camera_object_distance_estimate / max_distance, 1.0),
            self.camera_object_position / 2.0,  # 0, 0.5, or 1.0
            self.current_speed / 100.0,
            (self.heading_error + 90.0) / 180.0,  # Normalize -90 to 90 => 0 to 1
        ], dtype=np.float32)


class SensorFusion:
    """
    Fuses data from all sensors into a unified feature vector.

    Coordinates reading from ultrasonic sensors and camera,
    processes detections, and produces the feature vector
    expected by the decision ML model.

    Args:
        sensor_array: UltrasonicSensorArray instance
        camera_manager: CameraManager instance (optional)
        object_detector: ObjectDetector instance (optional)
        distance_estimator: DistanceEstimator instance (optional)
    """

    # Position encoding map
    POSITION_ENCODING = {
        "left": 0,
        "center": 1,
        "right": 2,
    }

    # Class encoding map (simplified)
    CLASS_ENCODING = {
        "none": 0,
        "person": 1,
        "bicycle": 2,
        "car": 2,
        "motorcycle": 2,
        "bus": 2,
        "truck": 2,
        "dog": 3,
        "cat": 3,
        "bird": 3,
        "chair": 4,
        "potted_plant": 4,
        "couch": 4,
        "dining_table": 4,
    }

    def __init__(
        self,
        sensor_array: Optional[UltrasonicSensorArray] = None,
        camera_manager: Optional[CameraManager] = None,
        object_detector: Optional[ObjectDetector] = None,
        distance_estimator: Optional[DistanceEstimator] = None,
    ):
        """Initialize the sensor fusion module."""
        self._settings = get_settings()

        self._sensor_array = sensor_array
        self._camera_manager = camera_manager
        self._object_detector = object_detector
        self._distance_estimator = distance_estimator or DistanceEstimator()

        self._initialized = False
        self._lock = threading.Lock()

        # Latest readings cache
        self._latest_ultrasonic: Dict[str, float] = {}
        self._latest_detection: Optional[Detection] = None
        self._latest_fused_data: Optional[FusedSensorData] = None

        # Performance tracking
        self._fusion_times: List[float] = []

        logger.info("SensorFusion created")

    def initialize(self) -> bool:
        """
        Initialize all sensors.

        Returns:
            True if initialization successful.
        """
        try:
            with self._lock:
                if self._initialized:
                    return True

                # Initialize ultrasonic sensors
                if self._sensor_array:
                    results = self._sensor_array.initialize_all()
                    if not all(results.values()):
                        logger.warning("Some ultrasonic sensors failed to initialize")

                # Initialize camera
                if self._camera_manager:
                    results = self._camera_manager.start()
                    if not any(results.values()):
                        logger.warning("No cameras could be started")

                # Load object detection model
                if self._object_detector:
                    if not self._object_detector.load_model():
                        logger.warning("Object detection model failed to load")

                self._initialized = True
                logger.info("SensorFusion initialized")
                return True

        except Exception as e:
            logger.error(f"SensorFusion initialization failed: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up all sensors."""
        with self._lock:
            if self._sensor_array:
                self._sensor_array.cleanup_all()

            if self._camera_manager:
                self._camera_manager.stop()

            self._initialized = False
            logger.info("SensorFusion cleaned up")

    def read_ultrasonic(self) -> Dict[str, float]:
        """
        Read all ultrasonic sensors.

        Returns:
            Dictionary mapping sensor_id to distance in cm.
        """
        if not self._sensor_array:
            return self._get_mock_ultrasonic()

        distances = self._sensor_array.read_all_distances()
        self._latest_ultrasonic = distances
        return distances

    def _get_mock_ultrasonic(self) -> Dict[str, float]:
        """Generate mock ultrasonic readings for testing."""
        import random
        return {
            "US_FRONT": random.uniform(50, 200),
            "US_FRONT_WP": random.uniform(50, 200),
            "US_RIGHT": random.uniform(30, 150),
            "US_LEFT": random.uniform(30, 150),
            "US_REAR": random.uniform(100, 300),
        }

    def read_camera_detection(self) -> Optional[Detection]:
        """
        Read camera and detect objects.

        Returns:
            Primary obstacle detection, or None if none detected.
        """
        if not self._camera_manager or not self._object_detector:
            return None

        frame = self._camera_manager.get_front_frame_skipped()
        if frame is None:
            return self._latest_detection  # Return cached

        detection = self._object_detector.get_primary_obstacle(frame)
        self._latest_detection = detection
        return detection

    def get_feature_vector(
        self,
        current_speed: int = 0,
        heading_error: float = 0.0,
    ) -> np.ndarray:
        """
        Get the fused feature vector for ML model input.

        Args:
            current_speed: Current vehicle speed (0-100)
            heading_error: Heading deviation in degrees

        Returns:
            Normalized feature vector (12 elements).
        """
        fused = self.get_fused_data(current_speed, heading_error)
        return fused.to_normalized_vector()

    def get_fused_data(
        self,
        current_speed: int = 0,
        heading_error: float = 0.0,
    ) -> FusedSensorData:
        """
        Get complete fused sensor data.

        Args:
            current_speed: Current vehicle speed (0-100)
            heading_error: Heading deviation in degrees

        Returns:
            FusedSensorData with all sensor readings.
        """
        start_time = time.time()

        try:
            # Read ultrasonic sensors
            ultrasonic = self.read_ultrasonic()

            # Read camera detection
            detection = self.read_camera_detection()

            # Build fused data
            fused = FusedSensorData(
                front_distance=ultrasonic.get("US_FRONT", 0.0),
                front_wp_distance=ultrasonic.get("US_FRONT_WP", 0.0),
                right_distance=ultrasonic.get("US_RIGHT", 0.0),
                left_distance=ultrasonic.get("US_LEFT", 0.0),
                rear_distance=ultrasonic.get("US_REAR", 0.0),
                current_speed=current_speed,
                heading_error=heading_error,
                timestamp=time.time(),
            )

            # Calculate minimum front distance
            fused.front_min_distance = min(
                fused.front_distance,
                fused.front_wp_distance
            )

            # Add camera detection data
            if detection:
                fused.camera_object_detected = 1
                fused.camera_object_class = self._encode_class(detection.class_name)
                fused.camera_object_position = self.POSITION_ENCODING.get(
                    detection.position, 1
                )

                # Estimate distance from bounding box
                distance_est = self._distance_estimator.estimate_distance(detection)
                fused.camera_object_distance_estimate = distance_est.distance_cm

            # Validate data
            fused.is_valid = self._validate_fused_data(fused)

            self._latest_fused_data = fused

            # Track fusion time
            fusion_time = time.time() - start_time
            self._fusion_times.append(fusion_time)
            if len(self._fusion_times) > 100:
                self._fusion_times.pop(0)

            logger.debug(f"Sensor fusion complete in {fusion_time*1000:.1f}ms")

            return fused

        except Exception as e:
            logger.error(f"Sensor fusion error: {e}")
            return FusedSensorData(is_valid=False)

    def _encode_class(self, class_name: str) -> int:
        """Encode class name to numeric ID."""
        return self.CLASS_ENCODING.get(class_name.lower(), 4)

    def _validate_fused_data(self, data: FusedSensorData) -> bool:
        """
        Validate fused sensor data.

        Checks for sensor errors and impossible readings.
        """
        # Check for all-zero ultrasonic (likely sensor failure)
        if (data.front_distance == 0 and
            data.front_wp_distance == 0 and
            data.right_distance == 0 and
            data.left_distance == 0):
            logger.warning("All ultrasonic sensors reading zero - possible failure")
            return False

        # Check for negative distances (impossible)
        if any([
            data.front_distance < 0,
            data.front_wp_distance < 0,
            data.right_distance < 0,
            data.left_distance < 0,
            data.rear_distance < 0,
        ]):
            logger.warning("Negative distance reading - sensor error")
            return False

        return True

    def get_obstacle_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current obstacle situation.

        Returns:
            Dictionary with obstacle summary.
        """
        fused = self._latest_fused_data
        if fused is None:
            fused = self.get_fused_data()

        settings = self._settings

        summary = {
            "front_clear": fused.front_min_distance > settings.CLEAR_DISTANCE_CM,
            "front_caution": settings.CAUTION_DISTANCE_CM < fused.front_min_distance <= settings.CLEAR_DISTANCE_CM,
            "front_danger": settings.DANGER_DISTANCE_CM < fused.front_min_distance <= settings.CAUTION_DISTANCE_CM,
            "front_critical": fused.front_min_distance <= settings.CRITICAL_DISTANCE_CM,
            "right_close": fused.right_distance < settings.SIDE_CRITICAL_CM,
            "left_close": fused.left_distance < settings.SIDE_CRITICAL_CM,
            "rear_close": fused.rear_distance < settings.CRITICAL_DISTANCE_CM,
            "camera_detected": fused.camera_object_detected == 1,
            "camera_class": fused.camera_object_class,
            "min_front_distance": fused.front_min_distance,
        }

        return summary

    def requires_stop(self) -> Tuple[bool, str]:
        """
        Check if current sensor data requires immediate stop.

        Returns:
            Tuple of (should_stop, reason).
        """
        fused = self._latest_fused_data
        if fused is None:
            fused = self.get_fused_data()

        settings = self._settings

        # Check critical front distance
        if fused.front_min_distance < settings.CRITICAL_DISTANCE_CM:
            return True, f"Front obstacle at {fused.front_min_distance:.0f}cm"

        # Check for person detected
        if fused.camera_object_detected and fused.camera_object_class == 1:
            if fused.camera_object_distance_estimate < settings.HIGH_DISTANCE:
                return True, f"Person detected at {fused.camera_object_distance_estimate:.0f}cm"

        return False, ""

    @property
    def latest_fused_data(self) -> Optional[FusedSensorData]:
        """Get the most recent fused data."""
        return self._latest_fused_data

    @property
    def average_fusion_time_ms(self) -> float:
        """Get average fusion time in milliseconds."""
        if not self._fusion_times:
            return 0.0
        return (sum(self._fusion_times) / len(self._fusion_times)) * 1000

    @property
    def is_initialized(self) -> bool:
        """Check if fusion module is initialized."""
        return self._initialized
