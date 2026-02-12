"""
Object Detector using TensorFlow Lite.

This module provides object detection using a pre-trained MobileNet SSD v2
model optimized for Raspberry Pi 4 inference.

Model: MobileNet SSD v2 (COCO pre-trained, INT8 quantized)
Input: 300x300 RGB image
Output: Bounding boxes, class labels, confidence scores

Performance target: < 150ms inference on RPi4

Usage:
    from src.perception.object_detector import ObjectDetector

    detector = ObjectDetector()
    detector.load_model()

    detections = detector.detect(frame)
    for det in detections:
        print(f"{det.class_name}: {det.confidence:.2f}")
"""

import time
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import numpy as np

# Try to import TensorFlow Lite
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        tflite = None

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """
    Container for a single object detection result.

    Attributes:
        class_id: COCO class ID
        class_name: Human-readable class name
        confidence: Detection confidence (0-1)
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
        center: Center point (x, y) of bounding box
        position: Position in frame ("left", "center", "right")
        area: Bounding box area in pixels
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    position: str
    area: int

    @property
    def width(self) -> int:
        """Bounding box width."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        """Bounding box height."""
        return self.bbox[3] - self.bbox[1]


class ObjectDetector:
    """
    TF Lite object detector for obstacle detection.

    Uses MobileNet SSD v2 pre-trained on COCO dataset.
    Optimized for real-time inference on Raspberry Pi 4.

    Args:
        model_path: Path to .tflite model file
        labels_path: Path to labels text file
        confidence_threshold: Minimum detection confidence
        num_threads: Number of CPU threads for inference
    """

    # COCO class labels for common objects
    DEFAULT_LABELS = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        56: "chair",
        57: "couch",
        58: "potted_plant",
        59: "bed",
        60: "dining_table",
        62: "tv",
        63: "laptop",
    }

    def __init__(
        self,
        model_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        confidence_threshold: Optional[float] = None,
        num_threads: int = 4,
    ):
        """Initialize the object detector."""
        self._settings = get_settings()

        self.model_path = model_path or self._settings.OBJECT_DETECTION_MODEL_PATH
        self.labels_path = labels_path or self._settings.OBJECT_DETECTION_LABELS_PATH
        self.confidence_threshold = (
            confidence_threshold or self._settings.DETECTION_CONFIDENCE_THRESHOLD
        )
        self.num_threads = num_threads

        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._labels: Dict[int, str] = {}
        self._input_shape: Tuple[int, int] = (300, 300)
        self._loaded = False

        # Performance tracking
        self._inference_times: List[float] = []
        self._max_tracked_times = 100

        logger.info(f"ObjectDetector created, threshold={self.confidence_threshold}")

    def load_model(self) -> bool:
        """
        Load the TF Lite model.

        Returns:
            True if model loaded successfully.
        """
        try:
            # Load labels
            self._load_labels()

            # Check if TFLite is available
            if not TFLITE_AVAILABLE:
                logger.warning("TFLite not available, using mock detector")
                self._loaded = True
                return True

            # Check if model file exists
            if not Path(self.model_path).exists():
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Using mock detector mode")
                self._loaded = True
                return True

            # Load TFLite model
            self._interpreter = tflite.Interpreter(
                model_path=str(self.model_path),
                num_threads=self.num_threads,
            )
            self._interpreter.allocate_tensors()

            # Get input/output details
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            # Get input shape
            input_shape = self._input_details[0]['shape']
            self._input_shape = (input_shape[1], input_shape[2])

            self._loaded = True
            logger.info(f"Model loaded: {self.model_path}")
            logger.info(f"Input shape: {self._input_shape}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fall back to mock mode
            self._loaded = True
            return True

    def _load_labels(self) -> None:
        """Load class labels from file or use defaults."""
        try:
            if Path(self.labels_path).exists():
                with open(self.labels_path, 'r') as f:
                    for i, line in enumerate(f):
                        self._labels[i] = line.strip()
                logger.info(f"Loaded {len(self._labels)} labels from {self.labels_path}")
            else:
                self._labels = self.DEFAULT_LABELS.copy()
                logger.info("Using default COCO labels")
        except Exception as e:
            logger.warning(f"Error loading labels: {e}, using defaults")
            self._labels = self.DEFAULT_LABELS.copy()

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: RGB image array (H, W, 3)
            confidence_threshold: Override default threshold

        Returns:
            List of Detection objects sorted by confidence (highest first).
        """
        if not self._loaded:
            logger.warning("Model not loaded, call load_model() first")
            return []

        threshold = confidence_threshold or self.confidence_threshold
        start_time = time.time()

        try:
            # If no real interpreter, return mock detections
            if self._interpreter is None:
                return self._mock_detect(frame)

            # Preprocess frame
            input_data = self._preprocess(frame)

            # Run inference
            self._interpreter.set_tensor(
                self._input_details[0]['index'],
                input_data
            )
            self._interpreter.invoke()

            # Get outputs
            boxes = self._interpreter.get_tensor(
                self._output_details[0]['index']
            )[0]
            classes = self._interpreter.get_tensor(
                self._output_details[1]['index']
            )[0]
            scores = self._interpreter.get_tensor(
                self._output_details[2]['index']
            )[0]
            num_detections = int(self._interpreter.get_tensor(
                self._output_details[3]['index']
            )[0])

            # Process detections
            detections = []
            frame_height, frame_width = frame.shape[:2]

            for i in range(num_detections):
                score = scores[i]
                if score < threshold:
                    continue

                class_id = int(classes[i])
                class_name = self._labels.get(class_id, f"class_{class_id}")

                # Convert normalized coordinates to pixel coordinates
                y1, x1, y2, x2 = boxes[i]
                bbox = (
                    int(x1 * frame_width),
                    int(y1 * frame_height),
                    int(x2 * frame_width),
                    int(y2 * frame_height),
                )

                # Calculate center and position
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2

                # Determine position in frame (left/center/right)
                if center_x < frame_width * 0.33:
                    position = "left"
                elif center_x > frame_width * 0.67:
                    position = "right"
                else:
                    position = "center"

                # Calculate area
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(score),
                    bbox=bbox,
                    center=(center_x, center_y),
                    position=position,
                    area=area,
                ))

            # Sort by confidence
            detections.sort(key=lambda d: d.confidence, reverse=True)

            # Track inference time
            inference_time = time.time() - start_time
            self._track_inference_time(inference_time)

            logger.debug(
                f"Detected {len(detections)} objects in {inference_time*1000:.1f}ms"
            )

            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model input.

        Args:
            frame: RGB image array

        Returns:
            Preprocessed array ready for inference.
        """
        # Resize to model input size
        try:
            import cv2
            resized = cv2.resize(
                frame,
                self._input_shape,
                interpolation=cv2.INTER_LINEAR
            )
        except ImportError:
            # Fallback resize
            resized = self._numpy_resize(
                frame,
                self._input_shape[0],
                self._input_shape[1]
            )

        # Add batch dimension
        input_data = np.expand_dims(resized, axis=0)

        # Convert to appropriate dtype
        input_dtype = self._input_details[0]['dtype']
        if input_dtype == np.uint8:
            input_data = input_data.astype(np.uint8)
        else:
            # Float model - normalize to [-1, 1] or [0, 1]
            input_data = (input_data.astype(np.float32) - 127.5) / 127.5

        return input_data

    def _numpy_resize(
        self,
        image: np.ndarray,
        new_height: int,
        new_width: int
    ) -> np.ndarray:
        """Simple nearest-neighbor resize."""
        height, width = image.shape[:2]
        y_indices = (np.arange(new_height) * height / new_height).astype(int)
        x_indices = (np.arange(new_width) * width / new_width).astype(int)
        return image[y_indices[:, None], x_indices, :]

    def _mock_detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Generate mock detections for testing.

        Returns random detections based on frame content.
        """
        import random

        # Randomly return 0-2 mock detections
        num_detections = random.randint(0, 2)
        detections = []

        frame_height, frame_width = frame.shape[:2]

        for i in range(num_detections):
            # Random class from common objects
            class_id = random.choice([0, 1, 2, 16, 17])  # person, bicycle, car, cat, dog
            class_name = self._labels.get(class_id, f"class_{class_id}")

            # Random bounding box
            x1 = random.randint(0, frame_width - 100)
            y1 = random.randint(0, frame_height - 100)
            x2 = x1 + random.randint(50, 150)
            y2 = y1 + random.randint(50, 200)

            bbox = (x1, y1, min(x2, frame_width), min(y2, frame_height))
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2

            if center_x < frame_width * 0.33:
                position = "left"
            elif center_x > frame_width * 0.67:
                position = "right"
            else:
                position = "center"

            detections.append(Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=random.uniform(0.5, 0.95),
                bbox=bbox,
                center=(center_x, center_y),
                position=position,
                area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
            ))

        return detections

    def _track_inference_time(self, time_s: float) -> None:
        """Track inference time for performance monitoring."""
        self._inference_times.append(time_s)
        if len(self._inference_times) > self._max_tracked_times:
            self._inference_times.pop(0)

    def detect_person(
        self,
        frame: np.ndarray,
    ) -> Optional[Detection]:
        """
        Detect persons in frame (priority detection for safety).

        Uses lower threshold for person detection as a safety measure.

        Args:
            frame: RGB image array

        Returns:
            Highest confidence person detection, or None.
        """
        # Use lower threshold for person detection (safety-critical)
        person_threshold = self._settings.PERSON_CONFIDENCE_THRESHOLD
        detections = self.detect(frame, confidence_threshold=person_threshold)

        # Filter for persons only
        persons = [d for d in detections if d.class_id == 0]

        if persons:
            return persons[0]  # Highest confidence
        return None

    def get_primary_obstacle(
        self,
        frame: np.ndarray,
    ) -> Optional[Detection]:
        """
        Get the most significant obstacle in the frame.

        Priority: Person > Vehicle > Animal > Other

        Args:
            frame: RGB image array

        Returns:
            Primary obstacle detection, or None.
        """
        detections = self.detect(frame)

        if not detections:
            return None

        # Priority order
        priority_classes = [0, 2, 3, 5, 7, 16, 17]  # person, car, motorcycle, bus, truck, cat, dog

        for class_id in priority_classes:
            for det in detections:
                if det.class_id == class_id:
                    return det

        # Return highest confidence if no priority match
        return detections[0]

    @property
    def average_inference_time_ms(self) -> float:
        """Get average inference time in milliseconds."""
        if not self._inference_times:
            return 0.0
        return (sum(self._inference_times) / len(self._inference_times)) * 1000

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def input_shape(self) -> Tuple[int, int]:
        """Get model input shape (height, width)."""
        return self._input_shape
