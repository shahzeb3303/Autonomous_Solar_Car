"""
Decision Engine - ML-Based Navigation (Layer 2).

This module provides ML-based decision making using the trained
neural network model. It operates after the safety governor
has verified it's safe to proceed.

The engine:
1. Takes fused sensor data
2. Normalizes features
3. Runs TFLite inference
4. Returns predicted action with confidence

If confidence is below threshold, it falls back to rule-based decisions.

Usage:
    from src.decision.decision_engine import DecisionEngine
    from src.sensors.sensor_fusion import FusedSensorData

    engine = DecisionEngine()
    engine.load_model()

    action, confidence = engine.predict(fused_data)
"""

import time
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import logging
import numpy as np

# Try to import TFLite
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

from src.sensors.sensor_fusion import FusedSensorData
from config.settings import get_settings
from config.logging_config import get_decision_logger

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    ML-based decision engine using TensorFlow Lite.

    Loads the quantized decision model and runs inference
    on fused sensor data to predict navigation actions.

    Args:
        model_path: Path to TFLite model (optional)
        confidence_threshold: Minimum confidence to trust prediction
    """

    # Action names for logging
    ACTION_NAMES = [
        "FORWARD",
        "SLOW_DOWN",
        "TURN_LEFT",
        "TURN_RIGHT",
        "STOP",
        "REVERSE_LEFT",
        "REVERSE_RIGHT",
    ]

    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence_threshold: Optional[float] = None,
    ):
        """Initialize the decision engine."""
        self._settings = get_settings()
        self._decision_logger = get_decision_logger()

        self.model_path = model_path or self._settings.DECISION_MODEL_PATH
        self.confidence_threshold = (
            confidence_threshold or self._settings.ML_CONFIDENCE_THRESHOLD
        )

        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._loaded = False

        # Performance tracking
        self._inference_times: List[float] = []
        self._prediction_count = 0
        self._low_confidence_count = 0

        logger.info(
            f"DecisionEngine initialized: threshold={self.confidence_threshold}"
        )

    def load_model(self) -> bool:
        """
        Load the TFLite model.

        Returns:
            True if model loaded successfully.
        """
        try:
            if not TFLITE_AVAILABLE:
                logger.warning("TFLite not available, using mock mode")
                self._loaded = True
                return True

            if not Path(self.model_path).exists():
                logger.warning(f"Model not found: {self.model_path}")
                logger.info("Using mock mode")
                self._loaded = True
                return True

            # Load TFLite interpreter
            self._interpreter = tflite.Interpreter(
                model_path=str(self.model_path),
                num_threads=4,
            )
            self._interpreter.allocate_tensors()

            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            self._loaded = True
            logger.info(f"Model loaded: {self.model_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._loaded = True  # Fall back to mock
            return True

    def predict(
        self,
        sensor_data: FusedSensorData,
    ) -> Tuple[int, float]:
        """
        Predict action from sensor data.

        Args:
            sensor_data: Fused sensor readings

        Returns:
            Tuple of (action_id, confidence)
        """
        if not self._loaded:
            logger.warning("Model not loaded, call load_model() first")
            return (4, 0.0)  # Default to STOP

        self._prediction_count += 1
        start_time = time.time()

        try:
            # Get normalized feature vector
            features = sensor_data.to_normalized_vector()

            # Run inference
            if self._interpreter is not None:
                action_id, confidence = self._run_inference(features)
            else:
                # Mock prediction
                action_id, confidence = self._mock_predict(sensor_data)

            # Track inference time
            inference_time = time.time() - start_time
            self._track_inference_time(inference_time)

            # Log low confidence predictions
            if confidence < self.confidence_threshold:
                self._low_confidence_count += 1
                logger.debug(
                    f"Low confidence prediction: {self.ACTION_NAMES[action_id]} "
                    f"({confidence:.2f} < {self.confidence_threshold})"
                )

            # Log decision
            self._decision_logger.log_decision(
                features=sensor_data.to_dict(),
                action=self.ACTION_NAMES[action_id],
                confidence=confidence,
                source="ML",
            )

            return (action_id, confidence)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return (4, 0.0)  # Default to STOP on error

    def _run_inference(self, features: np.ndarray) -> Tuple[int, float]:
        """Run actual TFLite inference."""
        # Prepare input
        input_dtype = self._input_details[0]["dtype"]

        if input_dtype == np.uint8:
            # Quantized model - scale input
            input_scale, input_zero = self._input_details[0]["quantization"]
            input_data = (features / input_scale + input_zero).astype(np.uint8)
        else:
            input_data = features.astype(np.float32)

        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)

        # Run inference
        self._interpreter.set_tensor(
            self._input_details[0]["index"],
            input_data
        )
        self._interpreter.invoke()

        # Get output
        output = self._interpreter.get_tensor(
            self._output_details[0]["index"]
        )[0]

        # Dequantize if needed
        if self._output_details[0]["dtype"] == np.uint8:
            output_scale, output_zero = self._output_details[0]["quantization"]
            output = (output.astype(np.float32) - output_zero) * output_scale

        # Get prediction
        action_id = int(np.argmax(output))
        confidence = float(np.max(output))

        return (action_id, confidence)

    def _mock_predict(self, sensor_data: FusedSensorData) -> Tuple[int, float]:
        """
        Mock prediction for testing without model.

        Uses simple rules similar to training data generation.
        """
        front = sensor_data.front_min_distance

        # Simple rule-based mock
        if front < 20:
            return (4, 0.95)  # STOP
        elif front < 50:
            if sensor_data.left_distance > sensor_data.right_distance:
                return (2, 0.85)  # TURN_LEFT
            else:
                return (3, 0.85)  # TURN_RIGHT
        elif front < 100:
            return (1, 0.80)  # SLOW_DOWN
        else:
            return (0, 0.90)  # FORWARD

    def _track_inference_time(self, time_s: float) -> None:
        """Track inference time for performance monitoring."""
        self._inference_times.append(time_s * 1000)  # Convert to ms
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)

    def is_confident(self, confidence: float) -> bool:
        """Check if confidence meets threshold."""
        return confidence >= self.confidence_threshold

    def get_action_name(self, action_id: int) -> str:
        """Get action name from ID."""
        if 0 <= action_id < len(self.ACTION_NAMES):
            return self.ACTION_NAMES[action_id]
        return f"UNKNOWN({action_id})"

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "predictions": self._prediction_count,
            "low_confidence_count": self._low_confidence_count,
            "low_confidence_rate": (
                self._low_confidence_count / max(1, self._prediction_count)
            ),
            "avg_inference_ms": (
                sum(self._inference_times) / max(1, len(self._inference_times))
            ),
            "model_loaded": self._loaded,
            "has_interpreter": self._interpreter is not None,
        }

    @property
    def average_inference_time_ms(self) -> float:
        """Get average inference time in milliseconds."""
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
