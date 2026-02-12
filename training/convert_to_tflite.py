"""
TensorFlow Lite Conversion Script.

Converts the trained Keras model to TFLite format with INT8 quantization
for efficient inference on Raspberry Pi 4.

Quantization Benefits:
- 4x smaller model size
- 2-3x faster inference
- Lower memory usage
- Minimal accuracy loss (typically < 1%)

Usage:
    python -m training.convert_to_tflite

    # Or with custom paths:
    python -m training.convert_to_tflite --input model.keras --output model.tflite
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Callable
import logging
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from config.settings import get_settings

logger = logging.getLogger(__name__)


class TFLiteConverter:
    """
    Convert Keras model to TFLite with quantization.

    Supports:
    - Dynamic range quantization (default)
    - Full integer quantization (INT8)
    - Float16 quantization
    """

    def __init__(self):
        """Initialize the converter."""
        self._settings = get_settings()

        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for TFLite conversion")

    def convert(
        self,
        model_path: Path,
        output_path: Path,
        quantization: str = "int8",
        representative_data: Optional[Callable] = None,
    ) -> Path:
        """
        Convert Keras model to TFLite.

        Args:
            model_path: Path to Keras model file
            output_path: Path to save TFLite model
            quantization: Quantization type ("none", "dynamic", "float16", "int8")
            representative_data: Generator function for INT8 calibration

        Returns:
            Path to saved TFLite model.
        """
        logger.info(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(str(model_path))

        logger.info(f"Converting with {quantization} quantization...")

        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Apply quantization
        if quantization == "none":
            pass  # No quantization

        elif quantization == "dynamic":
            # Dynamic range quantization (weights only)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        elif quantization == "float16":
            # Float16 quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        elif quantization == "int8":
            # Full integer quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

            if representative_data is not None:
                converter.representative_dataset = representative_data
            else:
                # Use default representative data
                converter.representative_dataset = self._get_representative_data()

        else:
            raise ValueError(f"Unknown quantization type: {quantization}")

        # Convert
        tflite_model = converter.convert()

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        # Log model sizes
        keras_size = model_path.stat().st_size / 1024
        tflite_size = output_path.stat().st_size / 1024
        reduction = (1 - tflite_size / keras_size) * 100

        logger.info(f"Conversion complete!")
        logger.info(f"Keras model size: {keras_size:.1f} KB")
        logger.info(f"TFLite model size: {tflite_size:.1f} KB")
        logger.info(f"Size reduction: {reduction:.1f}%")
        logger.info(f"Saved to: {output_path}")

        return output_path

    def _get_representative_data(self):
        """
        Generate representative data for INT8 calibration.

        Uses training data or generates synthetic data.
        """
        # Try to load training data
        data_path = self._settings.DATA_DIR / "synthetic" / "training_data.csv"

        if PANDAS_AVAILABLE and data_path.exists():
            logger.info(f"Loading calibration data from {data_path}")
            df = pd.read_csv(data_path)
            X = df.iloc[:, :-1].values.astype(np.float32)

            # Normalize
            X = self._normalize_features(X)

            # Use subset for calibration
            num_samples = min(1000, len(X))
            X_cal = X[:num_samples]

            def representative_dataset():
                for i in range(len(X_cal)):
                    yield [X_cal[i:i+1]]

            return representative_dataset

        else:
            logger.warning("Training data not found, using synthetic calibration data")
            return self._generate_synthetic_calibration_data()

    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features to 0-1 range."""
        max_distance = 400.0
        max_class = 10.0
        max_speed = 100.0
        max_heading = 90.0

        X_norm = X.copy()

        for i in [0, 1, 2, 3, 4, 5, 8]:
            X_norm[:, i] = np.clip(X_norm[:, i] / max_distance, 0, 1)
        X_norm[:, 7] = X_norm[:, 7] / max_class
        X_norm[:, 9] = X_norm[:, 9] / 2.0
        X_norm[:, 10] = X_norm[:, 10] / max_speed
        X_norm[:, 11] = (X_norm[:, 11] + max_heading) / (2 * max_heading)

        return X_norm

    def _generate_synthetic_calibration_data(self):
        """Generate synthetic calibration data."""
        num_samples = 500
        num_features = self._settings.FEATURE_VECTOR_SIZE

        def representative_dataset():
            for _ in range(num_samples):
                # Generate random normalized features
                sample = np.random.rand(1, num_features).astype(np.float32)
                yield [sample]

        return representative_dataset

    def verify_model(
        self,
        keras_model_path: Path,
        tflite_model_path: Path,
        num_samples: int = 100,
    ) -> dict:
        """
        Verify TFLite model accuracy matches Keras model.

        Args:
            keras_model_path: Path to original Keras model
            tflite_model_path: Path to converted TFLite model
            num_samples: Number of samples to test

        Returns:
            Dictionary with verification results.
        """
        logger.info("Verifying TFLite model accuracy...")

        # Load Keras model
        keras_model = tf.keras.models.load_model(str(keras_model_path))

        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Generate test samples
        test_data = np.random.rand(num_samples, self._settings.FEATURE_VECTOR_SIZE).astype(np.float32)

        # Get predictions from both models
        keras_preds = keras_model.predict(test_data, verbose=0)
        keras_classes = np.argmax(keras_preds, axis=1)

        tflite_classes = []
        for sample in test_data:
            # Handle quantized input
            if input_details[0]["dtype"] == np.uint8:
                input_scale, input_zero = input_details[0]["quantization"]
                sample_q = (sample / input_scale + input_zero).astype(np.uint8)
            else:
                sample_q = sample

            interpreter.set_tensor(input_details[0]["index"], [sample_q])
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]["index"])[0]

            # Handle quantized output
            if output_details[0]["dtype"] == np.uint8:
                output_scale, output_zero = output_details[0]["quantization"]
                output = (output.astype(np.float32) - output_zero) * output_scale

            tflite_classes.append(np.argmax(output))

        tflite_classes = np.array(tflite_classes)

        # Calculate agreement
        agreement = np.mean(keras_classes == tflite_classes)

        results = {
            "num_samples": num_samples,
            "agreement": agreement,
            "disagreements": np.sum(keras_classes != tflite_classes),
            "acceptable": agreement >= 0.99,  # 99% agreement threshold
        }

        logger.info(f"Model agreement: {agreement:.4f}")
        logger.info(f"Disagreements: {results['disagreements']}/{num_samples}")
        logger.info(f"Acceptable: {results['acceptable']}")

        return results

    def benchmark_inference(
        self,
        tflite_model_path: Path,
        num_iterations: int = 100,
    ) -> dict:
        """
        Benchmark TFLite model inference speed.

        Args:
            tflite_model_path: Path to TFLite model
            num_iterations: Number of inference iterations

        Returns:
            Dictionary with benchmark results.
        """
        import time

        logger.info(f"Benchmarking inference speed ({num_iterations} iterations)...")

        # Load model
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Prepare input
        input_shape = input_details[0]["shape"]
        input_dtype = input_details[0]["dtype"]

        if input_dtype == np.uint8:
            test_input = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
        else:
            test_input = np.random.rand(*input_shape).astype(np.float32)

        # Warm-up
        for _ in range(10):
            interpreter.set_tensor(input_details[0]["index"], test_input)
            interpreter.invoke()

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            interpreter.set_tensor(input_details[0]["index"], test_input)
            interpreter.invoke()
            times.append(time.perf_counter() - start)

        times_ms = np.array(times) * 1000

        results = {
            "iterations": num_iterations,
            "mean_ms": np.mean(times_ms),
            "std_ms": np.std(times_ms),
            "min_ms": np.min(times_ms),
            "max_ms": np.max(times_ms),
            "p95_ms": np.percentile(times_ms, 95),
            "meets_target": np.mean(times_ms) < 50,  # Target: < 50ms
        }

        logger.info(f"Mean inference time: {results['mean_ms']:.2f} ms")
        logger.info(f"Std deviation: {results['std_ms']:.2f} ms")
        logger.info(f"95th percentile: {results['p95_ms']:.2f} ms")
        logger.info(f"Meets target (<50ms): {results['meets_target']}")

        return results


def main():
    """Main conversion script."""
    parser = argparse.ArgumentParser(description="Convert model to TFLite")
    parser.add_argument("--input", type=str, default=None, help="Input Keras model path")
    parser.add_argument("--output", type=str, default=None, help="Output TFLite model path")
    parser.add_argument(
        "--quantization",
        type=str,
        default="int8",
        choices=["none", "dynamic", "float16", "int8"],
        help="Quantization type",
    )
    parser.add_argument("--verify", action="store_true", help="Verify model accuracy")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark inference speed")
    args = parser.parse_args()

    # Setup
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.logging_config import setup_logging
    setup_logging()

    settings = get_settings()

    # Paths
    input_path = Path(args.input) if args.input else settings.MODELS_DIR / "saved" / "decision_model.keras"
    output_path = Path(args.output) if args.output else settings.DECISION_MODEL_PATH

    print("=" * 60)
    print("TFLite Model Conversion")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Quantization: {args.quantization}")
    print()

    # Convert
    converter = TFLiteConverter()
    converter.convert(
        model_path=input_path,
        output_path=output_path,
        quantization=args.quantization,
    )

    # Verify
    if args.verify:
        print()
        print("=" * 60)
        print("Model Verification")
        print("=" * 60)
        results = converter.verify_model(input_path, output_path)
        print(f"Agreement: {results['agreement']:.4f}")

    # Benchmark
    if args.benchmark:
        print()
        print("=" * 60)
        print("Inference Benchmark")
        print("=" * 60)
        results = converter.benchmark_inference(output_path)
        print(f"Mean inference time: {results['mean_ms']:.2f} ms")

    print()
    print("Conversion complete!")


if __name__ == "__main__":
    main()
