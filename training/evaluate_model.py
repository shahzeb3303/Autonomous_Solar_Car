#!/usr/bin/env python3
"""
Model Evaluation Script.

Evaluates the trained decision model on test data:
- Overall accuracy
- Per-class precision, recall, F1
- Confusion matrix
- Action distribution analysis

Usage:
    python -m training.evaluate_model
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
import logging
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from config.settings import get_settings

logger = logging.getLogger(__name__)


ACTION_NAMES = [
    "FORWARD",
    "SLOW_DOWN",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STOP",
    "REVERSE_LEFT",
    "REVERSE_RIGHT",
]


def load_test_data(data_path: Path):
    """Load and prepare test data."""
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas required for evaluation")

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df["action"].values.astype(np.int32)

    # Normalize features
    max_distance = 400.0
    X_norm = X.copy()
    for i in [0, 1, 2, 3, 4, 5, 8]:
        X_norm[:, i] = np.clip(X_norm[:, i] / max_distance, 0, 1)
    X_norm[:, 7] = X_norm[:, 7] / 10.0
    X_norm[:, 9] = X_norm[:, 9] / 2.0
    X_norm[:, 10] = X_norm[:, 10] / 100.0
    X_norm[:, 11] = (X_norm[:, 11] + 90.0) / 180.0

    return X_norm, y


def evaluate_keras_model(model_path: Path, X: np.ndarray, y: np.ndarray):
    """Evaluate Keras model."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required for Keras evaluation")

    print(f"\nLoading model from {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    # Get predictions
    print("Running predictions...")
    y_pred_probs = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    return y_pred, y_pred_probs


def evaluate_tflite_model(model_path: Path, X: np.ndarray, y: np.ndarray):
    """Evaluate TFLite model."""
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow as tf
        tflite = tf.lite

    print(f"\nLoading TFLite model from {model_path}")
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get predictions
    print("Running predictions...")
    y_pred = []

    for sample in X:
        input_data = np.expand_dims(sample, axis=0)

        if input_details[0]["dtype"] == np.uint8:
            input_scale, input_zero = input_details[0]["quantization"]
            input_data = (input_data / input_scale + input_zero).astype(np.uint8)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]

        if output_details[0]["dtype"] == np.uint8:
            output_scale, output_zero = output_details[0]["quantization"]
            output = (output.astype(np.float32) - output_zero) * output_scale

        y_pred.append(np.argmax(output))

    return np.array(y_pred), None


def print_results(y_true: np.ndarray, y_pred: np.ndarray):
    """Print evaluation results."""
    if not SKLEARN_AVAILABLE:
        # Simple accuracy calculation
        accuracy = np.mean(y_true == y_pred)
        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        return

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Target check
    target_met = accuracy >= 0.98
    print(f"Target (98%) Met: {'YES [PASS]' if target_met else 'NO [MISS]'}")

    # Classification report
    print("\n" + "-" * 60)
    print("Per-Class Metrics:")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=ACTION_NAMES))

    # Confusion matrix
    print("\n" + "-" * 60)
    print("Confusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_true, y_pred)

    # Header
    header = "         " + " ".join(f"{n[:7]:>7}" for n in ACTION_NAMES)
    print(header)
    print("-" * len(header))

    for i, row in enumerate(cm):
        row_str = f"{ACTION_NAMES[i][:8]:>8} " + " ".join(f"{v:7d}" for v in row)
        print(row_str)

    # Action distribution
    print("\n" + "-" * 60)
    print("Action Distribution:")
    print("-" * 60)

    true_counts = np.bincount(y_true, minlength=7)
    pred_counts = np.bincount(y_pred, minlength=7)

    print(f"{'Action':<12} {'True':>8} {'Predicted':>10}")
    print("-" * 32)
    for i, name in enumerate(ACTION_NAMES):
        print(f"{name:<12} {true_counts[i]:>8} {pred_counts[i]:>10}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate decision model")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path (.keras or .tflite)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Test data path"
    )
    args = parser.parse_args()

    settings = get_settings()

    # Determine model path and type
    if args.model:
        model_path = Path(args.model)
    else:
        # Try Keras model first, then TFLite
        keras_path = settings.MODELS_DIR / "saved" / "decision_model.keras"
        tflite_path = settings.DECISION_MODEL_PATH

        if keras_path.exists():
            model_path = keras_path
        elif tflite_path.exists():
            model_path = tflite_path
        else:
            print("ERROR: No model found. Train a model first.")
            sys.exit(1)

    # Determine data path
    data_path = Path(args.data) if args.data else (
        settings.DATA_DIR / "synthetic" / "training_data.csv"
    )

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    print("=" * 60)
    print("Decision Model Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")

    # Load data
    X, y = load_test_data(data_path)
    print(f"Samples: {len(X)}")

    # Evaluate
    if model_path.suffix == ".tflite":
        y_pred, _ = evaluate_tflite_model(model_path, X, y)
    else:
        y_pred, _ = evaluate_keras_model(model_path, X, y)

    # Print results
    print_results(y, y_pred)


if __name__ == "__main__":
    main()
