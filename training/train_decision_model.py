"""
Decision Model Training Script.

Trains a neural network to predict navigation actions from sensor data.

Model Architecture:
    Input: 12 features (normalized 0-1)
    Dense(64, relu) -> BatchNorm -> Dropout(0.2)
    Dense(32, relu) -> BatchNorm -> Dropout(0.2)
    Dense(16, relu)
    Dense(7, softmax)
    Output: probability distribution over 7 actions

Training:
    - Loss: sparse_categorical_crossentropy
    - Optimizer: Adam with ReduceLROnPlateau
    - Early stopping: patience=10
    - Target: ≥98% validation accuracy

Usage:
    python -m training.train_decision_model

    # Or with custom parameters:
    python -m training.train_decision_model --epochs 50 --batch-size 64
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
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


class DecisionModelBuilder:
    """
    Builds and trains the decision neural network.

    The model is a small, efficient architecture optimized for
    Raspberry Pi 4 inference.
    """

    def __init__(self):
        """Initialize the model builder."""
        self._settings = get_settings()

        self.input_size = self._settings.FEATURE_VECTOR_SIZE
        self.num_actions = self._settings.NUM_ACTIONS
        self.hidden_sizes = self._settings.HIDDEN_LAYER_SIZES
        self.dropout_rate = self._settings.DROPOUT_RATE
        self.learning_rate = self._settings.LEARNING_RATE

        self.model = None
        self.history = None

        logger.info(
            f"DecisionModelBuilder initialized: "
            f"input={self.input_size}, actions={self.num_actions}"
        )

    def build_model(self) -> keras.Model:
        """
        Build the neural network architecture.

        Returns:
            Compiled Keras model.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for model training")

        # Input layer
        inputs = keras.Input(shape=(self.input_size,), name="sensor_input")

        # Hidden layers
        x = inputs

        for i, hidden_size in enumerate(self.hidden_sizes):
            x = layers.Dense(
                hidden_size,
                activation="relu",
                name=f"dense_{i+1}"
            )(x)
            x = layers.BatchNormalization(name=f"bn_{i+1}")(x)

            # Dropout only on first two layers
            if i < 2:
                x = layers.Dropout(self.dropout_rate, name=f"dropout_{i+1}")(x)

        # Output layer
        outputs = layers.Dense(
            self.num_actions,
            activation="softmax",
            name="action_output"
        )(x)

        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name="decision_model")

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model

        # Print model summary
        model.summary(print_fn=logger.info)

        return model

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        model_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            x_train: Training features
            y_train: Training labels
            x_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Training batch size
            early_stopping_patience: Early stopping patience
            model_dir: Directory to save model checkpoints

        Returns:
            Training results dictionary.
        """
        if self.model is None:
            self.build_model()

        epochs = epochs or self._settings.EPOCHS
        batch_size = batch_size or self._settings.BATCH_SIZE
        patience = early_stopping_patience or self._settings.EARLY_STOPPING_PATIENCE
        model_dir = model_dir or self._settings.MODELS_DIR / "saved"

        # Ensure model directory exists
        model_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks
        callback_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),

            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),

            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=str(model_dir / "checkpoint_best.keras"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),

            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=str(self._settings.LOGS_DIR / "training"),
                histogram_freq=1,
            ),
        ]

        logger.info(
            f"Starting training: epochs={epochs}, batch_size={batch_size}, "
            f"train_samples={len(x_train)}, val_samples={len(x_val)}"
        )

        start_time = time.time()

        # Train model
        self.history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1,
        )

        training_time = time.time() - start_time

        # Get best metrics
        best_epoch = np.argmax(self.history.history["val_accuracy"]) + 1
        best_val_acc = max(self.history.history["val_accuracy"])
        final_train_acc = self.history.history["accuracy"][-1]

        results = {
            "training_time_seconds": training_time,
            "epochs_run": len(self.history.history["accuracy"]),
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
            "final_train_accuracy": final_train_acc,
            "target_met": best_val_acc >= 0.98,
        }

        logger.info(f"Training complete in {training_time:.1f}s")
        logger.info(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
        logger.info(f"Target (98%) met: {results['target_met']}")

        return results

    def save_model(self, path: Optional[Path] = None) -> Path:
        """
        Save the trained model.

        Args:
            path: Save path. Defaults to models/saved/decision_model.keras

        Returns:
            Path where model was saved.
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")

        path = path or self._settings.MODELS_DIR / "saved" / "decision_model.keras"
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(path))
        logger.info(f"Model saved to {path}")

        return path

    def load_model(self, path: Optional[Path] = None) -> keras.Model:
        """
        Load a trained model.

        Args:
            path: Model path. Defaults to models/saved/decision_model.keras

        Returns:
            Loaded Keras model.
        """
        path = path or self._settings.MODELS_DIR / "saved" / "decision_model.keras"
        self.model = keras.models.load_model(str(path))
        logger.info(f"Model loaded from {path}")
        return self.model


class DataLoader:
    """Load and preprocess training data."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize data loader."""
        self._settings = get_settings()
        self.data_path = data_path or self._settings.DATA_DIR / "synthetic" / "training_data.csv"

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from CSV.

        Returns:
            Tuple of (features, labels) as numpy arrays.
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for data loading")

        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)

        # Separate features and labels
        feature_cols = df.columns[:-1]  # All except last column (action)
        X = df[feature_cols].values.astype(np.float32)
        y = df["action"].values.astype(np.int32)

        logger.info(f"Loaded {len(X)} samples, {X.shape[1]} features")

        return X, y

    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize features to 0-1 range.

        Uses predefined normalization based on expected value ranges.
        """
        # Normalization factors for each feature
        # [front, front_wp, right, left, rear, front_min,
        #  cam_det, cam_class, cam_dist, cam_pos, speed, heading]
        max_distance = 400.0
        max_class = 10.0
        max_speed = 100.0
        max_heading = 90.0

        X_norm = X.copy()

        # Distance features (indices 0-5, 8)
        for i in [0, 1, 2, 3, 4, 5, 8]:
            X_norm[:, i] = np.clip(X_norm[:, i] / max_distance, 0, 1)

        # Boolean features (index 6)
        X_norm[:, 6] = X_norm[:, 6]  # Already 0 or 1

        # Class (index 7)
        X_norm[:, 7] = X_norm[:, 7] / max_class

        # Position (index 9)
        X_norm[:, 9] = X_norm[:, 9] / 2.0  # 0, 0.5, or 1

        # Speed (index 10)
        X_norm[:, 10] = X_norm[:, 10] / max_speed

        # Heading (index 11) - convert from -90 to 90 to 0 to 1
        X_norm[:, 11] = (X_norm[:, 11] + max_heading) / (2 * max_heading)

        return X_norm

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/val/test sets.

        Args:
            X: Features
            y: Labels
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            seed: Random seed

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        np.random.seed(seed)

        n_samples = len(X)
        indices = np.random.permutation(n_samples)

        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        logger.info(
            f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )

        return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train decision model")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--data-path", type=str, default=None, help="Training data path")
    args = parser.parse_args()

    # Setup
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.logging_config import setup_logging
    setup_logging()

    settings = get_settings()

    print("=" * 60)
    print("Decision Model Training")
    print("=" * 60)

    # Load data
    data_path = Path(args.data_path) if args.data_path else None
    loader = DataLoader(data_path)

    X, y = loader.load_data()
    X_norm = loader.normalize_features(X)

    X_train, y_train, X_val, y_val, X_test, y_test = loader.split_data(X_norm, y)

    # Build and train model
    builder = DecisionModelBuilder()
    model = builder.build_model()

    results = builder.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Save model
    model_path = builder.save_model()

    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Training time: {results['training_time_seconds']:.1f}s")
    print(f"Epochs run: {results['epochs_run']}")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Target (98%) met: {'YES' if test_acc >= 0.98 else 'NO'}")


if __name__ == "__main__":
    main()
