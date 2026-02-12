"""
Synthetic Training Data Generator.

Generates training data for the decision model using expert rules.
The rules encode safe driving behavior that the ML model will learn.

Expert Rules (encoded in this module):
- CRITICAL ZONE: front < 20cm → STOP
- DANGER ZONE: front 20-50cm → Turn or STOP based on clearance
- CAUTION ZONE: front 50-100cm → SLOW_DOWN
- CLEAR ZONE: front > 100cm → FORWARD (with heading correction)
- SIDE PROXIMITY: React to close side obstacles
- CAMERA OVERRIDE: Person detected → STOP

Output Format:
    CSV with 13 columns (12 features + 1 action label)

Usage:
    python -m training.generate_synthetic_data

    # Or import and use programmatically:
    from training.generate_synthetic_data import SyntheticDataGenerator
    generator = SyntheticDataGenerator()
    df = generator.generate(num_samples=50000)
"""

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from config.settings import get_settings

logger = logging.getLogger(__name__)


# Action labels
class Actions:
    """Action IDs matching the model output."""
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


@dataclass
class SensorSample:
    """Container for a single sensor reading sample."""
    front_distance: float
    front_wp_distance: float
    right_distance: float
    left_distance: float
    rear_distance: float
    front_min_distance: float
    camera_object_detected: int
    camera_object_class: int
    camera_object_distance_estimate: float
    camera_object_position: int
    current_speed: int
    heading_error: float

    def to_list(self) -> List[float]:
        """Convert to list of feature values."""
        return [
            self.front_distance,
            self.front_wp_distance,
            self.right_distance,
            self.left_distance,
            self.rear_distance,
            self.front_min_distance,
            float(self.camera_object_detected),
            float(self.camera_object_class),
            self.camera_object_distance_estimate,
            float(self.camera_object_position),
            float(self.current_speed),
            self.heading_error,
        ]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
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


class ExpertRules:
    """
    Expert rules for determining the correct action.

    These rules encode safe driving behavior that the ML model
    will learn to approximate.
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

    @classmethod
    def determine_action(cls, sample: SensorSample) -> int:
        """
        Determine the correct action based on expert rules.

        Args:
            sample: SensorSample with current readings

        Returns:
            Action ID (0-6)
        """
        front = sample.front_min_distance
        left = sample.left_distance
        right = sample.right_distance
        rear = sample.rear_distance

        camera_detected = sample.camera_object_detected
        camera_class = sample.camera_object_class
        camera_position = sample.camera_object_position
        camera_distance = sample.camera_object_distance_estimate

        heading_error = sample.heading_error

        # ================================================================
        # RULE 1: Person detected - ALWAYS STOP (safety first)
        # ================================================================
        if camera_detected == 1 and camera_class == 1:  # 1 = person
            return Actions.STOP

        # ================================================================
        # RULE 2: CRITICAL ZONE (front < 20cm) - STOP
        # ================================================================
        if front < cls.CRITICAL_DISTANCE:
            return Actions.STOP

        # ================================================================
        # RULE 3: DANGER ZONE (front 20-50cm)
        # ================================================================
        if front < cls.DANGER_DISTANCE:
            # Check if we can turn to avoid
            if left > cls.SIDE_SAFE and left > right:
                return Actions.TURN_LEFT
            elif right > cls.SIDE_SAFE and right >= left:
                return Actions.TURN_RIGHT
            else:
                # Both sides blocked, try reverse
                if rear > cls.CRITICAL_DISTANCE:
                    if right > left:
                        return Actions.REVERSE_RIGHT
                    else:
                        return Actions.REVERSE_LEFT
                else:
                    return Actions.STOP

        # ================================================================
        # RULE 4: CAUTION ZONE (front 50-100cm) - SLOW_DOWN
        # ================================================================
        if front < cls.CAUTION_DISTANCE:
            # Check camera detection position
            if camera_detected:
                if camera_position == 0:  # left
                    return Actions.TURN_RIGHT
                elif camera_position == 2:  # right
                    return Actions.TURN_LEFT
                else:  # center
                    return Actions.SLOW_DOWN
            return Actions.SLOW_DOWN

        # ================================================================
        # RULE 5: CLEAR ZONE (front > 100cm)
        # ================================================================
        # Check for side proximity
        if left < cls.SIDE_CRITICAL:
            return Actions.TURN_RIGHT
        if right < cls.SIDE_CRITICAL:
            return Actions.TURN_LEFT

        # Check for heading error correction
        if abs(heading_error) > cls.HEADING_CORRECTION_THRESHOLD:
            if heading_error > 0:
                return Actions.TURN_RIGHT  # Correct to the right
            else:
                return Actions.TURN_LEFT  # Correct to the left

        # Camera object in path but far
        if camera_detected and camera_position == 1:  # center
            if camera_distance < cls.CAUTION_DISTANCE:
                return Actions.SLOW_DOWN

        # All clear - go forward
        return Actions.FORWARD


class SyntheticDataGenerator:
    """
    Generates synthetic training data using expert rules.

    Creates varied sensor scenarios and labels them with the
    correct action according to expert rules.

    Args:
        num_samples: Number of samples to generate
        noise_std: Standard deviation of sensor noise (cm)
        false_reading_prob: Probability of sensor error
    """

    def __init__(
        self,
        noise_std: Optional[float] = None,
        false_reading_prob: Optional[float] = None,
    ):
        """Initialize the generator."""
        settings = get_settings()

        self.noise_std = noise_std or settings.SENSOR_NOISE_STD_CM
        self.false_reading_prob = false_reading_prob or settings.FALSE_READING_PROBABILITY

        # Feature names
        self.feature_names = [
            "front_distance",
            "front_wp_distance",
            "right_distance",
            "left_distance",
            "rear_distance",
            "front_min_distance",
            "camera_object_detected",
            "camera_object_class",
            "camera_object_distance_estimate",
            "camera_object_position",
            "current_speed",
            "heading_error",
        ]

        logger.info(
            f"SyntheticDataGenerator initialized: noise_std={self.noise_std}, "
            f"false_reading_prob={self.false_reading_prob}"
        )

    def generate(
        self,
        num_samples: int = 50000,
        seed: Optional[int] = None,
    ) -> List[Tuple[List[float], int]]:
        """
        Generate synthetic training samples.

        Args:
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility

        Returns:
            List of (features, label) tuples
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        samples = []
        action_counts = {i: 0 for i in range(7)}

        logger.info(f"Generating {num_samples} synthetic samples...")
        start_time = time.time()

        for i in range(num_samples):
            sample, action = self._generate_sample()
            samples.append((sample.to_list(), action))
            action_counts[action] += 1

            if (i + 1) % 10000 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples...")

        elapsed = time.time() - start_time
        logger.info(f"Generated {num_samples} samples in {elapsed:.1f}s")

        # Log action distribution
        logger.info("Action distribution:")
        for action_id, count in action_counts.items():
            pct = count / num_samples * 100
            logger.info(f"  {Actions.NAMES[action_id]}: {count} ({pct:.1f}%)")

        return samples

    def _generate_sample(self) -> Tuple[SensorSample, int]:
        """Generate a single sample with its label."""

        # Generate scenario type (weighted to ensure variety)
        scenario = random.choices(
            ["critical", "danger", "caution", "clear", "side", "camera"],
            weights=[0.1, 0.2, 0.2, 0.3, 0.1, 0.1],
            k=1
        )[0]

        if scenario == "critical":
            sample = self._generate_critical_scenario()
        elif scenario == "danger":
            sample = self._generate_danger_scenario()
        elif scenario == "caution":
            sample = self._generate_caution_scenario()
        elif scenario == "clear":
            sample = self._generate_clear_scenario()
        elif scenario == "side":
            sample = self._generate_side_scenario()
        else:
            sample = self._generate_camera_scenario()

        # Add noise
        sample = self._add_noise(sample)

        # Determine correct action
        action = ExpertRules.determine_action(sample)

        return sample, action

    def _generate_critical_scenario(self) -> SensorSample:
        """Generate a critical zone scenario (front < 20cm)."""
        front = random.uniform(2, 20)
        front_wp = front + random.uniform(-5, 5)
        front_wp = max(2, front_wp)

        return SensorSample(
            front_distance=front,
            front_wp_distance=front_wp,
            right_distance=random.uniform(20, 200),
            left_distance=random.uniform(20, 200),
            rear_distance=random.uniform(50, 300),
            front_min_distance=min(front, front_wp),
            camera_object_detected=random.choice([0, 0, 1]),
            camera_object_class=random.choice([0, 0, 1, 2, 3, 4]),
            camera_object_distance_estimate=random.uniform(10, 100),
            camera_object_position=random.choice([0, 1, 2]),
            current_speed=random.randint(0, 70),
            heading_error=random.uniform(-30, 30),
        )

    def _generate_danger_scenario(self) -> SensorSample:
        """Generate a danger zone scenario (front 20-50cm)."""
        front = random.uniform(20, 50)
        front_wp = front + random.uniform(-10, 10)
        front_wp = max(10, front_wp)

        # Vary side clearance for different outcomes
        if random.random() < 0.3:
            # Both sides blocked
            right = random.uniform(20, 60)
            left = random.uniform(20, 60)
        elif random.random() < 0.5:
            # Left clear
            right = random.uniform(20, 50)
            left = random.uniform(60, 200)
        else:
            # Right clear
            right = random.uniform(60, 200)
            left = random.uniform(20, 50)

        return SensorSample(
            front_distance=front,
            front_wp_distance=front_wp,
            right_distance=right,
            left_distance=left,
            rear_distance=random.uniform(50, 300),
            front_min_distance=min(front, front_wp),
            camera_object_detected=random.choice([0, 0, 1]),
            camera_object_class=random.choice([0, 0, 2, 3, 4]),
            camera_object_distance_estimate=random.uniform(30, 150),
            camera_object_position=random.choice([0, 1, 2]),
            current_speed=random.randint(20, 60),
            heading_error=random.uniform(-20, 20),
        )

    def _generate_caution_scenario(self) -> SensorSample:
        """Generate a caution zone scenario (front 50-100cm)."""
        front = random.uniform(50, 100)
        front_wp = front + random.uniform(-15, 15)
        front_wp = max(30, front_wp)

        return SensorSample(
            front_distance=front,
            front_wp_distance=front_wp,
            right_distance=random.uniform(40, 250),
            left_distance=random.uniform(40, 250),
            rear_distance=random.uniform(100, 400),
            front_min_distance=min(front, front_wp),
            camera_object_detected=random.choice([0, 0, 0, 1]),
            camera_object_class=random.choice([0, 2, 3, 4]),
            camera_object_distance_estimate=random.uniform(50, 200),
            camera_object_position=random.choice([0, 1, 2]),
            current_speed=random.randint(30, 70),
            heading_error=random.uniform(-25, 25),
        )

    def _generate_clear_scenario(self) -> SensorSample:
        """Generate a clear zone scenario (front > 100cm)."""
        front = random.uniform(100, 400)
        front_wp = front + random.uniform(-20, 20)
        front_wp = max(80, front_wp)

        # Sometimes add heading error for correction
        if random.random() < 0.3:
            heading_error = random.uniform(-45, 45)
        else:
            heading_error = random.uniform(-10, 10)

        return SensorSample(
            front_distance=front,
            front_wp_distance=front_wp,
            right_distance=random.uniform(50, 300),
            left_distance=random.uniform(50, 300),
            rear_distance=random.uniform(100, 400),
            front_min_distance=min(front, front_wp),
            camera_object_detected=random.choice([0, 0, 0, 0, 1]),
            camera_object_class=random.choice([0, 0, 2, 3, 4]),
            camera_object_distance_estimate=random.uniform(100, 400),
            camera_object_position=random.choice([0, 1, 2]),
            current_speed=random.randint(40, 85),
            heading_error=heading_error,
        )

    def _generate_side_scenario(self) -> SensorSample:
        """Generate a side proximity scenario."""
        front = random.uniform(100, 300)
        front_wp = front + random.uniform(-10, 10)

        # One side very close
        if random.random() < 0.5:
            right = random.uniform(10, 30)
            left = random.uniform(60, 200)
        else:
            right = random.uniform(60, 200)
            left = random.uniform(10, 30)

        return SensorSample(
            front_distance=front,
            front_wp_distance=front_wp,
            right_distance=right,
            left_distance=left,
            rear_distance=random.uniform(100, 300),
            front_min_distance=min(front, front_wp),
            camera_object_detected=0,
            camera_object_class=0,
            camera_object_distance_estimate=0,
            camera_object_position=1,
            current_speed=random.randint(30, 60),
            heading_error=random.uniform(-15, 15),
        )

    def _generate_camera_scenario(self) -> SensorSample:
        """Generate a camera detection scenario."""
        front = random.uniform(50, 200)
        front_wp = front + random.uniform(-10, 10)

        # Person detected - will trigger STOP
        camera_class = random.choices([1, 2, 3, 4], weights=[0.5, 0.2, 0.2, 0.1], k=1)[0]
        camera_position = random.choice([0, 1, 2])

        return SensorSample(
            front_distance=front,
            front_wp_distance=front_wp,
            right_distance=random.uniform(60, 250),
            left_distance=random.uniform(60, 250),
            rear_distance=random.uniform(100, 400),
            front_min_distance=min(front, front_wp),
            camera_object_detected=1,
            camera_object_class=camera_class,
            camera_object_distance_estimate=random.uniform(30, 300),
            camera_object_position=camera_position,
            current_speed=random.randint(30, 70),
            heading_error=random.uniform(-20, 20),
        )

    def _add_noise(self, sample: SensorSample) -> SensorSample:
        """Add realistic sensor noise to sample."""
        # Gaussian noise on distances
        sample.front_distance = max(0, sample.front_distance + np.random.normal(0, self.noise_std))
        sample.front_wp_distance = max(0, sample.front_wp_distance + np.random.normal(0, self.noise_std))
        sample.right_distance = max(0, sample.right_distance + np.random.normal(0, self.noise_std))
        sample.left_distance = max(0, sample.left_distance + np.random.normal(0, self.noise_std))
        sample.rear_distance = max(0, sample.rear_distance + np.random.normal(0, self.noise_std))

        # Recalculate front_min
        sample.front_min_distance = min(sample.front_distance, sample.front_wp_distance)

        # Occasional sensor failure (reading 0)
        if random.random() < self.false_reading_prob:
            sensor = random.choice(["front", "front_wp", "right", "left"])
            if sensor == "front":
                sample.front_distance = 0
            elif sensor == "front_wp":
                sample.front_wp_distance = 0
            elif sensor == "right":
                sample.right_distance = 0
            else:
                sample.left_distance = 0
            sample.front_min_distance = min(sample.front_distance, sample.front_wp_distance)

        return sample

    def to_dataframe(self, samples: List[Tuple[List[float], int]]):
        """
        Convert samples to pandas DataFrame.

        Args:
            samples: List of (features, label) tuples

        Returns:
            pandas DataFrame with features and action column
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for DataFrame export")

        features = [s[0] for s in samples]
        labels = [s[1] for s in samples]

        df = pd.DataFrame(features, columns=self.feature_names)
        df["action"] = labels

        return df

    def save_to_csv(
        self,
        samples: List[Tuple[List[float], int]],
        output_path: Path,
    ) -> None:
        """
        Save samples to CSV file.

        Args:
            samples: List of (features, label) tuples
            output_path: Path to save CSV
        """
        df = self.to_dataframe(samples)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(samples)} samples to {output_path}")


def main():
    """Generate synthetic training data and save to CSV."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config.logging_config import setup_logging
    setup_logging()

    settings = get_settings()

    # Create generator
    generator = SyntheticDataGenerator()

    # Generate samples
    num_samples = settings.SYNTHETIC_SAMPLES
    samples = generator.generate(num_samples=num_samples, seed=42)

    # Save to CSV
    output_dir = settings.DATA_DIR / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "training_data.csv"
    generator.save_to_csv(samples, output_path)

    print(f"\nSynthetic data generation complete!")
    print(f"Output: {output_path}")
    print(f"Samples: {len(samples)}")


if __name__ == "__main__":
    main()
