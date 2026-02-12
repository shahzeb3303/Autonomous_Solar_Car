"""
Central Settings Configuration for Autonomous Solar Vehicle.

This module contains ALL configurable parameters for the vehicle system.
No magic numbers should exist elsewhere in the codebase - import from here.

Settings are organized by subsystem:
- Sensor configuration (thresholds, timing)
- Motor control parameters
- ML model settings
- Safety thresholds
- Logging and paths

Usage:
    from config import get_settings
    settings = get_settings()
    threshold = settings.CRITICAL_DISTANCE_CM
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """
    Immutable settings configuration for the autonomous vehicle.

    All measurements are in SI units unless otherwise noted:
    - Distances: centimeters (cm)
    - Time: seconds (s) or milliseconds (ms) as noted
    - Speed: percentage of max (0-100)
    - Angles: degrees
    """

    # =========================================================================
    # PATH CONFIGURATION
    # =========================================================================

    # Project root directory (computed relative to this file)
    PROJECT_ROOT: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.absolute()
    )

    @property
    def DATA_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data"

    @property
    def MODELS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "models"

    @property
    def LOGS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "logs"

    @property
    def DECISION_MODEL_PATH(self) -> Path:
        return self.MODELS_DIR / "tflite" / "decision_model.tflite"

    @property
    def OBJECT_DETECTION_MODEL_PATH(self) -> Path:
        return self.MODELS_DIR / "object_detection" / "mobilenet_ssd_v2.tflite"

    @property
    def OBJECT_DETECTION_LABELS_PATH(self) -> Path:
        return self.MODELS_DIR / "object_detection" / "coco_labels.txt"

    # =========================================================================
    # ULTRASONIC SENSOR CONFIGURATION
    # =========================================================================

    # Distance thresholds (in centimeters)
    CRITICAL_DISTANCE_CM: float = 20.0    # STOP zone - immediate halt
    DANGER_DISTANCE_CM: float = 50.0      # Turn/avoid zone
    CAUTION_DISTANCE_CM: float = 100.0    # Slow down zone
    CLEAR_DISTANCE_CM: float = 150.0      # All clear, full speed

    # Side clearance thresholds
    SIDE_CRITICAL_CM: float = 30.0        # Too close to side obstacle
    SIDE_SAFE_CM: float = 60.0            # Safe side clearance for turns

    # Sensor timing (in seconds)
    ULTRASONIC_TIMEOUT_S: float = 0.03    # 30ms timeout (~5m max range)
    ULTRASONIC_TRIGGER_PULSE_S: float = 0.00001  # 10μs trigger pulse
    ULTRASONIC_SETTLE_TIME_S: float = 0.002  # 2ms between measurements

    # Speed of sound constant for distance calculation
    # distance_cm = pulse_duration_s * SOUND_SPEED_CM_PER_S / 2
    SOUND_SPEED_CM_PER_S: float = 34300.0  # Speed of sound at 20°C

    # Sensor valid range
    ULTRASONIC_MIN_RANGE_CM: float = 2.0   # Minimum reliable reading
    ULTRASONIC_MAX_RANGE_CM: float = 400.0 # Maximum reliable reading

    # Measurement filtering
    ULTRASONIC_SAMPLE_COUNT: int = 3       # Readings to average per measurement
    ULTRASONIC_OUTLIER_THRESHOLD: float = 50.0  # Max deviation from median (cm)

    # =========================================================================
    # CAMERA CONFIGURATION
    # =========================================================================

    # Front camera (CSI)
    CAMERA_FRONT_WIDTH: int = 640
    CAMERA_FRONT_HEIGHT: int = 480
    CAMERA_FRONT_FPS: int = 15

    # Rear camera (USB)
    CAMERA_REAR_WIDTH: int = 640
    CAMERA_REAR_HEIGHT: int = 480
    CAMERA_REAR_FPS: int = 10

    # Object detection model input
    DETECTION_INPUT_WIDTH: int = 300
    DETECTION_INPUT_HEIGHT: int = 300

    # Detection confidence threshold
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.5
    PERSON_CONFIDENCE_THRESHOLD: float = 0.4  # Lower threshold for safety

    # Camera processing
    CAMERA_SKIP_FRAMES: int = 3  # Process every Nth frame

    # =========================================================================
    # OBJECT DETECTION CLASSES
    # =========================================================================

    # COCO class IDs we care about for obstacle avoidance
    # Full COCO has 90 classes, we focus on safety-relevant ones
    TARGET_DETECTION_CLASSES: Dict[int, str] = field(default_factory=lambda: {
        0: "person",      # CRITICAL - always stop
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        15: "bird",
        16: "cat",
        17: "dog",
        56: "chair",
        57: "couch",
        58: "potted_plant",
        60: "dining_table",
    })

    # Classes that require immediate stop (safety-critical)
    STOP_CLASSES: Tuple[int, ...] = (0,)  # Person only

    # Classes that require caution
    CAUTION_CLASSES: Tuple[int, ...] = (1, 2, 3, 5, 7, 16, 17)

    # =========================================================================
    # DISTANCE ESTIMATION FROM BOUNDING BOX
    # =========================================================================

    # Approximate object heights (cm) for distance estimation
    REFERENCE_HEIGHTS_CM: Dict[str, float] = field(default_factory=lambda: {
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
    })

    # Camera focal length (approximate, needs calibration)
    CAMERA_FOCAL_LENGTH_PX: float = 500.0

    # =========================================================================
    # MOTOR CONTROL CONFIGURATION
    # =========================================================================

    # PWM settings
    PWM_FREQUENCY_HZ: int = 1000
    PWM_MAX_DUTY_CYCLE: int = 100
    PWM_MIN_DUTY_CYCLE: int = 0

    # Speed settings (as percentage of max PWM)
    SPEED_STOP: int = 0
    SPEED_SLOW: int = 30
    SPEED_NORMAL: int = 50
    SPEED_FAST: int = 70
    SPEED_MAX: int = 85  # Never 100% to preserve motor life

    # Speed reduction factor when slowing down
    SLOW_DOWN_FACTOR: float = 0.5  # Reduce speed by 50%

    # Turning configuration
    TURN_SPEED_INNER: int = 20     # Inner wheel speed during turn
    TURN_SPEED_OUTER: int = 50     # Outer wheel speed during turn
    TURN_DURATION_S: float = 0.5   # Default turn duration

    # Motor ramp-up/down for smooth acceleration
    ACCELERATION_STEP: int = 5     # PWM increment per step
    ACCELERATION_DELAY_S: float = 0.05  # Delay between steps

    # =========================================================================
    # ML DECISION MODEL CONFIGURATION
    # =========================================================================

    # Feature vector size (must match training data)
    FEATURE_VECTOR_SIZE: int = 12

    # Feature names (for logging and debugging)
    FEATURE_NAMES: Tuple[str, ...] = (
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
    )

    # Action space
    ACTION_NAMES: Tuple[str, ...] = (
        "FORWARD",
        "SLOW_DOWN",
        "TURN_LEFT",
        "TURN_RIGHT",
        "STOP",
        "REVERSE_LEFT",
        "REVERSE_RIGHT",
    )

    NUM_ACTIONS: int = 7

    # Inference settings
    ML_CONFIDENCE_THRESHOLD: float = 0.7  # Minimum confidence to trust ML decision
    USE_RULE_FALLBACK: bool = True        # Fall back to rules if ML confidence low

    # =========================================================================
    # SAFETY CONFIGURATION
    # =========================================================================

    # Hard safety limits (cannot be overridden by ML)
    SAFETY_STOP_DISTANCE_CM: float = 20.0  # Always stop if closer
    SAFETY_PERSON_STOP: bool = True        # Always stop for people
    SAFETY_SENSOR_ERROR_STOP: bool = True  # Stop on sensor errors

    # Maximum time without valid sensor reading before emergency stop
    SENSOR_TIMEOUT_EMERGENCY_S: float = 1.0

    # Maximum allowed speed near obstacles
    MAX_SPEED_NEAR_OBSTACLE: int = 30

    # =========================================================================
    # MAIN LOOP CONFIGURATION
    # =========================================================================

    # Target loop frequencies (Hz)
    MAIN_LOOP_FREQUENCY_HZ: float = 10.0
    CAMERA_LOOP_FREQUENCY_HZ: float = 5.0
    LOGGING_FREQUENCY_HZ: float = 2.0

    # Loop timing
    MAIN_LOOP_PERIOD_S: float = 0.1      # 100ms main loop
    SENSOR_READ_TIMEOUT_S: float = 0.1   # Max time for sensor read cycle

    # =========================================================================
    # GPS CONFIGURATION (Optional)
    # =========================================================================

    GPS_SERIAL_PORT: str = "/dev/ttyAMA0"
    GPS_BAUD_RATE: int = 9600
    GPS_TIMEOUT_S: float = 1.0
    GPS_ENABLED: bool = False  # Set True if GPS module is connected

    # =========================================================================
    # LOGGING CONFIGURATION
    # =========================================================================

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_TO_FILE: bool = True
    LOG_TO_CONSOLE: bool = True
    LOG_ROTATION_SIZE_MB: int = 10
    LOG_RETENTION_COUNT: int = 5

    # =========================================================================
    # TRAINING CONFIGURATION
    # =========================================================================

    # Synthetic data generation
    SYNTHETIC_SAMPLES: int = 50000
    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT: float = 0.1
    TEST_SPLIT: float = 0.1

    # Sensor noise simulation
    SENSOR_NOISE_STD_CM: float = 5.0
    FALSE_READING_PROBABILITY: float = 0.01

    # Model architecture
    HIDDEN_LAYER_SIZES: Tuple[int, ...] = (64, 32, 16)
    DROPOUT_RATE: float = 0.2
    LEARNING_RATE: float = 0.001

    # Training parameters
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 10

    # =========================================================================
    # SYSTEM CONFIGURATION
    # =========================================================================

    # Raspberry Pi specific
    IS_RASPBERRY_PI: bool = field(default_factory=lambda: os.path.exists("/proc/device-tree/model"))

    # Debug mode (more verbose logging, safety overrides disabled)
    DEBUG_MODE: bool = False

    # Simulation mode (no GPIO, use mock sensors)
    SIMULATION_MODE: bool = field(default_factory=lambda: not os.path.exists("/proc/device-tree/model"))


# Singleton pattern for settings
_settings_instance: Optional[Settings] = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get the global settings instance.

    Uses a singleton pattern with LRU cache for efficiency.
    Settings are immutable (frozen dataclass) for thread safety.

    Returns:
        The global Settings instance.
    """
    return Settings()


def get_distance_zone(distance_cm: float) -> str:
    """
    Classify a distance measurement into a safety zone.

    Args:
        distance_cm: Distance in centimeters.

    Returns:
        Zone name: "CRITICAL", "DANGER", "CAUTION", or "CLEAR"
    """
    settings = get_settings()

    if distance_cm < settings.CRITICAL_DISTANCE_CM:
        return "CRITICAL"
    elif distance_cm < settings.DANGER_DISTANCE_CM:
        return "DANGER"
    elif distance_cm < settings.CAUTION_DISTANCE_CM:
        return "CAUTION"
    else:
        return "CLEAR"
