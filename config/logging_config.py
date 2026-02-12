"""
Logging Configuration for Autonomous Solar Vehicle.

This module provides structured logging setup using Python's logging module
with optional structured logging via structlog.

Features:
- Console and file logging
- Log rotation (by size)
- Different log levels per handler
- Structured context logging for ML decisions
- Thread-safe operation

Usage:
    from config import setup_logging, get_logger
    setup_logging()
    logger = get_logger(__name__)
    logger.info("Sensor reading", distance=42.5, sensor="US_FRONT")
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict

# Try to import structlog for structured logging
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    rotation_size_mb: int = 10,
    retention_count: int = 5,
) -> None:
    """
    Configure logging for the autonomous vehicle system.

    Sets up both console and file handlers with appropriate formatting.
    File logs are rotated by size to prevent disk space issues on RPi.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files. Defaults to PROJECT_ROOT/logs
        log_to_file: Whether to write logs to file
        log_to_console: Whether to output logs to console
        rotation_size_mb: Max log file size before rotation (in MB)
        retention_count: Number of rotated log files to keep
    """
    # Determine log directory
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"

    # Create log directories
    runtime_log_dir = log_dir / "runtime"
    error_log_dir = log_dir / "errors"
    runtime_log_dir.mkdir(parents=True, exist_ok=True)
    error_log_dir.mkdir(parents=True, exist_ok=True)

    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatter
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    simple_formatter = logging.Formatter(
        fmt="%(levelname)-8s | %(name)-15s | %(message)s"
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all, filter at handler level

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)

    # File handlers
    if log_to_file:
        # Main runtime log (all levels >= configured level)
        runtime_log_path = runtime_log_dir / "vehicle.log"
        runtime_handler = logging.handlers.RotatingFileHandler(
            filename=runtime_log_path,
            maxBytes=rotation_size_mb * 1024 * 1024,
            backupCount=retention_count,
            encoding="utf-8",
        )
        runtime_handler.setLevel(numeric_level)
        runtime_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(runtime_handler)

        # Error log (only errors and above)
        error_log_path = error_log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            filename=error_log_path,
            maxBytes=rotation_size_mb * 1024 * 1024,
            backupCount=retention_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)

        # Decision log (for ML decision analysis)
        decision_log_path = runtime_log_dir / "decisions.log"
        decision_handler = logging.handlers.RotatingFileHandler(
            filename=decision_log_path,
            maxBytes=rotation_size_mb * 1024 * 1024,
            backupCount=retention_count,
            encoding="utf-8",
        )
        decision_handler.setLevel(logging.INFO)
        decision_handler.setFormatter(detailed_formatter)
        decision_logger = logging.getLogger("decision")
        decision_logger.addHandler(decision_handler)
        decision_logger.propagate = False  # Don't duplicate to main log

    # Configure structlog if available
    if STRUCTLOG_AVAILABLE:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    logging.info(f"Logging configured: level={log_level}, file={log_to_file}, console={log_to_console}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


class DecisionLogger:
    """
    Specialized logger for ML decision events.

    Provides structured logging of all decisions for later analysis,
    including sensor inputs, model outputs, and final actions.

    Usage:
        decision_logger = DecisionLogger()
        decision_logger.log_decision(
            features=feature_vector,
            action="FORWARD",
            confidence=0.95,
            source="ML"
        )
    """

    def __init__(self):
        """Initialize the decision logger."""
        self._logger = logging.getLogger("decision")
        self._sequence_number = 0

    def log_decision(
        self,
        features: Dict[str, float],
        action: str,
        confidence: float,
        source: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a navigation decision.

        Args:
            features: Feature vector used for decision (sensor readings, etc.)
            action: Final action taken (FORWARD, STOP, TURN_LEFT, etc.)
            confidence: Model confidence for this decision (0.0-1.0)
            source: Decision source ("ML", "RULE", "SAFETY_GOVERNOR")
            additional_context: Any additional information to log
        """
        self._sequence_number += 1

        log_data = {
            "seq": self._sequence_number,
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "confidence": f"{confidence:.3f}",
            "source": source,
            "features": features,
        }

        if additional_context:
            log_data["context"] = additional_context

        # Format as a single-line JSON-like string for easy parsing
        feature_str = ", ".join(f"{k}={v:.1f}" for k, v in features.items() if isinstance(v, (int, float)))
        self._logger.info(
            f"seq={self._sequence_number} | action={action} | conf={confidence:.3f} | "
            f"src={source} | {feature_str}"
        )

    def log_safety_override(
        self,
        original_action: str,
        override_action: str,
        reason: str,
        sensor_data: Dict[str, float]
    ) -> None:
        """
        Log when safety governor overrides ML decision.

        Args:
            original_action: Action ML/rules wanted to take
            override_action: Action safety governor enforced
            reason: Why the override happened
            sensor_data: Current sensor readings
        """
        self._sequence_number += 1

        self._logger.warning(
            f"seq={self._sequence_number} | SAFETY_OVERRIDE | "
            f"original={original_action} | override={override_action} | "
            f"reason={reason} | front={sensor_data.get('front_distance', 'N/A'):.1f}cm"
        )

    def log_sensor_error(
        self,
        sensor_id: str,
        error_type: str,
        details: str
    ) -> None:
        """
        Log sensor errors that affect decision making.

        Args:
            sensor_id: Which sensor failed
            error_type: Type of error (timeout, invalid_reading, etc.)
            details: Additional error details
        """
        self._logger.error(
            f"SENSOR_ERROR | sensor={sensor_id} | type={error_type} | details={details}"
        )


# Create singleton decision logger
_decision_logger: Optional[DecisionLogger] = None


def get_decision_logger() -> DecisionLogger:
    """
    Get the singleton DecisionLogger instance.

    Returns:
        The global DecisionLogger instance.
    """
    global _decision_logger
    if _decision_logger is None:
        _decision_logger = DecisionLogger()
    return _decision_logger
