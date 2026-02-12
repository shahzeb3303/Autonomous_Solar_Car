"""
Camera Module for Autonomous Solar Vehicle.

This module provides camera capture and frame preprocessing for:
- Front camera (CSI interface via picamera2)
- Rear camera (USB interface via OpenCV)

Features:
- Thread-safe frame capture
- Automatic frame preprocessing for ML inference
- Support for both CSI and USB cameras
- Graceful fallback to mock mode for development

Hardware:
- Front: Raspberry Pi Camera Module v2 (CSI-0)
- Rear: Generic USB camera

Usage:
    from src.sensors.camera import CameraManager

    camera = CameraManager()
    camera.start()

    frame = camera.get_front_frame()
    preprocessed = camera.get_preprocessed_frame()

    camera.stop()
"""

import time
import threading
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

# Try to import camera libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    Picamera2 = None

from config.settings import get_settings

logger = logging.getLogger(__name__)


class CameraType(Enum):
    """Camera interface types."""
    CSI = "csi"      # Raspberry Pi Camera (CSI ribbon cable)
    USB = "usb"      # USB webcam
    MOCK = "mock"    # Mock camera for testing


@dataclass
class FrameMetadata:
    """Metadata for a captured frame."""
    timestamp: float
    width: int
    height: int
    camera_id: str
    frame_number: int


class MockFrame:
    """Mock frame generator for development/testing."""

    @staticmethod
    def generate(width: int = 640, height: int = 480, channels: int = 3) -> np.ndarray:
        """
        Generate a mock frame with simple patterns.

        Args:
            width: Frame width
            height: Frame height
            channels: Number of color channels (3 for RGB)

        Returns:
            Numpy array representing a mock frame.
        """
        # Create a gradient pattern
        frame = np.zeros((height, width, channels), dtype=np.uint8)

        # Add some variation for visual feedback
        for y in range(height):
            for x in range(0, width, 10):
                intensity = int((x / width) * 255)
                frame[y, x:x+10, :] = intensity

        # Add a center rectangle (simulating an object)
        center_x, center_y = width // 2, height // 2
        rect_size = 50
        frame[
            center_y - rect_size:center_y + rect_size,
            center_x - rect_size:center_x + rect_size,
            2  # Red channel
        ] = 200

        return frame


class Camera:
    """
    Generic camera interface.

    Provides a unified interface for different camera types (CSI, USB, mock).
    Handles frame capture, buffering, and basic preprocessing.

    Args:
        camera_id: Unique identifier for this camera
        camera_type: Type of camera (CSI, USB, or MOCK)
        width: Capture width
        height: Capture height
        fps: Target frames per second
    """

    def __init__(
        self,
        camera_id: str,
        camera_type: CameraType,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        device_index: int = 0,
    ):
        """Initialize the camera."""
        self._settings = get_settings()

        self.camera_id = camera_id
        self.camera_type = camera_type
        self.width = width
        self.height = height
        self.fps = fps
        self.device_index = device_index

        self._capture = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_number = 0

        logger.info(
            f"Camera created: id={camera_id}, type={camera_type.value}, "
            f"{width}x{height}@{fps}fps"
        )

    def start(self) -> bool:
        """
        Start the camera capture.

        Returns:
            True if camera started successfully.
        """
        try:
            with self._lock:
                if self._running:
                    return True

                if self.camera_type == CameraType.CSI:
                    return self._start_csi()
                elif self.camera_type == CameraType.USB:
                    return self._start_usb()
                else:
                    return self._start_mock()

        except Exception as e:
            logger.error(f"Failed to start camera {self.camera_id}: {e}")
            return False

    def _start_csi(self) -> bool:
        """Start CSI camera via picamera2."""
        if not PICAMERA_AVAILABLE:
            logger.warning("picamera2 not available, falling back to mock")
            return self._start_mock()

        try:
            self._capture = Picamera2(self.device_index)
            config = self._capture.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self._capture.configure(config)
            self._capture.start()
            self._running = True
            logger.info(f"CSI camera {self.camera_id} started")
            return True
        except Exception as e:
            logger.error(f"CSI camera error: {e}")
            return self._start_mock()

    def _start_usb(self) -> bool:
        """Start USB camera via OpenCV."""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, falling back to mock")
            return self._start_mock()

        try:
            self._capture = cv2.VideoCapture(self.device_index)
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)

            if not self._capture.isOpened():
                logger.warning("USB camera not available, falling back to mock")
                return self._start_mock()

            self._running = True
            logger.info(f"USB camera {self.camera_id} started")
            return True
        except Exception as e:
            logger.error(f"USB camera error: {e}")
            return self._start_mock()

    def _start_mock(self) -> bool:
        """Start mock camera for testing."""
        self.camera_type = CameraType.MOCK
        self._running = True
        logger.info(f"Mock camera {self.camera_id} started")
        return True

    def stop(self) -> None:
        """Stop the camera capture."""
        with self._lock:
            if not self._running:
                return

            if self.camera_type == CameraType.CSI and self._capture:
                try:
                    self._capture.stop()
                except Exception as e:
                    logger.warning(f"Error stopping CSI camera: {e}")

            elif self.camera_type == CameraType.USB and self._capture:
                try:
                    self._capture.release()
                except Exception as e:
                    logger.warning(f"Error stopping USB camera: {e}")

            self._capture = None
            self._running = False
            logger.info(f"Camera {self.camera_id} stopped")

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame.

        Returns:
            Numpy array of shape (height, width, 3) in RGB format,
            or None if capture fails.
        """
        with self._lock:
            if not self._running:
                return None

            try:
                if self.camera_type == CameraType.CSI:
                    frame = self._capture.capture_array()
                elif self.camera_type == CameraType.USB:
                    ret, frame = self._capture.read()
                    if not ret:
                        return None
                    # OpenCV captures in BGR, convert to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame = MockFrame.generate(self.width, self.height)

                self._latest_frame = frame
                self._frame_number += 1
                return frame

            except Exception as e:
                logger.error(f"Frame capture error on {self.camera_id}: {e}")
                return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recently captured frame.

        Returns:
            Latest frame or None if no frame captured yet.
        """
        return self._latest_frame

    def get_frame_metadata(self) -> Optional[FrameMetadata]:
        """Get metadata for the latest frame."""
        if self._latest_frame is None:
            return None

        return FrameMetadata(
            timestamp=time.time(),
            width=self.width,
            height=self.height,
            camera_id=self.camera_id,
            frame_number=self._frame_number,
        )

    @property
    def is_running(self) -> bool:
        """Check if camera is currently running."""
        return self._running

    @property
    def frame_count(self) -> int:
        """Get total number of frames captured."""
        return self._frame_number


class CameraManager:
    """
    Manager for multiple cameras with preprocessing.

    Provides coordinated access to front and rear cameras with:
    - Automatic frame preprocessing for ML inference
    - Thread-safe operation
    - Configurable frame skipping

    Usage:
        manager = CameraManager()
        manager.start()

        frame = manager.get_front_frame()
        preprocessed = manager.get_preprocessed_for_detection(frame)

        manager.stop()
    """

    def __init__(self):
        """Initialize the camera manager."""
        self._settings = get_settings()

        # Create front camera (CSI)
        self._front_camera = Camera(
            camera_id="CAM_FRONT",
            camera_type=CameraType.CSI,
            width=self._settings.CAMERA_FRONT_WIDTH,
            height=self._settings.CAMERA_FRONT_HEIGHT,
            fps=self._settings.CAMERA_FRONT_FPS,
            device_index=0,
        )

        # Create rear camera (USB)
        self._rear_camera = Camera(
            camera_id="CAM_REAR",
            camera_type=CameraType.USB,
            width=self._settings.CAMERA_REAR_WIDTH,
            height=self._settings.CAMERA_REAR_HEIGHT,
            fps=self._settings.CAMERA_REAR_FPS,
            device_index=0,
        )

        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame_skip_counter = 0

        logger.info("CameraManager initialized")

    def start(self) -> Dict[str, bool]:
        """
        Start all cameras.

        Returns:
            Dictionary mapping camera_id to start success status.
        """
        results = {}

        results["CAM_FRONT"] = self._front_camera.start()
        results["CAM_REAR"] = self._rear_camera.start()

        self._running = any(results.values())

        if self._running:
            logger.info("CameraManager started")
        else:
            logger.warning("No cameras could be started")

        return results

    def stop(self) -> None:
        """Stop all cameras."""
        self._running = False

        self._front_camera.stop()
        self._rear_camera.stop()

        logger.info("CameraManager stopped")

    def get_front_frame(self) -> Optional[np.ndarray]:
        """
        Get a frame from the front camera.

        Returns:
            RGB frame array or None if unavailable.
        """
        return self._front_camera.capture_frame()

    def get_rear_frame(self) -> Optional[np.ndarray]:
        """
        Get a frame from the rear camera.

        Returns:
            RGB frame array or None if unavailable.
        """
        return self._rear_camera.capture_frame()

    def get_front_frame_skipped(self) -> Optional[np.ndarray]:
        """
        Get front frame with frame skipping for efficiency.

        Only returns a new frame every CAMERA_SKIP_FRAMES captures.

        Returns:
            RGB frame array, or None if skipped or unavailable.
        """
        self._frame_skip_counter += 1

        if self._frame_skip_counter >= self._settings.CAMERA_SKIP_FRAMES:
            self._frame_skip_counter = 0
            return self.get_front_frame()

        return None

    def preprocess_for_detection(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for object detection model input.

        Resizes and normalizes frame for MobileNet SSD input.

        Args:
            frame: RGB frame array (H, W, 3)

        Returns:
            Preprocessed frame ready for model input (300, 300, 3)
        """
        if frame is None:
            # Return blank frame if input is None
            return np.zeros(
                (self._settings.DETECTION_INPUT_HEIGHT,
                 self._settings.DETECTION_INPUT_WIDTH, 3),
                dtype=np.uint8
            )

        # Resize to model input size
        if CV2_AVAILABLE:
            resized = cv2.resize(
                frame,
                (self._settings.DETECTION_INPUT_WIDTH,
                 self._settings.DETECTION_INPUT_HEIGHT),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            # Simple numpy resize fallback
            resized = self._numpy_resize(
                frame,
                self._settings.DETECTION_INPUT_HEIGHT,
                self._settings.DETECTION_INPUT_WIDTH
            )

        return resized

    def _numpy_resize(
        self,
        image: np.ndarray,
        new_height: int,
        new_width: int
    ) -> np.ndarray:
        """
        Simple image resize using numpy (fallback when OpenCV unavailable).

        Uses nearest-neighbor interpolation for simplicity.
        """
        height, width = image.shape[:2]

        # Calculate indices for nearest-neighbor
        y_indices = (np.arange(new_height) * height / new_height).astype(int)
        x_indices = (np.arange(new_width) * width / new_width).astype(int)

        return image[y_indices[:, None], x_indices, :]

    def get_detection_ready_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get both original and preprocessed frames for detection.

        Returns:
            Tuple of (original_frame, preprocessed_frame), or None if unavailable.
        """
        frame = self.get_front_frame()
        if frame is None:
            return None

        preprocessed = self.preprocess_for_detection(frame)
        return (frame, preprocessed)

    @property
    def front_camera(self) -> Camera:
        """Get front camera instance."""
        return self._front_camera

    @property
    def rear_camera(self) -> Camera:
        """Get rear camera instance."""
        return self._rear_camera

    @property
    def is_running(self) -> bool:
        """Check if any camera is running."""
        return self._running
