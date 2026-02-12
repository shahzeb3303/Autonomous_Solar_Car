"""
Perception Module for Autonomous Solar Vehicle.

This module provides computer vision and object detection:
- ObjectDetector: TF Lite MobileNet SSD inference
- DistanceEstimator: Estimate object distance from bounding box
- ThreatAssessor: Classify detected objects by threat level

All perception classes are optimized for Raspberry Pi 4 inference.
"""

from src.perception.object_detector import ObjectDetector, Detection
from src.perception.distance_estimator import DistanceEstimator
from src.perception.threat_assessor import ThreatAssessor, ThreatLevel

__all__ = [
    "ObjectDetector",
    "Detection",
    "DistanceEstimator",
    "ThreatAssessor",
    "ThreatLevel",
]
