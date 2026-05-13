#!/usr/bin/env python3
"""
Object detection using YOLOv8n (pretrained on COCO).
Transfer learning: we use it out-of-the-box to detect people, cars, etc.

Provides features for downstream ML decision model:
    - object_detected (0/1)
    - object_class (int: class id)
    - object_distance_px_area (proxy for proximity, normalized)
    - object_position (0=left, 1=center, 2=right)

Install: pip install ultralytics
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None


# Classes considered "obstacles / important"
# (COCO class ids). Person = 0.
IMPORTANT_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    15: 'cat',
    16: 'dog',
    56: 'chair',
    57: 'couch',
    59: 'bed',
    60: 'dining table',
}


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2
    area_ratio: float                         # bbox area / frame area (proxy for proximity)
    position: int                             # 0=left, 1=center, 2=right


@dataclass
class VisionFeatures:
    """Condensed vision features fed to the decision model."""
    object_detected: int          # 0 or 1
    nearest_class_id: int         # -1 if none
    nearest_area_ratio: float     # 0-1 (bigger = closer)
    nearest_position: int         # 0/1/2
    person_detected: int          # 0 or 1 (hard safety signal)
    num_objects: int


class ObjectDetector:
    def __init__(self, model_path: str = 'yolov8n.pt',
                 conf_threshold: float = 0.4,
                 device: str = 'cpu'):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. pip install ultralytics")
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None:
            return []
        h, w = frame.shape[:2]
        frame_area = float(h * w)

        results = self.model.predict(frame, conf=self.conf_threshold,
                                     verbose=False, device=self.device)[0]

        detections = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in IMPORTANT_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            area = (x2 - x1) * (y2 - y1)
            area_ratio = area / frame_area

            # position based on bbox center x
            cx = (x1 + x2) / 2
            if cx < w / 3:
                pos = 0
            elif cx < 2 * w / 3:
                pos = 1
            else:
                pos = 2

            detections.append(Detection(
                class_id=cls_id,
                class_name=IMPORTANT_CLASSES[cls_id],
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                area_ratio=area_ratio,
                position=pos,
            ))

        detections.sort(key=lambda d: d.area_ratio, reverse=True)
        return detections

    def extract_features(self, frame: np.ndarray) -> VisionFeatures:
        dets = self.detect(frame)
        if not dets:
            return VisionFeatures(0, -1, 0.0, 1, 0, 0)
        nearest = dets[0]
        person = 1 if any(d.class_id == 0 for d in dets) else 0
        return VisionFeatures(
            object_detected=1,
            nearest_class_id=nearest.class_id,
            nearest_area_ratio=nearest.area_ratio,
            nearest_position=nearest.position,
            person_detected=person,
            num_objects=len(dets),
        )

    @staticmethod
    def draw(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        import cv2
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d.bbox]
            color = (0, 0, 255) if d.class_id == 0 else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame


if __name__ == "__main__":
    import cv2
    from camera import Camera

    det = ObjectDetector()
    cam = Camera(device=0)
    cam.start()

    while True:
        frame, _ = cam.read()
        if frame is None:
            continue
        detections = det.detect(frame)
        features = det.extract_features(frame)
        frame = det.draw(frame, detections)
        info = f"objects={features.num_objects} person={features.person_detected} area={features.nearest_area_ratio:.3f}"
        cv2.putText(frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()
