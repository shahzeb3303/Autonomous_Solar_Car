#!/usr/bin/env python3
"""
USB Webcam capture (runs on laptop).
Threaded to always have the freshest frame with no blocking.
"""

import cv2
import threading
import time
from typing import Optional


class Camera:
    def __init__(self, device: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame = None
        self.frame_ts = 0.0
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            print(f"[Camera] Cannot open device {self.device}")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        time.sleep(0.5)  # let capture warm up
        print(f"[Camera] Started device {self.device} @ {self.width}x{self.height}")
        return True

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
                    self.frame_ts = time.time()
            else:
                time.sleep(0.01)

    def read(self):
        """Return (frame, timestamp) or (None, 0)."""
        with self.lock:
            if self.frame is None:
                return None, 0.0
            return self.frame.copy(), self.frame_ts

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    cam = Camera(device=0)
    if not cam.start():
        raise SystemExit(1)

    print("Press 'q' to quit")
    try:
        while True:
            frame, ts = cam.read()
            if frame is not None:
                cv2.imshow("Camera Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)
    finally:
        cam.stop()
        cv2.destroyAllWindows()
