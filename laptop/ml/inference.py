#!/usr/bin/env python3
"""
Production inference wrapper. Loads ONNX model (fast) or PyTorch checkpoint.
Exposes a single `predict(image, sensors, gps, prev_action) -> action_id`.
"""

import os
import numpy as np
from PIL import Image
from typing import Optional, Union
from torchvision import transforms

from .dataset import build_state_vector, IMAGENET_MEAN, IMAGENET_STD
from .actions import ACTION_NAMES, STOP

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import torch
    from .model import DecisionModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _preprocess_image(frame_bgr, size: int = 224):
    """BGR numpy -> normalized tensor (1, 3, size, size) as float32 numpy."""
    import cv2
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return t(img).unsqueeze(0).numpy().astype(np.float32)


class ONNXInference:
    def __init__(self, onnx_path: str, providers=None):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed")
        providers = providers or ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(onnx_path, providers=providers)

    def predict_logits(self, image_np: np.ndarray, state_np: np.ndarray) -> np.ndarray:
        out = self.sess.run(['logits'], {'image': image_np, 'state': state_np})[0]
        return out  # (1, num_actions)


class TorchInference:
    def __init__(self, ckpt_path: str, device: str = 'cpu'):
        if not TORCH_AVAILABLE:
            raise ImportError("torch not installed")
        self.device = device
        self.model = DecisionModel(pretrained_backbone=False).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

    def predict_logits(self, image_np: np.ndarray, state_np: np.ndarray) -> np.ndarray:
        import torch as T
        with T.no_grad():
            img = T.from_numpy(image_np).to(self.device)
            st  = T.from_numpy(state_np.astype(np.float32)).to(self.device)
            logits = self.model(img, st).cpu().numpy()
        return logits


class Predictor:
    """Unified predictor. Use `model_path` ending in .onnx or .pt"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        if model_path.endswith('.onnx'):
            self.backend = ONNXInference(model_path)
            self.kind = 'onnx'
        else:
            self.backend = TorchInference(model_path, device)
            self.kind = 'torch'
        print(f"[Predictor] loaded {self.kind} model: {model_path}")

    def predict(self, frame_bgr, sensors: dict, gps_valid: int, gps_speed: float,
                gps_heading_deg: float, prev_action: int, yolo: dict = None) -> dict:
        """Returns dict with action_id, action_name, probs, confidence.

        yolo: optional dict with keys person_detected, object_detected,
              nearest_area_ratio, nearest_position. If None, defaults to zeros.
        """
        if frame_bgr is None:
            return {'action_id': STOP, 'action_name': ACTION_NAMES[STOP],
                    'probs': None, 'confidence': 0.0, 'reason': 'no_frame'}

        img = _preprocess_image(frame_bgr)
        state = build_state_vector(sensors, gps_valid, gps_speed,
                                   gps_heading_deg, prev_action, yolo)
        state = state[np.newaxis, :]  # (1, state_dim)

        logits = self.backend.predict_logits(img, state)[0]
        probs = _softmax(logits)
        action_id = int(np.argmax(probs))
        conf = float(probs[action_id])
        return {
            'action_id': action_id,
            'action_name': ACTION_NAMES[action_id],
            'probs': probs.tolist(),
            'confidence': conf,
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
