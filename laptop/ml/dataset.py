#!/usr/bin/env python3
"""
Dataset for the autonomous driving model.

CSV layout (produced by data_collection/record.py):
    frame_path, FL, FR, FW, BC, LS, RS,
    gps_valid, gps_speed, gps_heading,
    prev_action, action_label

Split is per-session to avoid data leakage between consecutive frames.
Augmentation: horizontal flip with action-remap (LEFT<->RIGHT), color jitter,
noise on sensor values.
"""

import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .actions import (FORWARD, SLOW_DOWN, TURN_LEFT, TURN_RIGHT, STOP,
                      REVERSE_LEFT, REVERSE_RIGHT, REVERSE, NUM_ACTIONS)


# Normalization constants
MAX_DISTANCE_CM = 400.0
MAX_SPEED_MPS = 5.0

# Horizontal flip mapping (LEFT <-> RIGHT actions)
FLIP_ACTION_MAP = {
    FORWARD:       FORWARD,
    SLOW_DOWN:     SLOW_DOWN,
    TURN_LEFT:     TURN_RIGHT,
    TURN_RIGHT:    TURN_LEFT,
    STOP:          STOP,
    REVERSE_LEFT:  REVERSE_RIGHT,
    REVERSE_RIGHT: REVERSE_LEFT,
    REVERSE:       REVERSE,
}

# Sensor flip mapping: FL<->FR, LS<->RS (FW and BC unchanged)
def flip_sensors(sensors: dict) -> dict:
    flipped = dict(sensors)
    flipped['FL'], flipped['FR'] = sensors['FR'], sensors['FL']
    flipped['LS'], flipped['RS'] = sensors['RS'], sensors['LS']
    return flipped


def build_state_vector(sensors: dict, gps_valid: int, gps_speed: float,
                       gps_heading_deg: float, prev_action: int,
                       yolo: dict = None) -> np.ndarray:
    """
    Build normalized state vector for the model.

    Ordering (must match model.STATE_DIM):
      [FL/max, FR/max, FW/max, BC/max, LS/max, RS/max,
       front_min/max, back_min/max,
       gps_valid, gps_speed/max, gps_heading_rad/(2*pi),
       yolo_person, yolo_object, yolo_area_ratio, yolo_position/2,
       prev_action_onehot x NUM_ACTIONS]
    """
    u = [sensors['FL'], sensors['FR'], sensors['FW'],
         sensors['BC'], sensors['LS'], sensors['RS']]
    u = [max(0.0, min(MAX_DISTANCE_CM, v)) / MAX_DISTANCE_CM for v in u]
    front_vals = [v for v in [sensors['FL'], sensors['FR'], sensors['FW']] if 2 <= v <= MAX_DISTANCE_CM]
    back_vals  = [v for v in [sensors['BC']] if 2 <= v <= MAX_DISTANCE_CM]
    front_min = (min(front_vals) / MAX_DISTANCE_CM) if front_vals else 1.0
    back_min  = (min(back_vals) / MAX_DISTANCE_CM) if back_vals else 1.0

    gps_speed_n = max(0.0, min(MAX_SPEED_MPS, gps_speed)) / MAX_SPEED_MPS
    heading_rad = math.radians(gps_heading_deg % 360.0) / (2 * math.pi)

    # YOLO features (default zeros if not provided)
    if yolo is None:
        yolo = {}
    yolo_person = float(yolo.get('person_detected', 0))
    yolo_obj    = float(yolo.get('object_detected', 0))
    yolo_area   = float(yolo.get('nearest_area_ratio', 0.0))
    yolo_pos    = float(yolo.get('nearest_position', 1)) / 2.0  # normalize 0/1/2 → 0/0.5/1

    prev_oh = np.zeros(NUM_ACTIONS, dtype=np.float32)
    if 0 <= prev_action < NUM_ACTIONS:
        prev_oh[prev_action] = 1.0

    vec = np.array(u + [front_min, back_min, float(gps_valid),
                        gps_speed_n, heading_rad,
                        yolo_person, yolo_obj, yolo_area, yolo_pos],
                   dtype=np.float32)
    return np.concatenate([vec, prev_oh])


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(train: bool, img_size: int = 224):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@dataclass
class Sample:
    image_path: str
    sensors: dict
    gps_valid: int
    gps_speed: float
    gps_heading: float
    prev_action: int
    label: int
    yolo: dict


class DrivingDataset(Dataset):
    """
    Supervised dataset: (image, state_vector) -> action label.
    Horizontal-flip augmentation is applied per-sample during training with p=0.5;
    sensor left/right channels and action label are flipped accordingly.
    """

    def __init__(self, csv_path: str, train: bool = True,
                 sensor_noise_std: float = 0.02, flip_prob: float = 0.5):
        self.csv_path = csv_path
        self.train = train
        self.sensor_noise_std = sensor_noise_std
        self.flip_prob = flip_prob
        self.transform = build_transforms(train=train)
        self.samples: List[Sample] = self._load(csv_path)

    @staticmethod
    def _load(csv_path: str) -> List[Sample]:
        df = pd.read_csv(csv_path)
        samples = []
        for _, r in df.iterrows():
            samples.append(Sample(
                image_path=r['frame_path'],
                sensors={'FL': float(r['FL']), 'FR': float(r['FR']),
                         'FW': float(r['FW']), 'BC': float(r['BC']),
                         'LS': float(r['LS']), 'RS': float(r['RS'])},
                gps_valid=int(r.get('gps_valid', 0)),
                gps_speed=float(r.get('gps_speed', 0.0)),
                gps_heading=float(r.get('gps_heading', 0.0)),
                prev_action=int(r.get('prev_action', STOP)),
                label=int(r['action_label']),
                yolo={
                    'person_detected':    int(r.get('yolo_person', 0)),
                    'object_detected':    int(r.get('yolo_object', 0)),
                    'nearest_area_ratio': float(r.get('yolo_area', 0.0)),
                    'nearest_position':   int(r.get('yolo_pos', 1)),
                },
            ))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert('RGB')

        sensors = s.sensors
        label = s.label
        prev_action = s.prev_action
        yolo = dict(s.yolo)

        # Horizontal flip augmentation
        flip = self.train and random.random() < self.flip_prob
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            sensors = flip_sensors(sensors)
            label = FLIP_ACTION_MAP[label]
            prev_action = FLIP_ACTION_MAP[prev_action]
            # Flip YOLO position too (left ↔ right)
            yolo['nearest_position'] = 2 - yolo.get('nearest_position', 1)

        img_t = self.transform(img)

        # Sensor noise augmentation (on normalized state vector)
        state = build_state_vector(sensors, s.gps_valid, s.gps_speed,
                                   s.gps_heading, prev_action, yolo)
        if self.train and self.sensor_noise_std > 0:
            # Only add noise to the first 8 features (distances); keep one-hot crisp
            noise = np.random.normal(0, self.sensor_noise_std, 8).astype(np.float32)
            state[:8] = np.clip(state[:8] + noise, 0.0, 1.0)

        return {
            'image': img_t,
            'state': torch.from_numpy(state).float(),
            'label': torch.tensor(label, dtype=torch.long),
        }


def class_weights(labels: List[int], num_classes: int = NUM_ACTIONS) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss.
    Classes with 0 samples get weight 0 (not penalized/rewarded)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    present = counts > 0
    w = np.zeros(num_classes, dtype=np.float32)
    if present.any():
        w[present] = 1.0 / counts[present]
        # Normalize so average weight over present classes = 1
        w = w / w[present].mean()
    return torch.from_numpy(w).float()


def split_csv(csv_path: str, out_dir: str,
              train_ratio: float = 0.70, val_ratio: float = 0.15,
              seed: int = 42) -> dict:
    """Shuffle-split a single CSV into train/val/test CSVs (file paths returned)."""
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train, val, test = df[:n_train], df[n_train:n_train + n_val], df[n_train + n_val:]
    paths = {
        'train': os.path.join(out_dir, 'train.csv'),
        'val':   os.path.join(out_dir, 'val.csv'),
        'test':  os.path.join(out_dir, 'test.csv'),
    }
    train.to_csv(paths['train'], index=False)
    val.to_csv(paths['val'], index=False)
    test.to_csv(paths['test'], index=False)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return paths


if __name__ == "__main__":
    # Quick smoke test (needs existing CSV)
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m laptop.ml.dataset <csv_path>")
        raise SystemExit(0)
    ds = DrivingDataset(sys.argv[1], train=True)
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print(f"image shape: {sample['image'].shape}")
    print(f"state shape: {sample['state'].shape}")
    print(f"label: {sample['label'].item()}")
