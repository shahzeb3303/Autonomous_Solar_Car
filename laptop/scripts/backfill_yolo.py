#!/usr/bin/env python3
"""
One-time script: run YOLO on every image in dataset.csv and add YOLO columns.
Idempotent — safe to re-run.
"""

import os
import sys
import time

import cv2
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vision.object_detector import ObjectDetector


def backfill(csv_path: str):
    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"Loaded {n} samples from {csv_path}")

    # If columns already exist, only fill missing rows
    # Use float dtype to allow yolo_area to hold 0.0-1.0 values
    yolo_cols = ['yolo_person', 'yolo_object', 'yolo_area', 'yolo_pos', 'yolo_count']
    for c in yolo_cols:
        if c not in df.columns:
            df[c] = -1.0  # sentinel for "not yet computed", float dtype
        else:
            df[c] = df[c].astype('float64')

    # Find rows that still need computing
    todo_mask = df['yolo_person'].isna() | (df['yolo_person'] == -1)
    todo_idx = df.index[todo_mask].tolist()
    print(f"Need to process: {len(todo_idx)} rows")
    if not todo_idx:
        print("All rows already have YOLO features. Done.")
        return

    print("Loading YOLO model...")
    det = ObjectDetector(conf_threshold=0.4, device='cpu')

    t0 = time.time()
    processed = 0
    last_print = t0
    for i in todo_idx:
        path = df.at[i, 'frame_path']
        if not os.path.exists(path):
            # Mark as no-detect (image missing)
            df.at[i, 'yolo_person'] = 0
            df.at[i, 'yolo_object'] = 0
            df.at[i, 'yolo_area']   = 0.0
            df.at[i, 'yolo_pos']    = 1
            df.at[i, 'yolo_count']  = 0
            continue

        frame = cv2.imread(path)
        if frame is None:
            df.at[i, 'yolo_person'] = 0
            df.at[i, 'yolo_object'] = 0
            df.at[i, 'yolo_area']   = 0.0
            df.at[i, 'yolo_pos']    = 1
            df.at[i, 'yolo_count']  = 0
            continue

        feats = det.extract_features(frame)
        df.at[i, 'yolo_person'] = feats.person_detected
        df.at[i, 'yolo_object'] = feats.object_detected
        df.at[i, 'yolo_area']   = feats.nearest_area_ratio
        df.at[i, 'yolo_pos']    = feats.nearest_position
        df.at[i, 'yolo_count']  = feats.num_objects

        processed += 1
        now = time.time()
        if now - last_print > 5.0:
            elapsed = now - t0
            rate = processed / max(elapsed, 0.001)
            eta = (len(todo_idx) - processed) / max(rate, 0.001)
            print(f"  [{processed}/{len(todo_idx)}] "
                  f"{rate:.1f} img/s, ETA {eta/60:.1f} min")
            last_print = now
            # Save progress periodically
            df.to_csv(csv_path, index=False)

    df.to_csv(csv_path, index=False)
    elapsed = time.time() - t0
    print(f"\nDone. Processed {processed} rows in {elapsed/60:.1f} min")

    # Print stats
    print("\nYOLO feature distribution:")
    print(f"  Frames with person:     {(df['yolo_person'] > 0).sum()}")
    print(f"  Frames with any object: {(df['yolo_object'] > 0).sum()}")
    print(f"  Frames with no object:  {(df['yolo_object'] == 0).sum()}")
    print(f"  Avg objects per frame:  {df['yolo_count'].mean():.2f}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'laptop/data/dataset.csv'
    backfill(csv_path)
