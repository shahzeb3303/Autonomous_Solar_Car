#!/usr/bin/env python3
"""Build a 3-panel Figure 3 for the IJIST submission:
   (a) Pedestrian | (b) Vehicle | (c) Obstacle
EigenCAM saliency + class-filtered bbox per panel. Plain 640x640 resize
(no letterbox, no center-crop loss) so the full source content is visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.gradcam_vehicle_obstacle import (  # noqa: E402
    YOLOFwd, OBSTACLE_CLASSES, VEHICLE_CLASSES, REPO, SIZE, TARGET_LAYER_IDX,
)
import torch.nn as nn
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO

PEDESTRIAN_CLASSES = {0}
OUT_DIR = REPO / "gradcam_outputs"


def to_input(bgr: np.ndarray):
    import torch
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


def render_panel(yolo: YOLO, target_layer: nn.Module, img_path: Path,
                 target_set: set) -> tuple[np.ndarray, dict]:
    bgr_full = cv2.imread(str(img_path))
    if bgr_full is None:
        raise FileNotFoundError(img_path)
    # Plain resize (no center-crop, no letterbox) — full content visible, no gray bars.
    bgr = cv2.resize(bgr_full, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    rgb_float = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    wrapper = YOLOFwd(yolo.model).eval()
    cam = EigenCAM(model=wrapper, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=to_input(bgr))[0]
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    panel = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    res = yolo.predict(bgr, conf=0.4, verbose=False, device="cpu")[0]
    chosen = {}
    if res.boxes is not None:
        names = res.names
        # Sort target-class boxes by confidence so highest-conf box is on top.
        boxes = sorted(
            ((int(b.cls[0]), float(b.conf[0]), [int(v) for v in b.xyxy[0]])
             for b in res.boxes),
            key=lambda t: t[1], reverse=True,
        )
        for cid, conf, (x1, y1, x2, y2) in boxes:
            if cid not in target_set:
                continue
            cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{names.get(cid, cid)} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(th + 6, y1)
            cv2.rectangle(panel, (x1, ty - th - 6), (x1 + tw + 4, ty),
                          (0, 255, 0), -1)
            cv2.putText(panel, label, (x1 + 2, ty - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            if not chosen:
                chosen = dict(name=names.get(cid, cid), conf=conf,
                              bbox=(x1, y1, x2, y2))
    return panel, chosen


def add_border(p, t=2):
    return cv2.copyMakeBorder(p, t, t, t, t, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def build_three(panels, captions, sep=14, cap_h=56):
    bordered = [add_border(p) for p in panels]
    h = bordered[0].shape[0]
    sep_strip = np.full((h, sep, 3), 255, np.uint8)
    row_parts = []
    for i, p in enumerate(bordered):
        row_parts.append(p)
        if i != len(bordered) - 1:
            row_parts.append(sep_strip)
    row = np.hstack(row_parts)

    cap_band = np.full((cap_h, row.shape[1], 3), 255, np.uint8)
    panel_w = bordered[0].shape[1]
    centers = [panel_w // 2,
               panel_w + sep + panel_w // 2,
               2 * panel_w + 2 * sep + panel_w // 2]
    for text, cx in zip(captions, centers):
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(cap_band, text, (cx - tw // 2, (cap_h + th) // 2),
                    font, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return np.vstack([row, cap_band])


def main():
    OUT_DIR.mkdir(exist_ok=True)
    pedestrian = REPO / "data/images/20260420_183543_0692f005.jpg"
    vehicle = REPO / "data/images/20260420_183543_0b90b94c.jpg"
    obstacle = REPO / "data/images/20260420_183543_10ae8261.jpg"

    print(f"pedestrian: {pedestrian.name}")
    print(f"vehicle:    {vehicle.name}")
    print(f"obstacle:   {obstacle.name}")

    print("\nLoading YOLOv8n (pretrained COCO)...")
    yolo = YOLO("yolov8n.pt")
    yolo.model.eval()
    target_layer = yolo.model.model[TARGET_LAYER_IDX]

    print("\nRendering panels...")
    p_panel, p_det = render_panel(yolo, target_layer, pedestrian, PEDESTRIAN_CLASSES)
    print(f"  pedestrian: {p_det}")
    v_panel, v_det = render_panel(yolo, target_layer, vehicle, VEHICLE_CLASSES)
    print(f"  vehicle:    {v_det}")
    o_panel, o_det = render_panel(yolo, target_layer, obstacle, OBSTACLE_CLASSES)
    print(f"  obstacle:   {o_det}")

    cv2.imwrite(str(OUT_DIR / "pedestrian.jpg"), p_panel)
    cv2.imwrite(str(OUT_DIR / "vehicle.jpg"), v_panel)
    cv2.imwrite(str(OUT_DIR / "obstacle.jpg"), o_panel)

    combined = build_three(
        [p_panel, v_panel, o_panel],
        ["(a) Pedestrian", "(b) Vehicle", "(c) Obstacle"],
    )
    out_path = OUT_DIR / "figure3_three_panel.jpg"
    cv2.imwrite(str(out_path), combined)
    print(f"\nSaved: {out_path}")
    print(f"Size:  {combined.shape[1]}x{combined.shape[0]}")


if __name__ == "__main__":
    main()
