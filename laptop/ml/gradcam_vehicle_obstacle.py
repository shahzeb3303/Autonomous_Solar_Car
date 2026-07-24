#!/usr/bin/env python3
"""
Generate EigenCAM saliency panels for two class groups — "vehicle" and
"obstacle" — to be paired with the existing pedestrian panel as Figure 3
of the IJIST submission.

Pipeline per panel:
  1. center-crop the source frame to a square, resize to 640x640
  2. run YOLOv8n inference, keep only target-group detections
  3. compute EigenCAM at model.model.model[21] (last C2f) via pytorch-grad-cam
  4. overlay heatmap + draw target-class bboxes only
  5. save 640x640 JPG

Outputs (relative to repo root):
  gradcam_outputs/vehicle.jpg
  gradcam_outputs/obstacle.jpg
  gradcam_outputs/v_o_combined.jpg   (white 14px separator, black borders,
                                      captions "(b) Vehicle" / "(c) Obstacle")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO / "data" / "images"
OUT_DIR = REPO / "gradcam_outputs"
SIZE = 640
TARGET_LAYER_IDX = 21

# COCO ids — the pretrained YOLOv8n was trained on COCO. Group its classes
# into the paper's two semantic categories.
VEHICLE_CLASSES = {1, 2, 3, 5, 7}              # bicycle, car, motorcycle, bus, truck
OBSTACLE_CLASSES = {11, 13, 28, 56, 57, 58, 59, 60}
# stop sign, bench, suitcase, chair, couch, potted plant, bed, dining table
# (the campus footage contains cones/barriers/planters/benches; COCO's nearest
# matches end up firing on those even though they aren't literal "obstacle")


# ---------- IO + preprocessing ----------------------------------------------

def center_crop_resize(img_bgr: np.ndarray, size: int = SIZE,
                       crop: bool = True) -> np.ndarray:
    """If ``crop`` is True: take a center square and resize. Otherwise resize
    directly (slight aspect distortion but no content loss, no gray bars)."""
    if crop:
        h, w = img_bgr.shape[:2]
        s = min(h, w)
        top = (h - s) // 2
        left = (w - s) // 2
        img_bgr = img_bgr[top:top + s, left:left + s]
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)


def to_input_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return t


def sharpness(img_bgr: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())


# ---------- YOLO wrapper for pytorch-grad-cam --------------------------------

class YOLOFwd(nn.Module):
    """pytorch-grad-cam expects a Module returning a Tensor; YOLOv8's inner
    module returns (preds, intermediates) in eval mode. Strip to preds only."""

    def __init__(self, ym: nn.Module):
        super().__init__()
        self.m = ym

    def forward(self, x):
        out = self.m(x)
        return out[0] if isinstance(out, (tuple, list)) else out


# ---------- candidate selection ---------------------------------------------

def _det_table_row(p: Path, dets) -> Dict:
    return dict(
        path=p,
        classes=sorted({d["name"] for d in dets}),
        confs={d["name"]: round(d["conf"], 2) for d in dets},
        raw=dets,
    )


def find_candidates(yolo: YOLO, target_set: set, conf: float = 0.5,
                    sample_limit: int = 1500,
                    min_sharp: float = 350.0,
                    min_area_ratio: float = 0.04) -> List[Dict]:
    """Return scored candidate images for a target class set."""
    imgs = sorted(IMAGES_DIR.glob("*.jpg"))[:sample_limit]
    rows = []
    for p in imgs:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        if sharpness(bgr) < min_sharp:
            continue
        # Run on the same 640 center-crop the figure will use, so detections
        # match what the panel actually shows.
        cropped = center_crop_resize(bgr, SIZE)
        res = yolo.predict(cropped, conf=conf, verbose=False, device="cpu")[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        names = res.names
        dets = []
        for b in res.boxes:
            cid = int(b.cls[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            dets.append(dict(
                cls_id=cid, name=names.get(cid, str(cid)),
                conf=float(b.conf[0]), bbox=(x1, y1, x2, y2),
                area=(x2 - x1) * (y2 - y1) / (SIZE * SIZE),
            ))
        target_dets = [d for d in dets if d["cls_id"] in target_set]
        if not target_dets:
            continue
        target_dets.sort(key=lambda d: d["conf"] * (0.5 + d["area"]), reverse=True)
        top = target_dets[0]
        if top["area"] < min_area_ratio:
            continue
        # Score: confidence + bbox prominence + bonus if target is the dominant detection
        is_dominant = top["conf"] >= max(d["conf"] for d in dets) - 0.05
        score = top["conf"] * 1.5 + top["area"] * 2.0 + (0.5 if is_dominant else 0)
        rows.append(dict(
            path=p, score=score, top=top, dets=dets,
            sharp=sharpness(bgr), is_dominant=is_dominant,
        ))
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def print_table(rows: List[Dict], k: int = 10, label: str = ""):
    print(f"\nTop {min(k, len(rows))} candidates for {label}:")
    print(f"  {'score':<6} {'conf':<5} {'area':<5} {'dom':<3} {'classes':<35} file")
    for r in rows[:k]:
        cls = ",".join(sorted({d["name"] for d in r["dets"]}))
        print(f"  {r['score']:<6.2f} {r['top']['conf']:<5.2f} "
              f"{r['top']['area']:<5.2f} {'Y' if r['is_dominant'] else 'n':<3} "
              f"{cls[:35]:<35} {r['path'].name}")


# ---------- Grad-CAM panel rendering -----------------------------------------

def make_panel(yolo: YOLO, target_layer: nn.Module,
               img_path: Path, target_set: set) -> Tuple[np.ndarray, dict]:
    """Return a 640x640 BGR panel (heatmap overlay + target-class bboxes)
    plus the chosen detection metadata."""
    bgr_full = cv2.imread(str(img_path))
    bgr = center_crop_resize(bgr_full, SIZE)
    rgb_float = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = to_input_tensor(bgr)

    # EigenCAM is class-agnostic (PCA over activations) — no target needed.
    wrapper = YOLOFwd(yolo.model).eval()
    cam = EigenCAM(model=wrapper, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=tensor)[0]   # (640, 640) in [0,1]
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    panel = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    # Detections (already on the 640 crop) — keep only target-class boxes.
    res = yolo.predict(bgr, conf=0.4, verbose=False, device="cpu")[0]
    chosen = None
    if res.boxes is not None:
        names = res.names
        for b in res.boxes:
            cid = int(b.cls[0])
            if cid not in target_set:
                continue
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
            conf = float(b.conf[0])
            cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{names.get(cid, cid)} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(panel, (x1, y1 - th - 6), (x1 + tw + 4, y1),
                          (0, 255, 0), -1)
            cv2.putText(panel, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            if chosen is None or conf > chosen["conf"]:
                chosen = dict(name=names.get(cid, str(cid)), conf=conf,
                              bbox=(x1, y1, x2, y2))

    return panel, chosen or {}


def add_border(panel: np.ndarray, thickness: int = 2) -> np.ndarray:
    return cv2.copyMakeBorder(panel, thickness, thickness, thickness, thickness,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def make_combined(veh: np.ndarray, obs: np.ndarray, sep: int = 14,
                  caption_h: int = 56) -> np.ndarray:
    veh_b = add_border(veh)
    obs_b = add_border(obs)
    h = veh_b.shape[0]
    sep_strip = np.full((h, sep, 3), 255, np.uint8)
    row = np.hstack([veh_b, sep_strip, obs_b])

    cap_w = row.shape[1]
    captions = np.full((caption_h, cap_w, 3), 255, np.uint8)
    panel_w = veh_b.shape[1]

    def put(text, cx):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.95, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(captions, text, (cx - tw // 2, (caption_h + th) // 2),
                    font, scale, (0, 0, 0), thick, cv2.LINE_AA)

    put("(b) Vehicle",  panel_w // 2)
    put("(c) Obstacle", panel_w + sep + panel_w // 2)
    return np.vstack([row, captions])


# ---------- main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle", type=str, default=None,
                    help="explicit path for vehicle panel (else auto-pick)")
    ap.add_argument("--obstacle", type=str, default=None,
                    help="explicit path for obstacle panel (else auto-pick)")
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    print("\n[1/4] Loading YOLOv8n (pretrained COCO weights)...")
    yolo = YOLO("yolov8n.pt")
    yolo.model.eval()
    target_layer = yolo.model.model[TARGET_LAYER_IDX]
    print(f"  target layer: model.model[{TARGET_LAYER_IDX}] "
          f"({type(target_layer).__name__})")

    # ---- pick images
    if args.vehicle:
        veh_path = Path(args.vehicle)
        print(f"\n[2/4] Vehicle: using explicit path {veh_path}")
    else:
        print("\n[2/4] Searching for best VEHICLE candidate...")
        veh_rows = find_candidates(yolo, VEHICLE_CLASSES, conf=args.conf)
        print_table(veh_rows, label="VEHICLE")
        if not veh_rows:
            print("  no vehicle candidates found, lowering thresholds")
            veh_rows = find_candidates(yolo, VEHICLE_CLASSES, conf=0.3,
                                       min_sharp=200.0, min_area_ratio=0.01)
            print_table(veh_rows, label="VEHICLE (relaxed)")
        if not veh_rows:
            sys.exit("FAIL: no vehicle detection in dataset")
        veh_path = veh_rows[0]["path"]

    if args.obstacle:
        obs_path = Path(args.obstacle)
        print(f"\n[3/4] Obstacle: using explicit path {obs_path}")
    else:
        print("\n[3/4] Searching for best OBSTACLE candidate...")
        obs_rows = find_candidates(yolo, OBSTACLE_CLASSES, conf=args.conf)
        print_table(obs_rows, label="OBSTACLE")
        if not obs_rows:
            print("  no obstacle candidates found, lowering thresholds")
            obs_rows = find_candidates(yolo, OBSTACLE_CLASSES, conf=0.3,
                                       min_sharp=200.0, min_area_ratio=0.01)
            print_table(obs_rows, label="OBSTACLE (relaxed)")
        if not obs_rows:
            sys.exit("FAIL: no obstacle detection in dataset")
        # avoid picking the same frame for both panels
        obs_path = next((r["path"] for r in obs_rows if r["path"] != veh_path),
                       obs_rows[0]["path"])

    # ---- render panels
    print(f"\n[4/4] Rendering panels...")
    print(f"  vehicle  ← {veh_path.name}")
    veh_panel, veh_chosen = make_panel(yolo, target_layer, veh_path, VEHICLE_CLASSES)
    veh_out = OUT_DIR / "vehicle.jpg"
    cv2.imwrite(str(veh_out), veh_panel)
    print(f"     -> {veh_out}  detection: {veh_chosen}")

    print(f"  obstacle ← {obs_path.name}")
    obs_panel, obs_chosen = make_panel(yolo, target_layer, obs_path, OBSTACLE_CLASSES)
    obs_out = OUT_DIR / "obstacle.jpg"
    cv2.imwrite(str(obs_out), obs_panel)
    print(f"     -> {obs_out}  detection: {obs_chosen}")

    combined = make_combined(veh_panel, obs_panel)
    comb_out = OUT_DIR / "v_o_combined.jpg"
    cv2.imwrite(str(comb_out), combined)
    print(f"\nCombined figure: {comb_out}")
    print(f"Final size: {combined.shape[1]}x{combined.shape[0]} px")


if __name__ == "__main__":
    main()
