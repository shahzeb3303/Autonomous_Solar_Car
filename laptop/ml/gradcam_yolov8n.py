#!/usr/bin/env python3
"""
Generate Grad-CAM visualizations for the pretrained YOLOv8n detector
used by the perception stack. Produces a publication-quality figure
suitable for replacing the placeholder Figure 3 in the paper.

Usage:
    python -m ml.gradcam_yolov8n                       # auto-pick 3 images
    python -m ml.gradcam_yolov8n img1.jpg img2.jpg ... # explicit images

Outputs:
    <repo>/Autonomous_Solar_Vehicle_LaTeX/images/gradcam_yolov8n.png
    <repo>/Autonomous_Solar_Vehicle_LaTeX/images/gradcam_yolov8n_<idx>.png
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO / "data" / "images"
PAPER_IMAGES = (REPO / "../../Autonomous_Solar_Vehicle_LaTeX/images").resolve()

# COCO class ids the perception stack treats as relevant obstacles
# (mirrors vision/object_detector.py:IMPORTANT_CLASSES)
IMPORTANT_CLASSES = {0, 1, 2, 3, 5, 7, 15, 16, 56, 57, 59, 60}

CONF_THRESH = 0.25
INPUT_SIZE = 640
TARGET_LAYER_IDX = 21  # last C2f before Detect head — strongest semantic features


def letterbox(img: np.ndarray, new_size: int = INPUT_SIZE) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize + pad to square, mirroring ultralytics' inference preprocessing."""
    h, w = img.shape[:2]
    r = new_size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    top, left = (new_size - nh) // 2, (new_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, (top, left)


def preprocess(img_bgr: np.ndarray, device: str) -> Tuple[torch.Tensor, np.ndarray, float, Tuple[int, int]]:
    canvas, ratio, pad = letterbox(img_bgr, INPUT_SIZE)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    return tensor, canvas, ratio, pad


def compute_gradcam(
    yolo: YOLO,
    img_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
) -> Tuple[np.ndarray, int]:
    """Run forward + backward, return a (H, W) CAM in [0, 1] and number of anchors used."""
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def fwd_hook(_m, _inp, out):
        activations.append(out)

    def bwd_hook(_m, _grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    # ultralytics' predict() can leave the model in inference_mode; force grads on.
    for p in yolo.model.parameters():
        p.requires_grad_(True)

    try:
        yolo.model.zero_grad(set_to_none=True)
        # Inner module forward returns (preds, intermediates) in eval mode.
        # preds: (1, 84, 8400) = 4 box coords + 80 class scores (sigmoid'd)
        with torch.enable_grad():
            img_tensor = img_tensor.detach().clone().requires_grad_(True)
            preds, _ = yolo.model(img_tensor)
            cls_scores = preds[0, 4:, :]                      # (80, 8400)
            max_scores, class_ids = cls_scores.max(dim=0)     # (8400,), (8400,)

            mask = max_scores > CONF_THRESH
            if mask.any():
                important_mask = torch.tensor(
                    [cid.item() in IMPORTANT_CLASSES for cid in class_ids],
                    device=mask.device,
                )
                mask = mask & important_mask

            if mask.sum() == 0:
                # Fallback: top-K anchors regardless of class — gives a CAM even on
                # frames with no clear obstacle (still informative for the figure).
                topk = max_scores.topk(10).indices
                target_score = max_scores[topk].sum()
                n_used = 10
            else:
                target_score = max_scores[mask].sum()
                n_used = int(mask.sum().item())

            target_score.backward()
    finally:
        h1.remove()
        h2.remove()

    act = activations[0].detach()        # (1, C, h, w)
    grad = gradients[0].detach()         # (1, C, h, w)
    weights = grad.mean(dim=(2, 3), keepdim=True)   # global-avg-pool over spatial
    cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, h, w)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(INPUT_SIZE, INPUT_SIZE),
                        mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    # Normalize to [0, 1]
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)
    return cam, n_used


def overlay_cam(img_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)
    return blended


def annotate_detections(img_bgr: np.ndarray, dets) -> np.ndarray:
    out = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        color = (0, 0, 255) if d.cls_id == 0 else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.cls_name} {d.conf:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def run_inference(yolo: YOLO, img_bgr: np.ndarray):
    """Standard YOLO predict for the boxes-overlay panel (for context)."""
    res = yolo.predict(img_bgr, conf=CONF_THRESH, verbose=False, device="cpu")[0]
    out = []
    if res.boxes is not None:
        names = res.names
        for b in res.boxes:
            cls_id = int(b.cls[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            out.append(type("D", (), dict(
                cls_id=cls_id,
                cls_name=names.get(cls_id, str(cls_id)),
                conf=float(b.conf[0]),
                bbox=(x1, y1, x2, y2),
            )))
    return out


def sharpness(img_bgr: np.ndarray) -> float:
    """Variance of the Laplacian — standard blur metric. Higher = sharper."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def color_hist(img_bgr: np.ndarray) -> np.ndarray:
    """Normalized 3-channel color histogram for similarity comparison."""
    h = cv2.calcHist([img_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    cv2.normalize(h, h)
    return h.flatten()


def auto_pick_images(yolo: YOLO, n: int = 3, sample: int = 2500, seed: int = 0,
                     min_sharpness: float = 350.0,
                     min_person_height_ratio: float = 0.45,
                     max_person_height_ratio: float = 0.92,
                     min_conf: float = 0.6,
                     edge_margin: float = 0.02) -> List[Path]:
    """Pick sharp frames showing a full-body person plus at least one other obstacle.

    Constraints:
      * sharpness >= ``min_sharpness`` (Laplacian variance)
      * frame contains a person (class 0) with conf >= ``min_conf``,
        bbox height between [min, max]_person_height_ratio of frame height,
        and not cropped at top/bottom (small ``edge_margin``)
      * frame contains at least one OTHER important-class detection (obstacle)
    """
    random.seed(seed)
    all_imgs = sorted(IMAGES_DIR.glob("*.jpg"))
    if len(all_imgs) > sample:
        all_imgs = random.sample(all_imgs, sample)

    scored = []
    rej_blur = rej_nodet = rej_no_person = rej_cut = rej_no_obstacle = 0
    for p in all_imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        sharp = sharpness(img)
        if sharp < min_sharpness:
            rej_blur += 1
            continue
        dets = run_inference(yolo, img)
        relevant = [d for d in dets if d.cls_id in IMPORTANT_CLASSES]
        if not relevant:
            rej_nodet += 1
            continue

        H, W = img.shape[:2]
        persons = [d for d in relevant if d.cls_id == 0 and d.conf >= min_conf]
        if not persons:
            rej_no_person += 1
            continue

        full_persons = []
        for d in persons:
            x1, y1, x2, y2 = d.bbox
            h_ratio = (y2 - y1) / H
            top_margin = y1 / H
            bot_margin = (H - y2) / H
            # Standing person: tall bbox not cropped at top or bottom of frame
            if (min_person_height_ratio <= h_ratio <= max_person_height_ratio
                    and top_margin > edge_margin
                    and bot_margin > edge_margin
                    and (y2 - y1) > 1.4 * (x2 - x1)):  # tall not wide
                full_persons.append((d, h_ratio))
        if not full_persons:
            rej_cut += 1
            continue

        # Need at least one OTHER obstacle (cone/vehicle/etc.) in the scene.
        # COCO 'cone' isn't in IMPORTANT_CLASSES so we relax to any non-person det.
        non_person_dets = [d for d in dets if d.cls_id != 0 and d.conf >= 0.3]
        if not non_person_dets:
            rej_no_obstacle += 1
            continue

        best_person, h_ratio = max(full_persons, key=lambda t: t[1])
        person_conf = best_person.conf
        unique_classes = len({d.cls_id for d in dets})  # all classes for variety
        score = (h_ratio * 3.0
                 + (sharp / 2000.0)
                 + unique_classes * 0.3
                 + person_conf * 0.4
                 + min(len(non_person_dets), 3) * 0.2)
        scored.append((score, sharp, h_ratio, person_conf, p, dets))

    scored.sort(key=lambda t: t[0], reverse=True)
    print(f"  candidates: {len(scored)} | rejected: blur={rej_blur} nodet={rej_nodet} "
          f"no_person={rej_no_person} cut_off={rej_cut} no_obstacle={rej_no_obstacle}")
    if scored[:6]:
        print("  top picks (score, sharp, height, conf, classes):")
        for s, sh, hr, cf, p, r in scored[:6]:
            cls = ",".join(sorted({d.cls_name for d in r}))
            print(f"    {s:.2f}  sharp={sh:.0f}  h={hr:.2f}  conf={cf:.2f}  "
                  f"{cls:30s}  {p.name}")

    # Greedy diversity: take top-scoring, then skip any whose color histogram is
    # too close (Bhattacharyya < 0.35) to an already-chosen pick. Prevents the
    # figure from being three near-duplicate frames of the same scene.
    chosen: List[Path] = []
    chosen_hists: List[np.ndarray] = []
    for _, _, _, _, p, _ in scored:
        img = cv2.imread(str(p))
        h = color_hist(img)
        if all(cv2.compareHist(h, ch, cv2.HISTCMP_BHATTACHARYYA) > 0.5
               for ch in chosen_hists):
            chosen.append(p)
            chosen_hists.append(h)
            if len(chosen) == n:
                break
    return chosen


def make_figure(rows: List[dict], out_path: Path, dpi: int = 200):
    """rows: list of dicts with 'orig', 'boxes', 'cam_overlay', 'caption'."""
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = ["Input", "YOLOv8n detections", "Grad-CAM (last C2f, layer 21)"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=12, fontweight="bold")

    for i, r in enumerate(rows):
        for j, key in enumerate(["orig", "boxes", "cam_overlay"]):
            img = cv2.cvtColor(r[key], cv2.COLOR_BGR2RGB)
            axes[i, j].imshow(img)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        axes[i, 0].set_ylabel(r["caption"], fontsize=10, rotation=0,
                              labelpad=70, ha="right", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="explicit image paths (else auto-pick)")
    ap.add_argument("--out-dir", default=str(PAPER_IMAGES),
                    help="where to save figures (default: paper images dir)")
    ap.add_argument("--n", type=int, default=3, help="how many images for auto-pick")
    ap.add_argument("--seed", type=int, default=0, help="random seed for auto-pick")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading YOLOv8n...")
    yolo = YOLO("yolov8n.pt")
    yolo.model.eval()
    target_layer = yolo.model.model[TARGET_LAYER_IDX]
    print(f"Target layer: model.model[{TARGET_LAYER_IDX}] ({type(target_layer).__name__})")

    if args.images:
        img_paths = [Path(p) for p in args.images]
    else:
        print(f"Auto-picking {args.n} images from {IMAGES_DIR}...")
        img_paths = auto_pick_images(yolo, n=args.n, seed=args.seed)
        if not img_paths:
            print("No images with detected obstacles found. Falling back to first frames.")
            img_paths = sorted(IMAGES_DIR.glob("*.jpg"))[: args.n]

    print(f"Using: {[p.name for p in img_paths]}")

    rows = []
    for idx, p in enumerate(img_paths):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            print(f"  skip (unreadable): {p}")
            continue

        tensor, canvas, _ratio, _pad = preprocess(img_bgr, device="cpu")
        cam, n_used = compute_gradcam(yolo, tensor, target_layer)
        cam_overlay = overlay_cam(canvas, cam)

        # Recompute detections on the letterboxed canvas so all panels share geometry
        dets = run_inference(yolo, canvas)
        boxes_img = annotate_detections(canvas, dets)

        det_summary = ", ".join(sorted({d.cls_name for d in dets if d.cls_id in IMPORTANT_CLASSES})) or "—"
        caption = f"{p.name}\n{det_summary}\n({n_used} anchors)"
        rows.append(dict(orig=canvas, boxes=boxes_img, cam_overlay=cam_overlay, caption=caption))

        # Per-image side-by-side
        side = np.hstack([canvas, boxes_img, cam_overlay])
        per_img_path = out_dir / f"gradcam_yolov8n_{idx}.png"
        cv2.imwrite(str(per_img_path), side)
        print(f"  saved {per_img_path.name}  (anchors used: {n_used}, dets: {det_summary})")

    if rows:
        combined_path = out_dir / "gradcam_yolov8n.png"
        make_figure(rows, combined_path)
        print(f"\nSaved combined figure: {combined_path}")
        print(f"Per-image figures:    {out_dir}/gradcam_yolov8n_*.png")
    else:
        print("No figures produced.")
        sys.exit(1)


if __name__ == "__main__":
    main()
