#!/usr/bin/env python3
"""
Evaluate a trained model on a held-out set.

Reports: accuracy, per-class precision / recall / F1, confusion matrix,
inference latency (ms/frame). Saves a confusion_matrix.png.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import DrivingDataset
from .model import DecisionModel
from .actions import ACTION_NAMES, NUM_ACTIONS


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_metrics(cm: np.ndarray):
    k = cm.shape[0]
    precision = np.zeros(k); recall = np.zeros(k); f1 = np.zeros(k); support = np.zeros(k, dtype=np.int64)
    for i in range(k):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support[i] = cm[i, :].sum()
        precision[i] = tp / (tp + fp) if (tp + fp) else 0.0
        recall[i]    = tp / (tp + fn) if (tp + fn) else 0.0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) else 0.0
    return precision, recall, f1, support


def plot_cm(cm: np.ndarray, names, out_path: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right'); ax.set_yticklabels(names)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Confusion Matrix')
    thresh = cm.max() / 2.0 if cm.max() else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved: {out_path}")


@torch.no_grad()
def evaluate(ckpt_path: str, csv_path: str, device: str = None, batch_size: int = 64):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    ds = DrivingDataset(csv_path, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = DecisionModel(pretrained_backbone=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    all_y, all_p = [], []
    lat_ms = []
    for batch in loader:
        img = batch['image'].to(device); state = batch['state'].to(device)
        y = batch['label']

        t0 = time.time()
        logits = model(img, state)
        t1 = time.time()
        # ms / sample
        lat_ms.append((t1 - t0) * 1000.0 / y.size(0))

        p = logits.argmax(dim=1).cpu().numpy()
        all_y.append(y.numpy()); all_p.append(p)

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)

    acc = (y_true == y_pred).mean()
    cm = confusion_matrix(y_true, y_pred, NUM_ACTIONS)
    precision, recall, f1, support = per_class_metrics(cm)
    avg_latency = float(np.mean(lat_ms))

    print(f"\n=== Evaluation on {csv_path} ===")
    print(f"Samples:            {len(y_true)}")
    print(f"Overall accuracy:   {acc*100:.2f}%")
    print(f"Mean latency:       {avg_latency:.2f} ms/frame")
    print(f"\nPer-class:")
    print(f"{'class':<15} {'prec':>6} {'recall':>7} {'f1':>6} {'support':>8}")
    for i, name in enumerate(ACTION_NAMES):
        print(f"{name:<15} {precision[i]:>6.3f} {recall[i]:>7.3f} {f1[i]:>6.3f} {support[i]:>8d}")

    # Save confusion matrix plot
    out_dir = os.path.dirname(ckpt_path) or '.'
    plot_cm(cm, ACTION_NAMES, os.path.join(out_dir, 'confusion_matrix.png'))

    # Save metrics JSON
    metrics = {
        'accuracy': float(acc),
        'mean_latency_ms': avg_latency,
        'confusion_matrix': cm.tolist(),
        'per_class': {
            name: {'precision': float(precision[i]),
                   'recall': float(recall[i]),
                   'f1': float(f1[i]),
                   'support': int(support[i])}
            for i, name in enumerate(ACTION_NAMES)
        },
    }
    out_json = os.path.join(out_dir, 'eval_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {out_json}")

    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--csv', required=True)
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()
    evaluate(args.ckpt, args.csv, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
