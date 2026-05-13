#!/usr/bin/env python3
"""
Training pipeline for the autonomous driving decision model.

Proper ML practice:
    - Train / Val / Test split (70 / 15 / 15)
    - AdamW optimizer, CosineAnnealing LR schedule
    - CrossEntropyLoss with class weights (imbalanced data)
    - Early stopping on val loss
    - Model checkpointing (best + last)
    - TensorBoard logging
    - Two-stage training: frozen backbone -> unfrozen fine-tune

Usage:
    python -m laptop.ml.train --csv laptop/data/dataset.csv --out laptop/models
"""

import argparse
import os
import time
from dataclasses import dataclass, asdict
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import DrivingDataset, class_weights, split_csv
from .model import DecisionModel
from .actions import NUM_ACTIONS, ACTION_NAMES


@dataclass
class TrainConfig:
    csv_path: str
    out_dir: str
    batch_size: int = 32
    num_workers: int = 2
    epochs_frozen: int = 10
    epochs_unfrozen: int = 20
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 6
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            img = batch['image'].to(device, non_blocking=True)
            state = batch['state'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)

            logits = model(img, state)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            bs = y.size(0)
            loss_sum += loss.item() * bs
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += bs
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    log_dir = os.path.join(cfg.out_dir, 'tb_' + time.strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    print(f"Device: {cfg.device}")
    print(f"TensorBoard logs: {log_dir}")

    # 1) Split
    splits_dir = os.path.join(cfg.out_dir, 'splits')
    paths = split_csv(cfg.csv_path, splits_dir, seed=cfg.seed)

    train_ds = DrivingDataset(paths['train'], train=True)
    val_ds   = DrivingDataset(paths['val'],   train=False)
    test_ds  = DrivingDataset(paths['test'],  train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    # 2) Class-weighted loss
    labels = [s.label for s in train_ds.samples]
    cw = class_weights(labels, NUM_ACTIONS).to(cfg.device)
    print("Class weights:")
    for i, n in enumerate(ACTION_NAMES):
        print(f"  {n:15s}: {cw[i].item():.3f}")
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

    # 3) Model
    model = DecisionModel(pretrained_backbone=True, freeze_backbone=True).to(cfg.device)

    # 4) Phase 1: frozen backbone (head-only)
    print(f"\n=== Phase 1: train head only ({cfg.epochs_frozen} epochs) ===")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs_frozen)

    best_val = float('inf')
    patience = 0
    global_step = 0
    best_path = os.path.join(cfg.out_dir, 'best.pt')

    def _step(epoch, phase):
        nonlocal best_val, patience, global_step
        tl, ta = run_epoch(model, train_loader, criterion, opt, cfg.device, train=True)
        vl, va = run_epoch(model, val_loader,   criterion, opt, cfg.device, train=False)
        sched.step()

        writer.add_scalar(f'{phase}/train_loss', tl, epoch)
        writer.add_scalar(f'{phase}/train_acc',  ta, epoch)
        writer.add_scalar(f'{phase}/val_loss',   vl, epoch)
        writer.add_scalar(f'{phase}/val_acc',    va, epoch)
        writer.add_scalar(f'{phase}/lr', opt.param_groups[0]['lr'], epoch)

        print(f"[{phase}] epoch {epoch:3d} | "
              f"train_loss={tl:.4f} acc={ta*100:.2f}% | "
              f"val_loss={vl:.4f} acc={va*100:.2f}% | lr={opt.param_groups[0]['lr']:.2e}")

        improved = vl < best_val - 1e-4
        if improved:
            best_val = vl
            patience = 0
            torch.save({
                'model': model.state_dict(),
                'phase': phase,
                'epoch': epoch,
                'val_loss': vl,
                'val_acc': va,
            }, best_path)
            print(f"  * saved best to {best_path}")
        else:
            patience += 1
        return improved

    for e in range(1, cfg.epochs_frozen + 1):
        _step(e, 'phase1')
        if patience >= cfg.early_stop_patience:
            print("Early stop in phase 1"); break

    # 5) Phase 2: unfreeze backbone, low LR
    print(f"\n=== Phase 2: fine-tune backbone ({cfg.epochs_unfrozen} epochs) ===")
    model.unfreeze_backbone()
    opt = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': cfg.lr_backbone},
        {'params': model.head.parameters(),     'lr': cfg.lr_head * 0.5},
    ], weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs_unfrozen)
    patience = 0

    for e in range(1, cfg.epochs_unfrozen + 1):
        _step(e, 'phase2')
        if patience >= cfg.early_stop_patience:
            print("Early stop in phase 2"); break

    # 6) Evaluate best on test set
    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt['model'])
    tl, ta = run_epoch(model, test_loader, criterion, opt, cfg.device, train=False)
    print(f"\n=== TEST RESULTS ===")
    print(f"Test loss: {tl:.4f}  accuracy: {ta*100:.2f}%")
    writer.add_scalar('test/loss', tl, 0)
    writer.add_scalar('test/acc',  ta, 0)

    # Save metadata
    with open(os.path.join(cfg.out_dir, 'train_config.json'), 'w') as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(os.path.join(cfg.out_dir, 'test_metrics.json'), 'w') as f:
        json.dump({'test_loss': tl, 'test_acc': ta,
                   'best_val_loss': best_val,
                   'splits': paths}, f, indent=2)

    writer.close()
    print(f"\nBest model: {best_path}")
    print(f"Run `python -m laptop.ml.evaluate --ckpt {best_path} --csv {paths['test']}` for full metrics.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='Path to dataset CSV')
    p.add_argument('--out', default='laptop/models', help='Output directory')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--epochs-frozen', type=int, default=10)
    p.add_argument('--epochs-unfrozen', type=int, default=20)
    args = p.parse_args()

    cfg = TrainConfig(
        csv_path=args.csv,
        out_dir=args.out,
        batch_size=args.batch_size,
        num_workers=args.workers,
        epochs_frozen=args.epochs_frozen,
        epochs_unfrozen=args.epochs_unfrozen,
    )
    train(cfg)


if __name__ == "__main__":
    main()
