#!/usr/bin/env python3
"""
Export a trained PyTorch checkpoint to ONNX for fast inference.

Usage:
    python -m laptop.ml.export --ckpt laptop/models/best.pt --out laptop/models/model.onnx
"""

import argparse
import torch

from .model import DecisionModel, STATE_DIM


def export(ckpt_path: str, out_path: str, opset: int = 17):
    device = 'cpu'
    model = DecisionModel(pretrained_backbone=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    dummy_img   = torch.randn(1, 3, 224, 224)
    dummy_state = torch.randn(1, STATE_DIM)

    torch.onnx.export(
        model, (dummy_img, dummy_state), out_path,
        input_names=['image', 'state'],
        output_names=['logits'],
        dynamic_axes={
            'image': {0: 'batch'},
            'state': {0: 'batch'},
            'logits': {0: 'batch'},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"Exported: {out_path}")
    print("Verify with onnxruntime:")
    print(f"  python -c \"import onnxruntime as ort; ort.InferenceSession('{out_path}')\"")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    export(args.ckpt, args.out)


if __name__ == "__main__":
    main()
