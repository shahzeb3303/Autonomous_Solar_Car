#!/usr/bin/env python3
"""
Decision Model Architecture (PyTorch).

Two-stream network:
    Stream A (vision): MobileNetV3-Small backbone (pretrained), frozen first,
                       fine-tuned in later epochs. Outputs 576-dim features.
    Stream B (state):  6 ultrasonic + 1 front_min + 1 back_min + GPS (speed,
                       heading) + prev_action(one-hot 8) = 19-dim sensor vec.

Fusion head:
    Concat(vision_feats, state_feats) -> MLP -> 8 action logits.

Standard supervised classification. Softmax at inference time.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

from .actions import NUM_ACTIONS

# State vector layout:
# - 6 ultrasonic (FL, FR, FW, BC, LS, RS)
# - 2 mins (front_min, back_min)
# - 3 GPS (valid, speed, heading_rad)
# - 4 YOLO features (person_detected, object_detected, nearest_area_ratio, nearest_position)
# - 8 prev_action one-hot
YOLO_FEATURES = 4
STATE_DIM = 6 + 2 + 3 + YOLO_FEATURES + NUM_ACTIONS


def make_backbone(pretrained: bool = True):
    """MobileNetV3-Small, drop the classifier head."""
    weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    net = tvm.mobilenet_v3_small(weights=weights)
    feat_dim = net.classifier[0].in_features  # 576
    # Replace classifier with identity to use as feature extractor
    net.classifier = nn.Identity()
    return net, feat_dim


class DecisionModel(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM,
                 num_actions: int = NUM_ACTIONS,
                 pretrained_backbone: bool = True,
                 freeze_backbone: bool = True):
        super().__init__()
        self.backbone, vision_feat_dim = make_backbone(pretrained_backbone)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        fused_dim = vision_feat_dim + state_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_actions),
        )
        self.state_dim = state_dim
        self.num_actions = num_actions

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, 224, 224) normalized ImageNet stats
            state: (B, state_dim) normalized sensor vector

        Returns:
            logits: (B, num_actions)
        """
        v = self.backbone(image)           # (B, 576)
        x = torch.cat([v, state], dim=1)   # (B, 576+state_dim)
        return self.head(x)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


if __name__ == "__main__":
    m = DecisionModel()
    img = torch.randn(2, 3, 224, 224)
    st = torch.randn(2, STATE_DIM)
    out = m(img, st)
    print(f"Output shape: {out.shape}")
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total params: {total:,}  Trainable: {trainable:,}")
