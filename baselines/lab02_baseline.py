"""
Baseline for ЛР-2: CNN comparison (ResNet, SE block, ECA block).

This is a structural demonstration showing the expected pipeline,
NOT a full solution. Students must implement their own version.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block — students should implement this."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * scale


class ECABlock(nn.Module):
    """Efficient Channel Attention — students should implement this."""
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


def prepare_model(arch: str = "resnet50", num_classes: int = 10) -> nn.Module:
    """Load a pretrained model and replace the classification head."""
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


def default_transform(train: bool = True):
    """Standard transforms — students should customise."""
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


if __name__ == "__main__":
    print("Baseline for ЛР-2 (structural demo only)")
    print("This script shows helper classes. Train on real data to evaluate.")
