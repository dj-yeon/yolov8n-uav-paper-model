import torch
import torch.nn as nn


class FReLU(nn.Module):
    """
    FReLU: Funnel ReLU

    논문에서는 기존 YOLOv8의 SiLU 대신 FReLU를 사용한다고 설명한다.
    y = max(x, T(x))
    """

    def __init__(self, channels: int):
        super().__init__()

        self.branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return torch.max(x, self.branch(x))