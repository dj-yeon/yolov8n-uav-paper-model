import torch
import torch.nn as nn
import torch.nn.functional as F


class FReLU(nn.Module):
    """
    FReLU
    y = max(x, T(x))
    """

    def __init__(self, c1):
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c1,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=c1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))


class CondConv2d(nn.Module):
    """
    Conditionally Parameterized Convolution
    """

    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        d=1,
        bias=False,
        num_experts=4,
    ):
        super().__init__()

        if p is None:
            p = k // 2 if isinstance(k, int) else 0

        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = p
        self.g = g
        self.d = d
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.randn(
                num_experts,
                c2,
                c1 // g,
                k,
                k,
            )
        )

        self.bias = nn.Parameter(torch.zeros(num_experts, c2)) if bias else None

        self.routing = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, num_experts),
            nn.Sigmoid(),
        )

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        batch_size = x.shape[0]
        routing_weights = self.routing(x)

        outputs = []

        for i in range(batch_size):
            mixed_weight = torch.sum(
                routing_weights[i].view(self.num_experts, 1, 1, 1, 1) * self.weight,
                dim=0,
            )

            if self.bias is not None:
                mixed_bias = torch.sum(
                    routing_weights[i].view(self.num_experts, 1) * self.bias,
                    dim=0,
                )
            else:
                mixed_bias = None

            out = F.conv2d(
                x[i].unsqueeze(0),
                mixed_weight,
                mixed_bias,
                stride=self.s,
                padding=self.p,
                dilation=self.d,
                groups=self.g,
            )

            outputs.append(out)

        return torch.cat(outputs, dim=0)


class CondConv(nn.Module):
    """
    YOLOv8 Conv replacement

    Original:
    Conv2d -> BatchNorm2d -> SiLU

    B experiment:
    CondConv2d -> BatchNorm2d -> FReLU
    """

    default_act = FReLU

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, num_experts=4):
        super().__init__()

        self.conv = CondConv2d(
            c1=c1,
            c2=c2,
            k=k,
            s=s,
            p=p,
            g=g,
            d=d,
            bias=False,
            num_experts=num_experts,
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = FReLU(c2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))