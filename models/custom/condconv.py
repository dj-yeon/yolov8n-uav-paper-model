import torch
import torch.nn as nn
import torch.nn.functional as F


class CondConv2d(nn.Module):
    """
    Conditionally Parameterized Convolution

    여러 expert convolution weight를 두고,
    입력 x에 따라 routing weight를 계산해서
    동적으로 convolution kernel을 조합한다.

    논문 수식:
    Output(x) = σ((α1W1 + ... + αnWn) × x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=False,
        num_experts=4,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.randn(
                num_experts,
                out_channels,
                in_channels // groups,
                self.kernel_size,
                self.kernel_size,
            )
        )

        self.bias = nn.Parameter(torch.zeros(num_experts, out_channels)) if bias else None

        self.routing = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_experts),
            nn.Sigmoid(),
        )

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        batch_size, _, h, w = x.shape

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
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            outputs.append(out)

        return torch.cat(outputs, dim=0)


class CondConv(nn.Module):
    """
    YOLOv8 Conv 대체용 블록

    기존:
    Conv2d -> BatchNorm2d -> SiLU

    변경:
    CondConv2d -> BatchNorm2d -> FReLU
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
        num_experts=4,
    ):
        super().__init__()

        if p is None:
            p = k // 2 if isinstance(k, int) else 0

        from models.custom.frelu import FReLU

        self.conv = CondConv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            groups=g,
            bias=False,
            num_experts=num_experts,
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = FReLU(c2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))