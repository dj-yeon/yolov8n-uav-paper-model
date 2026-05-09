#!/usr/bin/env python3
"""Patch a local Ultralytics source tree for the SOD-YOLOv8 modules.

Run this after copying this repository's cfg/ and nn/ folders into a cloned
Ultralytics repository:

    python scripts/patch_ultralytics_for_sod.py ~/ultralytics_sod
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


EMA_CLASS = r'''

class EMA(nn.Module):
    """Efficient Multi-Scale Attention block."""

    def __init__(self, channels, factor=8):
        super().__init__()
        self.groups = min(factor, channels)
        while channels % self.groups != 0 and self.groups > 1:
            self.groups -= 1
        group_channels = channels // self.groups
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_channels, group_channels)
        self.conv1x1 = nn.Conv2d(group_channels, group_channels, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, 3, 1, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(
            b * self.groups, 1, h, w
        )
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
'''


MMCV_FALLBACK = r'''try:
    from mmcv.cnn import ConvModule
    from mmengine.model import caffe2_xavier_init, constant_init
except Exception:
    class ConvModule(nn.Sequential):
        def __init__(self, in_channels, out_channels, kernel_size=1, act_cfg=None, **kwargs):
            padding = kwargs.pop("padding", 0)
            stride = kwargs.pop("stride", 1)
            bias = kwargs.pop("bias", True)
            super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
            self.conv = self[0]

    def caffe2_xavier_init(module):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def constant_init(module, val):
        nn.init.constant_(module.weight, val)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
'''


def resolve_ultralytics_root(raw: str) -> Path:
    root = Path(raw).expanduser().resolve()
    if (root / "ultralytics" / "nn" / "tasks.py").exists():
        return root
    if (root / "nn" / "tasks.py").exists() and root.name == "ultralytics":
        return root.parent
    raise SystemExit(f"Could not find an Ultralytics source tree at {root}")


def patch_block(block_py: Path) -> None:
    modules_dir = block_py.parent
    bra = modules_dir / "bra_legacy.py"
    if not bra.exists():
        bra.write_text('"""Compatibility stub for SOD-YOLOv8 legacy BRA import."""\n\n__all__ = ()\n', encoding="utf-8")

    text = block_py.read_text(encoding="utf-8")

    old_mmcv = "from mmcv.cnn import ConvModule\nfrom mmengine.model import caffe2_xavier_init, constant_init"
    if old_mmcv in text and "class ConvModule(nn.Sequential):" not in text:
        text = text.replace(old_mmcv, MMCV_FALLBACK)

    if "class EMA(nn.Module):" not in text:
        marker = "\nclass C2f_EMA(nn.Module):"
        if marker not in text:
            raise SystemExit(f"Could not find C2f_EMA insertion point in {block_py}")
        text = text.replace(marker, EMA_CLASS + marker)

    block_py.write_text(text, encoding="utf-8")


def add_symbol_after_lines(text: str, anchor: str, symbol: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    changed = False
    for i, line in enumerate(lines):
        output.append(line)
        if not re.match(rf"^(\s*){re.escape(anchor)},\s*$", line):
            continue
        window = "\n".join(lines[max(0, i - 3) : i + 5])
        if re.search(rf"^\s*{re.escape(symbol)},\s*$", window, flags=re.MULTILINE):
            continue
        indent = re.match(r"^(\s*)", line).group(1)
        output.append(f"{indent}{symbol},")
        changed = True
    return "\n".join(output) + ("\n" if text.endswith("\n") else ""), changed


def patch_modules_init(init_py: Path) -> None:
    text = init_py.read_text(encoding="utf-8")
    text, _ = add_symbol_after_lines(text, "C2f", "C2f_EMA")
    if '"C2f_EMA",' not in text and "'C2f_EMA'," not in text:
        text = re.sub(r'(\s*["\']C2f["\'],)', r'\1\n    "C2f_EMA",', text, count=1)
    init_py.write_text(text, encoding="utf-8")


def patch_tasks(tasks_py: Path) -> None:
    text = tasks_py.read_text(encoding="utf-8")
    text, _ = add_symbol_after_lines(text, "C2f", "C2f_EMA")

    # Newer Ultralytics versions keep C2f in base_modules/repeat_modules sets.
    # Older versions use tuples. Adding the symbol after every standalone C2f
    # line makes C2f_EMA receive the same channel and repeat handling.
    #
    # PyTorch 2.6+ changed torch.load's default weights_only value to True.
    # Ultralytics 8.2.0 expects full checkpoint loading for trusted .pt files.
    text = re.sub(
        r"torch\.load\(([^,\n]+),\s*map_location=(['\"]cpu['\"])\)",
        r"torch.load(\1, map_location=\2, weights_only=False)",
        text,
    )
    tasks_py.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ultralytics_root", help="Path such as ~/ultralytics_sod")
    args = parser.parse_args()

    root = resolve_ultralytics_root(args.ultralytics_root)
    patch_block(root / "ultralytics" / "nn" / "modules" / "block.py")
    patch_modules_init(root / "ultralytics" / "nn" / "modules" / "__init__.py")
    patch_tasks(root / "ultralytics" / "nn" / "tasks.py")

    print(f"Patched {root}")
    print("Now test with:")
    print("  python - <<'PY'")
    print("  from ultralytics import YOLO")
    print("  YOLO('ultralytics/cfg/model/yolov8s-GFPN_ema.yaml')")
    print("  print('model loaded')")
    print("  PY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
