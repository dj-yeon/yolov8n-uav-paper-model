#!/usr/bin/env python3
"""Run YOLOv8 ablation experiments with one command.

This runner avoids editing the installed Ultralytics package. For EMA variants,
it temporarily replaces Ultralytics' C2f module with C2f_EMA while building the
model, then restores the original module after the experiment.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Experiment:
    key: str
    run_name: str
    model: str
    use_ema_c2f: bool
    description: str


EXPERIMENTS: dict[str, Experiment] = {
    "yolov8s": Experiment(
        key="yolov8s",
        run_name="01_yolov8s_original",
        model="yolov8s.yaml",
        use_ema_c2f=False,
        description="Original YOLOv8s P3-P5 detector.",
    ),
    "ema": Experiment(
        key="ema",
        run_name="02_yolov8s_ema",
        model="{generated}/yolov8s-ema.yaml",
        use_ema_c2f=True,
        description="Original YOLOv8s topology with C2f replaced by C2f_EMA.",
    ),
    "p2": Experiment(
        key="p2",
        run_name="03_yolov8s_p2",
        model="{generated}/yolov8s-p2.yaml",
        use_ema_c2f=False,
        description="YOLOv8s with P2-P5 detection heads for small objects.",
    ),
    "gfpn": Experiment(
        key="gfpn",
        run_name="04_yolov8s_gfpn",
        model="{generated}/yolov8s-gfpn.yaml",
        use_ema_c2f=False,
        description="Proposed GFPN/P2-P5 head without EMA C2f blocks.",
    ),
    "gfpn_ema": Experiment(
        key="gfpn_ema",
        run_name="05_yolov8s_gfpn_ema",
        model="{generated}/yolov8s-gfpn-ema.yaml",
        use_ema_c2f=True,
        description="Full proposed GFPN/P2-P5 head with C2f_EMA blocks.",
    ),
}


def str_to_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "yes", "y"}:
        return True
    if lowered in {"false", "no", "n"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_extra(pairs: list[str]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Extra argument must be key=value, got: {pair}")
        key, value = pair.split("=", 1)
        if not key:
            raise ValueError(f"Extra argument has empty key: {pair}")
        extra[key.replace("-", "_")] = str_to_value(value)
    return extra


def parse_batch(value: str) -> int | float | str:
    parsed = str_to_value(value)
    return parsed


def copy_text(src: Path, dst: Path, replacements: dict[str, str] | None = None) -> None:
    text = src.read_text(encoding="utf-8")
    for old, new in (replacements or {}).items():
        text = text.replace(old, new)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


def ensure_generated_configs(project: Path) -> Path:
    """Create fixed ablation YAMLs under the run project directory."""
    generated = project / "_generated_configs"
    generated.mkdir(parents=True, exist_ok=True)

    copy_text(ROOT / "cfg" / "model" / "yolov8.yaml", generated / "yolov8s-ema.yaml")
    copy_text(ROOT / "cfg" / "model" / "yolov8-p2.yaml", generated / "yolov8s-p2.yaml")
    copy_text(
        ROOT / "cfg" / "model" / "yolov8s-GFPN_ema.yaml",
        generated / "yolov8s-gfpn.yaml",
        replacements={"C2f_EMA": "C2f"},
    )
    copy_text(
        ROOT / "cfg" / "model" / "yolov8s-GFPN_ema.yaml",
        generated / "yolov8s-gfpn-ema.yaml",
        replacements={"C2f_EMA": "C2f"},
    )
    return generated


def make_ema_classes():
    import torch
    import torch.nn as nn
    from ultralytics.nn.modules import Bottleneck, Conv

    class EMA(nn.Module):
        """Efficient Multi-Scale Attention block."""

        def __init__(self, channels: int, factor: int = 8):
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

    class C2f_EMA(nn.Module):
        """YOLOv8 C2f block with EMA applied to the second bottleneck when present."""

        def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
            super().__init__()
            self.c = int(c2 * e)
            self.cv1 = Conv(c1, 2 * self.c, 1, 1)
            self.cv2 = Conv((2 + n) * self.c, c2, 1)
            self.m = nn.ModuleList(
                Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
                for _ in range(n)
            )
            self.ema = EMA(self.c)

        def forward(self, x):
            y = list(self.cv1(x).chunk(2, 1))
            for i, m in enumerate(self.m):
                out = m(y[-1])
                y.append(self.ema(out) if i == 1 else out)
            return self.cv2(torch.cat(y, 1))

        def forward_split(self, x):
            y = list(self.cv1(x).split((self.c, self.c), 1))
            for i, m in enumerate(self.m):
                out = m(y[-1])
                y.append(self.ema(out) if i == 1 else out)
            return self.cv2(torch.cat(y, 1))

    return EMA, C2f_EMA


@contextlib.contextmanager
def patched_c2f_with_ema(enabled: bool):
    if not enabled:
        yield
        return

    import ultralytics.nn.modules as modules
    import ultralytics.nn.modules.block as block
    import ultralytics.nn.tasks as tasks

    if not hasattr(block, "C2f_EMA"):
        raise RuntimeError(
            "ultralytics.nn.modules.block.C2f_EMA was not found. "
            "Run `python scripts/patch_ultralytics_for_sod.py ~/ultralytics_sod` first."
        )

    module_objects: list[tuple[Any, str, Any]] = []
    for obj in (tasks, modules):
        if hasattr(obj, "C2f"):
            module_objects.append((obj, "C2f", getattr(obj, "C2f")))
        if hasattr(obj, "EMA"):
            module_objects.append((obj, "EMA", getattr(obj, "EMA")))
        if hasattr(obj, "C2f_EMA"):
            module_objects.append((obj, "C2f_EMA", getattr(obj, "C2f_EMA")))

    if hasattr(block, "C2f"):
        module_objects.append((block, "C2f", getattr(block, "C2f")))
    if hasattr(block, "EMA"):
        module_objects.append((block, "EMA", getattr(block, "EMA")))
    if hasattr(block, "C2f_EMA"):
        module_objects.append((block, "C2f_EMA", getattr(block, "C2f_EMA")))

    EMA = getattr(block, "EMA", None)
    C2f_EMA = block.C2f_EMA
    targets = {obj for obj, _, _ in module_objects} | {tasks, modules}
    targets.add(block)

    for obj in targets:
        if EMA is not None:
            setattr(obj, "EMA", EMA)
        setattr(obj, "C2f_EMA", C2f_EMA)
        setattr(obj, "C2f", C2f_EMA)

    try:
        yield
    finally:
        restored = {(id(obj), attr) for obj, attr, _ in module_objects}
        for obj, attr, original in module_objects:
            setattr(obj, attr, original)
        for obj in targets:
            for attr in ("EMA", "C2f_EMA"):
                if (id(obj), attr) not in restored and hasattr(obj, attr):
                    with contextlib.suppress(AttributeError):
                        delattr(obj, attr)


def read_results_csv(path: Path) -> dict[str, str]:
    csv_path = path / "results.csv"
    if not csv_path.exists():
        return {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {f"train/{k.strip()}": v.strip() for k, v in rows[-1].items()}


def last_completed_epoch(path: Path) -> int:
    csv_path = path / "results.csv"
    if not csv_path.exists():
        return 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0
    raw = rows[-1].get("epoch") or rows[-1].get("                  epoch") or "0"
    try:
        return int(float(str(raw).strip()))
    except ValueError:
        return 0


def write_summary(project: Path, rows: list[dict[str, Any]]) -> None:
    project.mkdir(parents=True, exist_ok=True)
    json_path = project / "ablation_summary.json"
    csv_path = project / "ablation_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")

    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def select_experiments(raw: str) -> list[Experiment]:
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [key for key in keys if key not in EXPERIMENTS]
    if unknown:
        known = ", ".join(EXPERIMENTS)
        raise ValueError(f"Unknown experiments: {unknown}. Known: {known}")
    return [EXPERIMENTS[key] for key in keys]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to dataset YAML.")
    parser.add_argument(
        "--experiments",
        default="yolov8s,ema,p2,gfpn,gfpn_ema",
        help="Comma-separated experiment keys.",
    )
    parser.add_argument("--project", default="runs/ablation", help="Ultralytics project directory.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", default="4")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--pretrained", default="False", help="False, True, or a weight path.")
    parser.add_argument("--cache", nargs="?", const="True", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--single-cls", action="store_true")
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--enable-wandb", action="store_true")
    parser.add_argument("--val-split", default="val", choices=["val", "test", "none"])
    parser.add_argument("--extra", nargs="*", default=[], help="Additional Ultralytics key=value args.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_path = Path(args.data).expanduser()
    if args.data == "/absolute/path/to/data.yaml" or not data_path.exists():
        raise SystemExit(
            "Dataset YAML was not found.\n"
            f"  given: {args.data}\n\n"
            "Replace it with your real dataset YAML path, for example:\n"
            "  bash scripts/run_ablation.sh /home/djyeon/datasets/visdrone/data.yaml\n\n"
            "To search for candidates on the server:\n"
            "  find ~ -name '*.yaml' -o -name '*.yml'\n"
        )
    if not args.enable_wandb:
        os.environ.setdefault("WANDB_DISABLED", "true")

    project = Path(args.project).expanduser()
    generated = ensure_generated_configs(project)
    selected = select_experiments(args.experiments)
    extra = parse_extra(args.extra)

    common_train_kwargs: dict[str, Any] = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": parse_batch(args.batch),
        "device": args.device,
        "workers": args.workers,
        "project": str(project),
        "optimizer": args.optimizer,
        "patience": args.patience,
        "seed": args.seed,
        "pretrained": str_to_value(args.pretrained),
        "amp": args.amp,
        "exist_ok": args.exist_ok,
        "cos_lr": args.cos_lr,
        "single_cls": args.single_cls,
    }
    if args.cache is not None:
        common_train_kwargs["cache"] = str_to_value(args.cache)
    common_train_kwargs.update(extra)

    print("Ablation plan")
    print(f"  data: {args.data}")
    print(f"  project: {project}")
    print(f"  generated configs: {generated}")
    for exp in selected:
        print(f"  - {exp.run_name}: {exp.description}")

    if args.dry_run:
        return 0

    from ultralytics import YOLO

    summaries: list[dict[str, Any]] = []
    for exp in selected:
        model_ref = exp.model.format(generated=generated)
        save_dir = project / exp.run_name
        best_pt = save_dir / "weights" / "best.pt"
        row: dict[str, Any] = {
            "experiment": exp.key,
            "run_name": exp.run_name,
            "description": exp.description,
            "model": model_ref,
            "save_dir": str(save_dir),
            "best_pt": str(best_pt),
            "status": "pending",
        }

        if args.skip_existing and best_pt.exists():
            done_epochs = last_completed_epoch(save_dir)
            if done_epochs >= args.epochs:
                print(f"\n[{exp.run_name}] skipping completed run: {best_pt}")
                row["status"] = "skipped_existing"
                row.update(read_results_csv(save_dir))
                summaries.append(row)
                write_summary(project, summaries)
                continue
            print(
                f"\n[{exp.run_name}] existing run is incomplete "
                f"({done_epochs}/{args.epochs} epochs); rerunning."
            )

        print(f"\n[{exp.run_name}] training {model_ref}")
        started = time.time()
        try:
            with patched_c2f_with_ema(exp.use_ema_c2f):
                model = YOLO(model_ref)
                train_kwargs = dict(common_train_kwargs)
                train_kwargs["name"] = exp.run_name
                model.train(**train_kwargs)

            row["status"] = "trained"
            row["seconds"] = round(time.time() - started, 3)
            row.update(read_results_csv(save_dir))

            if args.val_split != "none" and best_pt.exists():
                print(f"[{exp.run_name}] validating {best_pt} on split={args.val_split}")
                val_model = YOLO(str(best_pt))
                metrics = val_model.val(
                    data=args.data,
                    imgsz=args.imgsz,
                    batch=parse_batch(args.batch),
                    device=args.device,
                    workers=args.workers,
                    split=args.val_split,
                    project=str(project),
                    name=f"{exp.run_name}_{args.val_split}",
                    exist_ok=args.exist_ok,
                )
                for key, value in getattr(metrics, "results_dict", {}).items():
                    row[f"{args.val_split}/{key}"] = value

        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)
            summaries.append(row)
            write_summary(project, summaries)
            if not args.continue_on_error:
                raise
            print(f"[{exp.run_name}] failed: {exc}", file=sys.stderr)
        else:
            summaries.append(row)
            write_summary(project, summaries)
        finally:
            gc.collect()
            with contextlib.suppress(Exception):
                import torch

                torch.cuda.empty_cache()

    write_summary(project, summaries)
    print(f"\nSummary written to {project / 'ablation_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
