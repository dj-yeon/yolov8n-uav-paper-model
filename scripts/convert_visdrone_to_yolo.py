#!/usr/bin/env python3
"""Convert VisDrone2019-DET annotations to Ultralytics YOLO format."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


@dataclass(frozen=True)
class SplitSpec:
    source_name: str
    target_name: str
    has_labels: bool
    aliases: tuple[str, ...]


SPLITS = (
    SplitSpec("VisDrone2019-DET-train", "train", True, ("train", "VisDrone2019-DET-train", "VisDrone2019-DET_train")),
    SplitSpec("VisDrone2019-DET-val", "val", True, ("val", "valid", "validation", "VisDrone2019-DET-val", "VisDrone2019-DET_val")),
    SplitSpec("VisDrone2019-DET-test-dev", "test", False, ("test", "test-dev", "VisDrone2019-DET-test-dev")),
)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def link_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def convert_annotation(annotation: Path, width: int, height: int) -> tuple[list[str], dict[str, int]]:
    lines: list[str] = []
    stats = {"kept": 0, "ignored": 0, "bad": 0}
    if not annotation.exists():
        return lines, stats

    for raw in annotation.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 8:
            stats["bad"] += 1
            continue
        try:
            x, y, w, h = map(float, parts[:4])
            score = int(float(parts[4]))
            category = int(float(parts[5]))
        except ValueError:
            stats["bad"] += 1
            continue

        # VisDrone category 0 is ignored regions and 11 is others. YOLO class ids
        # are remapped from VisDrone 1-10 to 0-9.
        if score == 0 or category <= 0 or category > 10 or w <= 0 or h <= 0:
            stats["ignored"] += 1
            continue

        x1 = max(0.0, min(x, width))
        y1 = max(0.0, min(y, height))
        x2 = max(0.0, min(x + w, width))
        y2 = max(0.0, min(y + h, height))
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 1 or bh <= 1:
            stats["ignored"] += 1
            continue

        cls = category - 1
        xc = (x1 + bw / 2.0) / width
        yc = (y1 + bh / 2.0) / height
        nw = bw / width
        nh = bh / height
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        stats["kept"] += 1

    return lines, stats


def write_data_yaml(output: Path, include_test: bool) -> None:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(NAMES))
    test_line = "test: images/test\n" if include_test else "test: images/val\n"
    text = f"""path: {output.resolve()}
train: images/train
val: images/val
{test_line}
names:
{names}
"""
    (output / "data.yaml").write_text(text, encoding="utf-8")


def find_split_root(source_root: Path, split: SplitSpec, override: str | None = None) -> Path | None:
    if override:
        root = Path(override).expanduser().resolve()
        return root if root.exists() else None

    candidates = [source_root / split.source_name]
    candidates.extend(source_root / alias for alias in split.aliases)
    if source_root.name in split.aliases or source_root.name == split.source_name:
        candidates.append(source_root)

    for candidate in candidates:
        if (candidate / "images").is_dir() and ((candidate / "annotations").is_dir() or not split.has_labels):
            return candidate
        nested = candidate / split.source_name
        if (nested / "images").is_dir() and ((nested / "annotations").is_dir() or not split.has_labels):
            return nested

    lower_tokens = {split.target_name, split.source_name.lower(), *[a.lower() for a in split.aliases]}
    for candidate in source_root.rglob("*"):
        if not candidate.is_dir():
            continue
        name = candidate.name.lower()
        if not any(token in name for token in lower_tokens):
            continue
        if (candidate / "images").is_dir() and ((candidate / "annotations").is_dir() or not split.has_labels):
            return candidate
    return None


def convert_split(source_root: Path, output: Path, split: SplitSpec, image_mode: str, override: str | None = None) -> dict[str, int]:
    split_root = find_split_root(source_root, split, override)
    if split_root is None:
        if split.target_name == "test":
            return {"images": 0, "labels": 0, "kept": 0, "ignored": 0, "bad": 0}
        aliases = ", ".join((split.source_name, *split.aliases))
        raise FileNotFoundError(
            f"Could not find {split.target_name} split under {source_root}. "
            f"Expected a directory named one of [{aliases}] containing images/ and annotations/."
        )
    image_dir = split_root / "images"
    annotation_dir = split_root / "annotations"
    if split.has_labels and not annotation_dir.exists():
        raise FileNotFoundError(f"Missing annotation directory: {annotation_dir}")

    out_images = output / "images" / split.target_name
    out_labels = output / "labels" / split.target_name
    out_images.mkdir(parents=True, exist_ok=True)
    if split.has_labels:
        out_labels.mkdir(parents=True, exist_ok=True)

    totals = {"images": 0, "labels": 0, "kept": 0, "ignored": 0, "bad": 0}
    for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}):
        link_image(image, out_images / image.name, image_mode)
        totals["images"] += 1

        if not split.has_labels:
            continue

        width, height = image_size(image)
        annotation = annotation_dir / f"{image.stem}.txt"
        yolo_lines, stats = convert_annotation(annotation, width, height)
        (out_labels / f"{image.stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        totals["labels"] += 1
        for key in ("kept", "ignored", "bad"):
            totals[key] += stats[key]

    return totals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="Directory containing VisDrone2019-DET-train and VisDrone2019-DET-val.",
    )
    parser.add_argument("--output", required=True, help="Output YOLO dataset directory.")
    parser.add_argument("--train-dir", default=None, help="Optional explicit VisDrone train split directory.")
    parser.add_argument("--val-dir", default=None, help="Optional explicit VisDrone val split directory.")
    parser.add_argument("--test-dir", default=None, help="Optional explicit VisDrone test-dev split directory.")
    parser.add_argument(
        "--image-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to place images in the YOLO directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Source directory does not exist: {source}")
    output.mkdir(parents=True, exist_ok=True)

    include_test = False
    overrides = {"train": args.train_dir, "val": args.val_dir, "test": args.test_dir}
    print(f"source: {source}")
    print(f"output: {output}")
    for split in SPLITS:
        stats = convert_split(source, output, split, args.image_mode, overrides[split.target_name])
        include_test = include_test or (split.target_name == "test" and stats["images"] > 0)
        print(
            f"{split.target_name}: images={stats['images']} labels={stats['labels']} "
            f"kept={stats['kept']} ignored={stats['ignored']} bad={stats['bad']}"
        )

    write_data_yaml(output, include_test)
    print(f"wrote: {output / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
