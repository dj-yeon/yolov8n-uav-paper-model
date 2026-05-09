#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_ablation.sh /path/to/data.yaml [extra ultralytics args...]"
  echo "Example: EPOCHS=100 IMGSZ=1280 BATCH=4 DEVICE=0 bash scripts/run_ablation.sh /data/visdrone.yaml"
  exit 2
fi

DATA_YAML="$1"
shift

EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-1280}"
BATCH="${BATCH:-4}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
PROJECT="${PROJECT:-runs/ablation}"
EXPERIMENTS="${EXPERIMENTS:-yolov8s,ema,p2,gfpn,gfpn_ema}"
VAL_SPLIT="${VAL_SPLIT:-val}"

python scripts/run_ablation.py \
  --data "$DATA_YAML" \
  --experiments "$EXPERIMENTS" \
  --project "$PROJECT" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --val-split "$VAL_SPLIT" \
  "$@"
