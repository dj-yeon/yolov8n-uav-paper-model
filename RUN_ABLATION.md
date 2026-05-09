# Running the Ablation Study

This repository is not a standalone training package. Use `scripts/run_ablation.py`
to run all comparison models through Ultralytics with the same training arguments.

## 1. Server setup

```bash
cd Yolov8-Small-Object-Detection-Arial-Images
bash scripts/setup_gtx4090.sh
conda activate sod_yolo
```

If your cluster already has a PyTorch/Ultralytics environment, activate it instead.

If you copied this repository's `cfg/`, `nn/`, and `utils/` folders directly into
a cloned Ultralytics repository, patch that Ultralytics tree once:

```bash
python scripts/patch_ultralytics_for_sod.py ~/ultralytics_sod
```

## 2. Dataset YAML

Prepare an Ultralytics detection dataset YAML:

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test
names:
  0: class0
  1: class1
```

For original VisDrone2019-DET data, convert it first:

```bash
python scripts/convert_visdrone_to_yolo.py \
  --source /home/djyeon/datasets/VisDrone2019-DET \
  --output /home/djyeon/datasets/visdrone-yolo \
  --image-mode symlink
```

Then use:

```bash
/home/djyeon/datasets/visdrone-yolo/data.yaml
```

## 3. Run all experiments

```bash
EPOCHS=100 IMGSZ=1280 BATCH=4 DEVICE=0 \
bash scripts/run_ablation.sh /home/djyeon/datasets/visdrone/data.yaml
```

Default experiment order:

| Key | Run name | Meaning |
| --- | --- | --- |
| `yolov8s` | `01_yolov8s_original` | Original YOLOv8s baseline |
| `ema` | `02_yolov8s_ema` | Original YOLOv8s topology with EMA C2f |
| `p2` | `03_yolov8s_p2` | YOLOv8s with P2-P5 heads |
| `gfpn` | `04_yolov8s_gfpn` | Proposed GFPN/P2-P5 head without EMA |
| `gfpn_ema` | `05_yolov8s_gfpn_ema` | Full proposed model |

Outputs:

```text
runs/ablation/
  01_yolov8s_original/
  02_yolov8s_ema/
  03_yolov8s_p2/
  04_yolov8s_gfpn/
  05_yolov8s_gfpn_ema/
  ablation_summary.csv
  ablation_summary.json
```

## Useful variants

Run only the original YOLO and full model:

```bash
EXPERIMENTS=yolov8s,gfpn_ema bash scripts/run_ablation.sh /absolute/path/to/data.yaml
```

Resume by skipping completed runs:

```bash
bash scripts/run_ablation.sh /absolute/path/to/data.yaml --skip-existing
```

Pass extra Ultralytics arguments:

```bash
bash scripts/run_ablation.sh /absolute/path/to/data.yaml \
  --extra lr0=0.01 cos_lr=True close_mosaic=10
```

For strict architecture ablation, the default is `pretrained=False`, so every model
trains from scratch with the same seed and schedule.
