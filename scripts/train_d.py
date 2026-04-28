import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_env():
    print("========== ENV INFO ==========")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    print("==============================")


def main():
    set_seed(42)
    print_env()

    model_yaml = "configs/yolov8n_d_condconv_frelu_p2_wiseiou.yaml"
    data_yaml = "configs/visdrone_colab.yaml"

    project_dir = os.getenv(
        "YOLO_UAV_PROJECT_DIR",
        "/content/drive/MyDrive/yolo_uav_experiments/runs",
    )
    run_name = "D_condconv_frelu_p2_wiseiou"

    model = YOLO(model_yaml)

    results = model.train(
        data=data_yaml,
        epochs=150,
        imgsz=640,
        batch=16,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        mosaic=1.0,
        box_loss="wise_iou",
        pretrained=False,
        device=0,
        workers=2,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        seed=42,
    )

    print("========== TRAIN DONE ==========")
    print(results)


if __name__ == "__main__":
    main()
