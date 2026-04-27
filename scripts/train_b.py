import os
import random
import numpy as np
import torch

from ultralytics import YOLO


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 재현성 우선
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

    MODEL_YAML = "configs/yolov8n_b_condconv_frelu.yaml"
    DATA_YAML = "configs/visdrone.yaml"

    PROJECT_DIR = "/content/drive/MyDrive/yolo_uav_experiments/runs"
    RUN_NAME = "B_condconv_frelu"

    model = YOLO(MODEL_YAML)

    results = model.train(
        data=DATA_YAML,
        epochs=150,
        imgsz=640,
        batch=16,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,          # 최종 LR = lr0 * lrf = 0.0001
        momentum=0.937,
        weight_decay=0.0005,
        mosaic=1.0,
        pretrained=False,
        device=0,
        workers=2,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        seed=42,
    )

    print("========== TRAIN DONE ==========")
    print(results)


if __name__ == "__main__":
    main()