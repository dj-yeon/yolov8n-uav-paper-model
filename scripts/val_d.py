import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def main():
    model = YOLO("configs/yolov8n_d_condconv_frelu_p2_wiseiou.yaml")

    metrics = model.val(
        data="configs/visdrone_colab.yaml",
        imgsz=640,
        batch=16,
        device=0,
        project="/content/drive/MyDrive/yolo_uav_experiments/runs",
        name="D_condconv_frelu_p2_wiseiou_val",
        exist_ok=True,
    )

    print("========== VAL DONE ==========")
    print(metrics)


if __name__ == "__main__":
    main()
