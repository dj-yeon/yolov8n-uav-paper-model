import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def main():
    model = YOLO("configs/yolov8n_d_condconv_frelu_p2_wiseiou.yaml")

    print("========== MODEL INFO ==========")
    print(model.model)
    model.info()
    print("================================")


if __name__ == "__main__":
    main()
