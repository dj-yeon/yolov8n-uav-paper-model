import json
import torch
from ultralytics import YOLO


def main():
    WEIGHT_PATH = "/content/drive/MyDrive/yolo_uav_experiments/runs/B_condconv_frelu/weights/best.pt"
    DATA_YAML = "configs/visdrone.yaml"

    model = YOLO(WEIGHT_PATH)

    metrics = model.val(
        data=DATA_YAML,
        imgsz=640,
        batch=16,
        device=0,
        split="val",
        project="/content/drive/MyDrive/yolo_uav_experiments/runs",
        name="B_condconv_frelu_val",
        exist_ok=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    print("\n========== B EXPERIMENT METRICS ==========")
    print(f"Precision     : {precision * 100:.1f}%")
    print(f"Recall        : {recall * 100:.1f}%")
    print(f"mAP@0.5       : {map50 * 100:.1f}%")
    print(f"mAP@0.5:0.95  : {map50_95 * 100:.1f}%")
    print("==========================================")

    result_dict = {
        "experiment": "B_condconv_frelu",
        "precision_percent": round(precision * 100, 3),
        "recall_percent": round(recall * 100, 3),
        "map50_percent": round(map50 * 100, 3),
        "map50_95_percent": round(map50_95 * 100, 3),
    }

    save_path = "/content/drive/MyDrive/yolo_uav_experiments/runs/B_condconv_frelu_val/metrics_summary.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    print(f"Saved metrics to: {save_path}")


if __name__ == "__main__":
    main()