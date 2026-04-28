from ultralytics import YOLO


def main():
    model = YOLO("configs/yolov8n_paper.yaml")

    model.train(
        data="configs/visdrone_colab.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        mosaic=1.0,
        project="/content/drive/MyDrive/yolo_uav_experiments/runs",
        name="paper_model_structure_only",
        pretrained=False,
        device=0,
    )


if __name__ == "__main__":
    main()