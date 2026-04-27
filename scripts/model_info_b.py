from ultralytics import YOLO


def main():
    model = YOLO("configs/yolov8n_b_condconv_frelu.yaml")

    print("========== MODEL INFO ==========")
    model.info(detailed=True, verbose=True)
    print("================================")


if __name__ == "__main__":
    main()