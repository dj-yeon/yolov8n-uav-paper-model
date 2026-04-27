import os
import shutil
from pathlib import Path
from PIL import Image

# =========================
# 1. 경로 설정
# =========================

# VisDrone 원본 데이터셋 위치
SOURCE_ROOT = Path("./raw_visdrone")

# 변환 결과 저장 위치
OUTPUT_ROOT = Path("./VisDrone")

DATASETS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}

# VisDrone 클래스
CLASS_NAMES = [
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


# =========================
# 2. 폴더 생성
# =========================

def make_dirs():
    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


# =========================
# 3. VisDrone bbox → YOLO bbox 변환
# =========================

def convert_bbox_to_yolo(img_w, img_h, x, y, w, h):
    x_center = x + w / 2
    y_center = y + h / 2

    x_center /= img_w
    y_center /= img_h
    w /= img_w
    h /= img_h

    return x_center, y_center, w, h


# =========================
# 4. annotation 변환
# =========================

def convert_annotations(split, source_folder):
    ann_dir = SOURCE_ROOT / source_folder / "annotations"
    img_dir = SOURCE_ROOT / source_folder / "images"

    out_label_dir = OUTPUT_ROOT / "labels" / split

    if not ann_dir.exists():
        print(f"[경고] annotation 폴더 없음: {ann_dir}")
        return

    for ann_file in ann_dir.glob("*.txt"):
        image_file = img_dir / ann_file.with_suffix(".jpg").name

        if not image_file.exists():
            print(f"[스킵] 이미지 없음: {image_file}")
            continue

        with Image.open(image_file) as img:
            img_w, img_h = img.size

        yolo_lines = []

        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")

                if len(parts) < 6:
                    continue

                x = float(parts[0])
                y = float(parts[1])
                w = float(parts[2])
                h = float(parts[3])
                category = int(parts[5])

                # VisDrone category 0은 ignored region
                if category == 0:
                    continue

                # VisDrone은 1~10, YOLO는 0~9
                class_id = category - 1

                if class_id < 0 or class_id >= len(CLASS_NAMES):
                    continue

                x_center, y_center, yolo_w, yolo_h = convert_bbox_to_yolo(
                    img_w, img_h, x, y, w, h
                )

                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}"
                )

        out_file = out_label_dir / ann_file.name

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))


# =========================
# 5. 이미지 복사
# =========================

def copy_images(split, source_folder):
    src_img_dir = SOURCE_ROOT / source_folder / "images"
    dst_img_dir = OUTPUT_ROOT / "images" / split

    if not src_img_dir.exists():
        print(f"[경고] 이미지 폴더 없음: {src_img_dir}")
        return

    for img_file in src_img_dir.glob("*.jpg"):
        shutil.copy2(img_file, dst_img_dir / img_file.name)


# =========================
# 6. dataset yaml 생성
# =========================

def create_yaml():
    yaml_path = OUTPUT_ROOT / "visdrone.yaml"

    content = f"""path: /content/drive/MyDrive/yolo_uav_experiments/datasets/VisDrone

train: images/train
val: images/val
test: images/test

names:
"""

    for i, name in enumerate(CLASS_NAMES):
        content += f"  {i}: {name}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)


# =========================
# 7. 실행
# =========================

def main():
    make_dirs()

    for split, folder in DATASETS.items():
        print(f"\n[{split}] 이미지 복사 중...")
        copy_images(split, folder)

        print(f"[{split}] annotation 변환 중...")
        convert_annotations(split, folder)

    create_yaml()

    print("\n완료!")
    print(f"결과 폴더: {OUTPUT_ROOT.resolve()}")
    print("이 VisDrone 폴더를 Google Drive의 아래 위치에 업로드하면 됨:")
    print("MyDrive/yolo_uav_experiments/datasets/VisDrone")


if __name__ == "__main__":
    main()