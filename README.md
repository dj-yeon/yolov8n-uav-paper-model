GitHub 저장소
yolov8-uav-paper-impl/
│
├── colab/
│ └── train_colab.ipynb
│
├── configs/
│ ├── visdrone.yaml
│ └── yolov8n_paper.yaml
│
├── models/
│ └── custom/
│ ├── condconv.py
│ └── frelu.py
│
├── scripts/
│ ├── train.py
│ ├── val.py
│ └── predict.py
│
└── README.md

구글 드라이브 구조

MyDrive/
└── yolo_uav_experiments/
├── datasets/
│ └── VisDrone/
│ ├── images/
│ │ ├── train/
│ │ ├── val/
│ │ └── test/
│ └── labels/
│ ├── train/
│ ├── val/
│ └── test/
│
├── weights/
│ ├── yolov8n.pt
│ └── paper_model/
│
└── runs/
├── A_yolov8n_baseline/
└── paper_model/
