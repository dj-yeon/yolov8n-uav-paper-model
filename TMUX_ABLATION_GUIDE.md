# YOLOv8 Ablation Study 서버 실행 가이드

논문에서 제시한 ablation study를 GPU 서버에서 한 번에 실행하기 위한 가이드입니다.

목표는 다음과 같습니다.

- YOLOv8 원본 baseline과 개선 모델들을 한 번에 실행
- SSH 터미널을 꺼도 서버에서 학습 유지
- 중간 결과와 로그 확인 가능
- 실패한 실험이 있어도 다음 실험으로 진행
- 중간에 끊겨도 완료된 실험은 건너뛰고 재실행 가능

---

## 1. tmux를 사용하는 이유

GPU 서버에서 장시간 학습을 돌릴 때는 `tmux`를 사용하는 것이 좋습니다.

`tmux`를 사용하면 내 컴퓨터의 SSH 터미널을 꺼도 서버에서는 학습이 계속 진행됩니다.

---

## 2. 서버 접속

```bash
ssh djyeon@vis-yolo
```

---

## 3. tmux 세션 생성

```bash
tmux new -s sod_ablation
```

여기서 `sod_ablation`은 tmux 세션 이름입니다.

---

## 4. 프로젝트 폴더로 이동

tmux 안에서 실행합니다.

```bash
cd ~/Yolov8-Small-Object-Detection-Arial-Images
```

---

## 5. Conda 환경 활성화

```bash
conda activate sod_yolo
```

---

## 6. Ultralytics 패치 적용

PyTorch 2.6 이상과 Ultralytics 8.2.0 호환 문제를 피하기 위해 패치 스크립트를 먼저 실행합니다.

```bash
python scripts/patch_ultralytics_for_sod.py ~/ultralytics_sod
```

이 패치는 다음 문제를 해결하기 위한 것입니다.

- PyTorch 2.6+에서 `torch.load`의 기본값이 `weights_only=True`로 변경됨
- Ultralytics 8.2.0에서 AMP 체크 중 `yolov8n.pt` 로드 실패 가능
- trusted YOLO checkpoint 로딩 시 `weights_only=False`가 필요함

---

## 7. 로그 폴더 생성

```bash
mkdir -p logs
```

---

## 8. Ablation 전체 실험 실행

```bash
EPOCHS=100 IMGSZ=1280 BATCH=4 DEVICE=0 \
bash scripts/run_ablation.sh /home/djyeon/datasets/visdrone-yolo/data.yaml \
  --skip-existing --continue-on-error \
  2>&1 | tee -a logs/ablation_$(date +%Y%m%d_%H%M%S).log
```

---

## 9. 실행되는 실험 목록

위 명령은 다음 실험들을 순서대로 실행합니다.

| 순서 | 실험 이름 | 설명 |
|---:|---|---|
| 1 | `01_yolov8s_original` | 원본 YOLOv8s baseline |
| 2 | `02_yolov8s_ema` | YOLOv8s + EMA C2f |
| 3 | `03_yolov8s_p2` | P2 detection head 추가 |
| 4 | `04_yolov8s_gfpn` | GFPN 구조, EMA 없음 |
| 5 | `05_yolov8s_gfpn_ema` | 최종 제안 모델 |

---

## 10. 옵션 설명

### `--skip-existing`

이미 완료된 실험은 다시 실행하지 않고 건너뜁니다.

중간에 학습이 끊긴 후 다시 실행할 때 유용합니다.

```bash
bash scripts/run_ablation.sh /home/djyeon/datasets/visdrone-yolo/data.yaml --skip-existing
```

### `--continue-on-error`

한 실험이 실패해도 전체 실행을 멈추지 않고 다음 실험으로 넘어갑니다.

ablation study를 장시간 돌릴 때 안전합니다.

### `tee -a logs/...`

터미널 출력 내용을 화면에 보여주면서 동시에 로그 파일에도 저장합니다.

---

## 11. tmux에서 빠져나오기

학습이 시작된 후 tmux에서 빠져나오려면 다음 키를 누릅니다.

```text
Ctrl + b
```

손을 뗀 뒤,

```text
d
```

즉,

```text
Ctrl + b → d
```

이렇게 하면 tmux 세션은 백그라운드에서 계속 실행됩니다.

SSH 터미널을 꺼도 GPU 학습은 계속 진행됩니다.

---

## 12. 다시 tmux에 접속하기

나중에 다시 확인할 때는 서버에 접속한 뒤 다음 명령을 실행합니다.

```bash
ssh djyeon@vis-yolo
tmux attach -t sod_ablation
```

---

## 13. tmux 세션 목록 확인

```bash
tmux ls
```

예시:

```text
sod_ablation: 1 windows (created Sun ...)
```

---

## 14. GPU 사용 상태 확인

```bash
nvidia-smi
```

현재 GPU 사용량, 메모리 사용량, 학습 프로세스를 확인할 수 있습니다.

---

## 15. 현재까지 summary 결과 보기

프로젝트 폴더로 이동합니다.

```bash
cd ~/Yolov8-Small-Object-Detection-Arial-Images
```

summary CSV 확인:

```bash
cat runs/ablation/ablation_summary.csv
```

좀 더 보기 좋게 확인:

```bash
column -s, -t < runs/ablation/ablation_summary.csv | less -S
```

---

## 16. 각 실험의 마지막 metric 확인

예를 들어 원본 YOLOv8s 실험 결과를 확인하려면:

```bash
tail -n 5 runs/ablation/01_yolov8s_original/results.csv
```

마지막 epoch 결과만 보려면:

```bash
tail -n 1 runs/ablation/01_yolov8s_original/results.csv
```

---

## 17. 진행 중인 로그 확인

```bash
tail -f logs/ablation_*.log
```

로그 확인을 종료하려면:

```text
Ctrl + c
```

---

## 18. 완료된 모델 weight 확인

```bash
ls runs/ablation/01_yolov8s_original/weights/
```

보통 다음 파일들이 생성됩니다.

```text
best.pt
last.pt
```

| 파일 | 의미 |
|---|---|
| `best.pt` | validation 성능이 가장 좋았던 모델 |
| `last.pt` | 마지막 epoch의 모델 |

---

## 19. 실험 결과 위치

ablation 전체 요약 결과:

```bash
runs/ablation/ablation_summary.csv
runs/ablation/ablation_summary.json
```

각 실험별 결과:

```bash
runs/ablation/01_yolov8s_original/
runs/ablation/02_yolov8s_ema/
runs/ablation/03_yolov8s_p2/
runs/ablation/04_yolov8s_gfpn/
runs/ablation/05_yolov8s_gfpn_ema/
```

각 실험 폴더 안에는 보통 다음 파일들이 있습니다.

```text
results.csv
weights/best.pt
weights/last.pt
args.yaml
```

---

## 20. AMP 에러가 계속 날 때 임시 우회

먼저 패치 재적용을 권장합니다.

```bash
python scripts/patch_ultralytics_for_sod.py ~/ultralytics_sod
```

그래도 같은 에러가 발생하면 AMP를 끄고 실행할 수 있습니다.

```bash
EPOCHS=100 IMGSZ=1280 BATCH=4 DEVICE=0 \
bash scripts/run_ablation.sh /home/djyeon/datasets/visdrone-yolo/data.yaml \
  --skip-existing --continue-on-error --no-amp \
  2>&1 | tee -a logs/ablation_$(date +%Y%m%d_%H%M%S).log
```

다만 RTX 4090에서는 AMP를 켜는 것이 속도 면에서 유리하므로, 가능하면 패치 적용 후 AMP를 유지하는 것이 좋습니다.

---

## 21. 자주 쓰는 명령 모음

### 서버 접속

```bash
ssh djyeon@vis-yolo
```

### tmux 생성

```bash
tmux new -s sod_ablation
```

### tmux 재접속

```bash
tmux attach -t sod_ablation
```

### tmux 목록

```bash
tmux ls
```

### tmux 빠져나오기

```text
Ctrl + b → d
```

### GPU 확인

```bash
nvidia-smi
```

### 로그 확인

```bash
tail -f logs/ablation_*.log
```

### summary 확인

```bash
cat runs/ablation/ablation_summary.csv
```

### summary 보기 좋게 확인

```bash
column -s, -t < runs/ablation/ablation_summary.csv | less -S
```

### 특정 실험 마지막 결과 확인

```bash
tail -n 1 runs/ablation/01_yolov8s_original/results.csv
```

---

## 22. 최종 실행 명령 요약

가장 중요한 실행 명령은 아래입니다.

```bash
ssh djyeon@vis-yolo
tmux new -s sod_ablation

cd ~/Yolov8-Small-Object-Detection-Arial-Images
conda activate sod_yolo

python scripts/patch_ultralytics_for_sod.py ~/ultralytics_sod

mkdir -p logs

EPOCHS=100 IMGSZ=1280 BATCH=4 DEVICE=0 \
bash scripts/run_ablation.sh /home/djyeon/datasets/visdrone-yolo/data.yaml \
  --skip-existing --continue-on-error \
  2>&1 | tee -a logs/ablation_$(date +%Y%m%d_%H%M%S).log
```

실행 후 tmux에서 빠져나오기:

```text
Ctrl + b → d
```