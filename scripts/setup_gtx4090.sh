#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-sod_yolo}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Load an Anaconda/Miniconda module first, or install Miniconda."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install ultralytics==8.2.0 opencv-python matplotlib pandas pyyaml tqdm scipy seaborn tensorboard

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
