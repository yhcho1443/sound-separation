#!/bin/bash
# Colab Environment Setup Script for Sound Separation (MixIT)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================================"
echo "[1/3] Installing system utilities (aria2, sox, libsox-fmt-all)..."
echo "============================================================"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq aria2 sox libsox-fmt-all
fi

echo "============================================================"
echo "[2/3] Installing Python dependencies for Colab..."
echo "============================================================"
python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r "${SCRIPT_DIR}/requirements_colab.txt"

echo "============================================================"
echo "[3/3] Verifying TensorFlow + Legacy Keras (TF v1 Graph Mode)..."
echo "============================================================"
export TF_USE_LEGACY_KERAS=1
export PYTHONPATH="${REPO_ROOT}/models:${REPO_ROOT}/models/neurips2020_mixit:${REPO_ROOT}/models/tools:${PYTHONPATH}"

python3 - << 'EOF'
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from train import tf_estimator_compat as tf_estimator
import model as mixit_model

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("Detected GPUs:", gpus)
hparams = mixit_model.get_model_hparams()
print("MixIT HParams loaded successfully. Num output sources:", len(hparams.signal_names))
EOF

echo "Colab setup completed successfully!"
