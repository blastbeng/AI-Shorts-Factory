#!/bin/bash
set -e

echo "Inizio installazione dei modelli video..."

PYTHON_BIN="venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

WAN_DIR="./models/video/wan_2_1_1_3b"
if [ ! -d "$WAN_DIR" ] || [ ! -f "$WAN_DIR/.download_complete" ]; then
    ./scripts/download_models.sh "wan_2_1_1_3b" "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" "$WAN_DIR" "video" "wan_2_1_1_3b"
else
    echo "[OK] Modello wan_2_1_1_3b già installato."
fi

LTX_FILE="./models/video/ltxv-13b-0.9.8-distilled-fp8.safetensors"
if [ ! -f "$LTX_FILE" ]; then
    echo "[INFO] Downloading LTX Video 13b FP8 model..."
    mkdir -p "./models/video"
    wget -O "$LTX_FILE" "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.8-distilled-fp8.safetensors"
else
    echo "[OK] Modello ltx_video già installato."
fi

echo "Installazione dei modelli video completata."
