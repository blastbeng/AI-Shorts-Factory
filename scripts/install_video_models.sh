#!/bin/bash
set -e

echo "Inizio installazione dei modelli video..."

PYTHON_BIN="venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

LTX_DIR="./models/video/ltx_video"
LTX_MODEL="$LTX_DIR/ltxv-13b-0.9.8-dev-fp8.safetensors"

if [ ! -f "$LTX_MODEL" ]; then
    mkdir -p "$LTX_DIR"
    echo "Downloading ltxv-13b-0.9.8-dev-fp8.safetensors..."
    wget -q --show-progress -O "$LTX_MODEL" "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.8-dev-fp8.safetensors"
    touch "$LTX_DIR/.download_complete"
else
    echo "[OK] Modello ltx_video già installato."
fi

WAN_2_2_5B_DIR="./models/video/wan_2_2_5b"
WAN_2_2_5B_MODEL="$WAN_2_2_5B_DIR/Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors"

if [ ! -f "$WAN_2_2_5B_MODEL" ]; then
    mkdir -p "$WAN_2_2_5B_DIR"
    echo "Downloading Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors..."
    wget -q --show-progress -O "$WAN_2_2_5B_MODEL" "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/TI2V/Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors"
    touch "$WAN_2_2_5B_DIR/.download_complete"
else
    echo "[OK] Modello wan_2_2_5b già installato."
fi

echo "Installazione dei modelli video completata."
