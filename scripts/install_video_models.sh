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
WAN_2_2_5B_MODEL="$WAN_2_2_5B_DIR/wan2.2_ti2v_5B_fp8.safetensors"

if [ ! -f "$WAN_2_2_5B_DIR/.download_complete" ]; then
    mkdir -p "$WAN_2_2_5B_DIR"
    echo "Downloading wan2.2_ti2v_5B_fp8.safetensors..."
    wget -q --show-progress -O "$WAN_2_2_5B_MODEL" "https://huggingface.co/meimeilook/Wan2.2-TI2V-5B_FP8/resolve/main/wan2.2_ti2v_5B_fp8.safetensors"
    touch "$WAN_2_2_5B_DIR/.download_complete"
else
    echo "[OK] Modello wan_2_2_5b già installato."
fi

WAN_BASE_DIR="./models/video/wan_2_2_5b/base_model"
if [ ! -f "$WAN_BASE_DIR/model_index.json" ]; then
    mkdir -p "$WAN_BASE_DIR"
    echo "Downloading Wan2.2-TI2V-5B-Diffusers base model..."
    $PYTHON_BIN -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.2-TI2V-5B-Diffusers', local_dir='$WAN_BASE_DIR', ignore_patterns=['transformer/*'])"
else
    echo "[OK] Modello base wan_2_2_5b già installato."
fi

echo "Installazione dei modelli video completata."
