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

LTX_DIR="./models/video/ltx_video"
LTX_MODEL="$LTX_DIR/ltx-video-13b-0.9.8-distilled.safetensors"
LTX_VAE_DIR="$LTX_DIR/vae"
LTX_VAE="$LTX_VAE_DIR/ltx-video-13b-0.9.8-vae.safetensors"

if [ ! -f "$LTX_MODEL" ] || [ ! -f "$LTX_VAE" ]; then
    mkdir -p "$LTX_DIR" "$LTX_VAE_DIR"
    echo "Downloading ltx-video-13b-0.9.8-distilled.safetensors..."
    wget -q --show-progress -O "$LTX_MODEL" "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-13b-0.9.8-distilled.safetensors"
    echo "Downloading ltx-video-13b-0.9.8-vae.safetensors..."
    wget -q --show-progress -O "$LTX_VAE" "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-13b-0.9.8-vae.safetensors"
    touch "$LTX_DIR/.download_complete"
else
    echo "[OK] Modello ltx_video già installato."
fi

echo "Installazione dei modelli video completata."
