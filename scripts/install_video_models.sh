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
WAN_2_2_5B_MODEL="$WAN_2_2_5B_DIR/wan22-i2v-a14b-high-q4-k-s.gguf"

if [ ! -f "$WAN_2_2_5B_DIR/.download_complete" ]; then
    mkdir -p "$WAN_2_2_5B_DIR"
    echo "Downloading wan22-i2v-a14b-high-q4-k-s.gguf..."
    wget -q --show-progress -O "$WAN_2_2_5B_MODEL" "https://huggingface.co/wangkanai/wan22-fp8-i2v-gguf/resolve/main/diffusion_models/wan/wan22-i2v-a14b-high-q4-k-s.gguf"
    
    touch "$WAN_2_2_5B_DIR/.download_complete"
else
    echo "[OK] Modello wan_2_2_5b già installato."
fi

WAN_BASE_DIR="./models/video/wan_2_2_5b/base_model"
if [ ! -d "$WAN_BASE_DIR" ]; then
    mkdir -p "$WAN_BASE_DIR"
    echo "Downloading Wan2.1-I2V-5B-480P-Diffusers base model..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-5B-480P-Diffusers --local-dir "$WAN_BASE_DIR"
else
    echo "[OK] Modello base wan_2_2_5b già installato."
fi

echo "Installazione dei modelli video completata."
