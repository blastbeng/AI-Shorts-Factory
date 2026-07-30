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
    mkdir -p "$WAN_2_2_5B_DIR/tokenizer"
    echo "Downloading Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors..."
    wget -q --show-progress -O "$WAN_2_2_5B_MODEL" "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/TI2V/Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors"
    
    echo "Downloading VAE (wan_2.1_vae.safetensors)..."
    wget -q --show-progress -O "$WAN_2_2_5B_DIR/wan_2.1_vae.safetensors" "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/vae/wan_2.1_vae.safetensors"
    
    echo "Downloading Text Encoder (umt5_xxl.safetensors)..."
    wget -q --show-progress -O "$WAN_2_2_5B_DIR/umt5_xxl.safetensors" "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models/umt5_xxl.safetensors"
    
    echo "Downloading Tokenizer files..."
    wget -q --show-progress -O "$WAN_2_2_5B_DIR/tokenizer/tokenizer.json" "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/tokenizer/tokenizer.json"
    wget -q --show-progress -O "$WAN_2_2_5B_DIR/tokenizer/spiece.model" "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/tokenizer/spiece.model"
    wget -q --show-progress -O "$WAN_2_2_5B_DIR/tokenizer/tokenizer_config.json" "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/tokenizer/tokenizer_config.json"
    wget -q --show-progress -O "$WAN_2_2_5B_DIR/tokenizer/special_tokens_map.json" "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/tokenizer/special_tokens_map.json"
    
    touch "$WAN_2_2_5B_DIR/.download_complete"
else
    echo "[OK] Modello wan_2_2_5b già installato."
fi

echo "Installazione dei modelli video completata."
