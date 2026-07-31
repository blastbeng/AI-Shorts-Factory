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

if [ ! -f "$WAN_2_2_5B_DIR/.download_complete" ]; then
    mkdir -p "$WAN_2_2_5B_DIR/tokenizer"
    echo "Downloading Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors..."
    wget -q --show-progress -O "$WAN_2_2_5B_MODEL" "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/TI2V/Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors"
    
    echo "Downloading VAE, Text Encoder, and Tokenizer using huggingface_hub..."
    "$PYTHON_BIN" -c "
import os
from huggingface_hub import hf_hub_download, snapshot_download
token = os.getenv('HF_TOKEN')

# VAE
hf_hub_download(repo_id='Wan-AI/Wan2.1-T2V-14B', filename='Wan2.1_VAE.pth', local_dir='$WAN_2_2_5B_DIR', token=token)

# Text Encoder
snapshot_download(repo_id='Wan-AI/Wan2.1-T2V-14B-Diffusers', allow_patterns='text_encoder/*', local_dir='$WAN_2_2_5B_DIR', token=token)

# Tokenizer
for f in ['tokenizer.json', 'spiece.model', 'tokenizer_config.json', 'special_tokens_map.json']:
    hf_hub_download(repo_id='Wan-AI/Wan2.1-T2V-14B-Diffusers', filename=f'tokenizer/{f}', local_dir='$WAN_2_2_5B_DIR', token=token)
"
    
    touch "$WAN_2_2_5B_DIR/.download_complete"
else
    echo "[OK] Modello wan_2_2_5b già installato."
fi

echo "Installazione dei modelli video completata."
