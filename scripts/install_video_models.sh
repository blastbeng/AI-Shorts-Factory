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
LTX_MODEL="$LTX_DIR/ltxv-13b-0.9.8-distilled-fp8.safetensors"
LTX_VAE_DIR="$LTX_DIR/vae"
LTX_VAE="$LTX_VAE_DIR/LTXV-13B-0.9.8-distilled-VAE.safetensors"

if [ ! -f "$LTX_MODEL" ] || [ ! -f "$LTX_VAE" ]; then
    mkdir -p "$LTX_DIR" "$LTX_VAE_DIR"
    echo "Downloading ltxv-13b-0.9.8-distilled-fp8.safetensors..."
    wget -q --show-progress -O "$LTX_MODEL" "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.8-distilled-fp8.safetensors"
    echo "Downloading LTXV-13B-0.9.8-distilled-VAE.safetensors..."
    wget -q --show-progress -O "$LTX_VAE" "https://huggingface.co/QuantStack/LTXV-13B-0.9.8-distilled-GGUF/resolve/main/vae/LTXV-13B-0.9.8-distilled-VAE.safetensors"
    touch "$LTX_DIR/.download_complete"
else
    echo "[OK] Modello ltx_video già installato."
fi

echo "Installazione dei modelli video completata."
