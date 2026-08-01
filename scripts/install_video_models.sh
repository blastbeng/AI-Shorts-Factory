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

WAN_2_2_14B_DIR="./models/video/wan_2_2_14b"
WAN_2_2_14B_MODEL="$WAN_2_2_14B_DIR/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors"

if [ ! -f "$WAN_2_2_14B_DIR/.download_complete" ]; then
    mkdir -p "$WAN_2_2_14B_DIR"
    echo "Downloading Wan2.2 14B FP8 model..."
    wget -q --show-progress -O "$WAN_2_2_14B_MODEL" "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors"
    touch "$WAN_2_2_14B_DIR/.download_complete"
else
    echo "[OK] Modello wan_2_2_14b già installato."
fi

WAN_2_2_14B_BASE_DIR="./models/video/wan_2_2_14b/base_model"

# Clean up any incomplete transformer variant folders (e.g. transformer_2)
# that would cause the pipeline to try loading missing shards.
if [ -d "$WAN_2_2_14B_BASE_DIR" ]; then
    echo "Cleaning up transformer variant folders in $WAN_2_2_14B_BASE_DIR ..."
    find "$WAN_2_2_14B_BASE_DIR" -maxdepth 1 -type d -name 'transformer_*' ! -name 'transformer' -exec rm -rf {} +
    # Also remove any weight files inside the main transformer folder (keep only config.json)
    if [ -d "$WAN_2_2_14B_BASE_DIR/transformer" ]; then
        find "$WAN_2_2_14B_BASE_DIR/transformer" -type f ! -name 'config.json' -delete
    fi
fi

if [ ! -f "$WAN_2_2_14B_BASE_DIR/model_index.json" ] || [ ! -f "$WAN_2_2_14B_BASE_DIR/transformer/config.json" ]; then
    mkdir -p "$WAN_2_2_14B_BASE_DIR"
    echo "Downloading Wan2.2-I2V-A14B-Diffusers base model (full)..."
    $PYTHON_BIN -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.2-I2V-A14B-Diffusers', local_dir='$WAN_2_2_14B_BASE_DIR', allow_patterns=['*.json', '*.txt', 'vae/**', 'scheduler/**', 'tokenizer/**', 'image_encoder/**', 'text_encoder/**', 'transformer/config.json'])"
else
    echo "[OK] Modello base wan_2_2_14b già installato."
fi

# Post-download cleanup to ensure no stray transformer variants or weight files remain
if [ -d "$WAN_2_2_14B_BASE_DIR" ]; then
    echo "Post-download cleanup of transformer variant folders in $WAN_2_2_14B_BASE_DIR ..."
    find "$WAN_2_2_14B_BASE_DIR" -maxdepth 1 -type d -name 'transformer_*' ! -name 'transformer' -exec rm -rf {} +
    if [ -d "$WAN_2_2_14B_BASE_DIR/transformer" ]; then
        find "$WAN_2_2_14B_BASE_DIR/transformer" -type f ! -name 'config.json' -delete
    fi
fi

echo "Installazione dei modelli video completata."
