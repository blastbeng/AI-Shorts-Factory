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
    mkdir -p "$WAN_2_2_5B_DIR/transformer_config"
    cat << 'EOF' > "$WAN_2_2_5B_DIR/transformer_config/config.json"
{
  "_class_name": "WanTransformer3DModel",
  "dim": 3072,
  "num_heads": 24,
  "num_attention_heads": 24,
  "num_layers": 30,
  "patch_size": [1, 2, 2],
  "text_dim": 3072,
  "cross_attn_dim": 3072,
  "in_channels": 48,
  "out_channels": 48,
  "freq_dim": 256,
  "ffn_dim": 14336,
  "text_len": 512,
  "num_freqs": 1,
  "rope_theta": [1.0, 1.0, 1.0],
  "guidance_embed": false,
  "attention_head_dim": 128,
  "mlp_ratio": 4.0,
  "norm_eps": 1e-6
}
EOF
    echo "Downloading Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors..."
    wget -q --show-progress -O "$WAN_2_2_5B_MODEL" "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/TI2V/Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors"
    
    if [ -z "$HF_TOKEN" ]; then
        echo "Warning: HF_TOKEN environment variable is not set. Downloads will be unauthenticated and may be rate-limited."
    else
        echo "HF_TOKEN found. Using authenticated requests for Hugging Face Hub."
    fi

    echo "Downloading VAE, Text Encoder, and Tokenizer using huggingface_hub..."
    "$PYTHON_BIN" -c "
import os
from huggingface_hub import hf_hub_download, snapshot_download
token = os.getenv('HF_TOKEN')
if not token:
    print('Proceeding without HF_TOKEN. Unauthenticated requests may be rate-limited.')

# VAE
hf_hub_download(repo_id='Wan-AI/Wan2.1-T2V-14B', filename='Wan2.1_VAE.pth', local_dir='$WAN_2_2_5B_DIR', token=token)

# Text Encoder
snapshot_download(repo_id='Wan-AI/Wan2.1-T2V-14B-Diffusers', allow_patterns='text_encoder/*', local_dir='$WAN_2_2_5B_DIR', token=token)

# Tokenizer
for f in ['tokenizer.json', 'spiece.model', 'tokenizer_config.json', 'special_tokens_map.json']:
    hf_hub_download(repo_id='Wan-AI/Wan2.1-T2V-14B-Diffusers', filename=f'tokenizer/{f}', local_dir='$WAN_2_2_5B_DIR', token=token)

# Image Encoder (CLIP) and Feature Extractor
snapshot_download(repo_id='openai/clip-vit-large-patch14', local_dir='$WAN_2_2_5B_DIR/clip_image_encoder', token=token)
"
    
    touch "$WAN_2_2_5B_DIR/.download_complete"
else
    echo "[OK] Modello wan_2_2_5b già installato."
fi

echo "Installazione dei modelli video completata."
