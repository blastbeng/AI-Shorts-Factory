#!/bin/bash
set -e

echo "Inizio installazione dei modelli immagine (Flux, Qwen Image)..."

PYTHON_BIN="venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

FLUX_GGUF="./models/image/flux1-schnell-Q8_0.gguf"
if [ ! -f "$FLUX_GGUF" ]; then
    echo "[INFO] Downloading Flux GGUF model..."
    mkdir -p "./models/image"
    wget -O "$FLUX_GGUF" "https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q8_0.gguf"
else
    echo "[OK] Flux GGUF model already installed."
fi

T5_GGUF="./models/image/t5-v1_1-xxl-encoder-Q8_0.gguf"
if [ ! -f "$T5_GGUF" ]; then
    echo "[INFO] Downloading T5 encoder GGUF model..."
    mkdir -p "./models/image"
    wget -O "$T5_GGUF" "https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q4_K_M.gguf"
else
    echo "[OK] T5 encoder GGUF model already installed."
fi

QWEN_DIR="./models/image/qwen_image"
if [ ! -d "$QWEN_DIR" ] || [ ! -f "$QWEN_DIR/.download_complete" ]; then
    ./scripts/download_models.sh "qwen_image" "Qwen/Qwen2-VL-2B-Instruct" "$QWEN_DIR" "image" "qwen_image"
else
    echo "[OK] Modello qwen_image già installato."
fi

echo "Installazione dei modelli immagine completata."
