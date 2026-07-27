#!/bin/bash
set -e

echo "Inizio installazione dei modelli immagine (Flux, Qwen Image)..."

PYTHON_BIN="venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

FLUX_DIR="./models/image/flux"
if [ ! -d "$FLUX_DIR" ] || [ ! -f "$FLUX_DIR/.download_complete" ]; then
    ./scripts/download_models.sh "flux" "lllyasviel/flux1-dev-bnb-nf4" "$FLUX_DIR" "image" "flux"
else
    echo "[OK] Modello flux già installato."
fi

QWEN_DIR="./models/image/qwen_image"
if [ ! -d "$QWEN_DIR" ] || [ ! -f "$QWEN_DIR/.download_complete" ]; then
    ./scripts/download_models.sh "qwen_image" "Qwen/Qwen2-VL-2B-Instruct" "$QWEN_DIR" "image" "qwen_image"
else
    echo "[OK] Modello qwen_image già installato."
fi

echo "Installazione dei modelli immagine completata."
