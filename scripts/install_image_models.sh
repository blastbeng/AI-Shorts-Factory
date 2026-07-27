#!/bin/bash
set -e

echo "Inizio installazione dei modelli immagine (Flux, Qwen Image)..."

PYTHON_BIN="venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

FLUX_STATUS=$("$PYTHON_BIN" -c "import yaml; print(yaml.safe_load(open('configs/models.yaml')).get('image', {}).get('flux', {}).get('status', 'not_installed'))")
if [ "$FLUX_STATUS" != "installed" ]; then
    ./scripts/download_models.sh "flux" "lllyasviel/flux1-dev-bnb-nf4" "./models/image/flux" "image" "flux"
else
    echo "[OK] Modello flux già installato."
fi

QWEN_STATUS=$("$PYTHON_BIN" -c "import yaml; print(yaml.safe_load(open('configs/models.yaml')).get('image', {}).get('qwen_image', {}).get('status', 'not_installed'))")
if [ "$QWEN_STATUS" != "installed" ]; then
    ./scripts/download_models.sh "qwen_image" "Qwen/Qwen2-VL-2B-Instruct" "./models/image/qwen_image" "image" "qwen_image"
else
    echo "[OK] Modello qwen_image già installato."
fi

echo "Installazione dei modelli immagine completata."
