#!/bin/bash
set -e

echo "Inizio installazione dei modelli immagine (Flux, Qwen Image)..."
./scripts/download_models.sh "flux" "black-forest-labs/FLUX.1-dev" "./models/image/flux" "image" "flux"
./scripts/download_models.sh "qwen_image" "Qwen/Qwen2-VL-2B-Instruct" "./models/image/qwen_image" "image" "qwen_image"
echo "Installazione dei modelli immagine completata."
