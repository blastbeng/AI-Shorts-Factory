#!/bin/bash
set -e

# Uso: ./download_models.sh <nome_modello> <repo_id_huggingface> <directory_destinazione>
MODEL_NAME=$1
REPO_ID=$2
DEST_DIR=$3

if [ -z "$MODEL_NAME" ] || [ -z "$REPO_ID" ] || [ -z "$DEST_DIR" ]; then
    echo "Uso: $0 <nome_modello> <repo_id_huggingface> <directory_destinazione>"
    exit 1
fi

echo "Download del modello: $MODEL_NAME da $REPO_ID"
mkdir -p "$DEST_DIR"

if command -v huggingface-cli &> /dev/null; then
    echo "Utilizzo di huggingface-cli per il download..."
    huggingface-cli download "$REPO_ID" --local-dir "$DEST_DIR" --local-dir-use-symlinks False
else
    echo "huggingface-cli non trovato. Installa 'huggingface_hub'."
    exit 1
fi

echo "Download completato per $MODEL_NAME in $DEST_DIR."
