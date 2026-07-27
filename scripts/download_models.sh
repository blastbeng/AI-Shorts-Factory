#!/bin/bash
set -e

# Uso: ./download_models.sh <nome_modello> <url> <directory_destinazione>
MODEL_NAME=$1
URL=$2
DEST_DIR=$3

if [ -z "$MODEL_NAME" ] || [ -z "$URL" ] || [ -z "$DEST_DIR" ]; then
    echo "Uso: $0 <nome_modello> <url> <directory_destinazione>"
    exit 1
fi

echo "Download del modello: $MODEL_NAME"
echo "URL: $URL"
echo "Directory di destinazione: $DEST_DIR"

mkdir -p "$DEST_DIR"

# Verifica se huggingface-cli è disponibile, altrimenti usa wget
if command -v huggingface-cli &> /dev/null; then
    echo "Utilizzo di huggingface-cli per il download..."
    # huggingface-cli download <repo_id> --local-dir "$DEST_DIR"
    echo "TODO: Implementa il comando huggingface-cli specifico per $MODEL_NAME"
else
    echo "huggingface-cli non trovato, utilizzo di wget..."
    # wget -P "$DEST_DIR" "$URL"
    echo "TODO: Implementa il comando wget specifico per $MODEL_NAME"
fi

echo "Download completato per $MODEL_NAME."
