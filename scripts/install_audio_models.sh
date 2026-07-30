#!/bin/bash
set -e

echo "Inizio installazione dei modelli audio (MMAudio)..."

PYTHON_BIN="venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

MMAUDIO_DIR="./models/audio/mmaudio"
if [ ! -d "$MMAUDIO_DIR" ] || [ ! -f "$MMAUDIO_DIR/.download_complete" ]; then
    ./scripts/download_models.sh "mmaudio" "hkchengrex/MMAudio" "$MMAUDIO_DIR" "audio" "mmaudio"
else
    echo "[OK] Modello mmaudio già installato."
fi

echo "Installazione dei modelli audio completata."
