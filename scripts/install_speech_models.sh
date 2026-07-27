#!/bin/bash
set -e

echo "Inizio installazione dei modelli speech (Whisper)..."

PYTHON_BIN="venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

WHISPER_DIR="./models/speech/whisper"
if [ ! -d "$WHISPER_DIR" ] || [ ! -f "$WHISPER_DIR/.download_complete" ]; then
    ./scripts/download_models.sh "whisper" "openai/whisper-large-v3-turbo" "$WHISPER_DIR" "speech" "whisper"
else
    echo "[OK] Modello whisper già installato."
fi

echo "Installazione dei modelli speech completata."
