#!/bin/bash
set -e

echo "Inizio installazione dei modelli voce (Kokoro TTS)..."

PYTHON_BIN="venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

KOKORO_DIR="./models/voice/kokoro_tts"
if [ ! -d "$KOKORO_DIR" ] || [ ! -f "$KOKORO_DIR/.download_complete" ]; then
    ./scripts/download_models.sh "kokoro_tts" "hexgrad/Kokoro-82M" "$KOKORO_DIR" "voice" "kokoro_tts"
else
    echo "[OK] Modello kokoro_tts già installato."
fi

echo "Installazione dei modelli voce completata."
