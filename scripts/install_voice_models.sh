#!/bin/bash
set -e

echo "Inizio installazione dei modelli voce (Kokoro TTS)..."

PYTHON_BIN="venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

KOKORO_STATUS=$("$PYTHON_BIN" -c "import yaml; print(yaml.safe_load(open('configs/models.yaml')).get('voice', {}).get('kokoro_tts', {}).get('status', 'not_installed'))")
if [ "$KOKORO_STATUS" != "installed" ]; then
    ./scripts/download_models.sh "kokoro_tts" "hexgrad/Kokoro-82M" "./models/voice/kokoro_tts" "voice" "kokoro_tts"
else
    echo "[OK] Modello kokoro_tts già installato."
fi

echo "Installazione dei modelli voce completata."
