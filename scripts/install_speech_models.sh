#!/bin/bash
set -e

echo "Inizio installazione dei modelli speech (Whisper)..."

PYTHON_BIN="venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

WHISPER_STATUS=$("$PYTHON_BIN" -c "import yaml; print(yaml.safe_load(open('configs/models.yaml')).get('speech', {}).get('whisper', {}).get('status', 'not_installed'))")
if [ "$WHISPER_STATUS" != "installed" ]; then
    ./scripts/download_models.sh "whisper" "openai/whisper-large-v3" "./models/speech/whisper" "speech" "whisper"
else
    echo "[OK] Modello whisper già installato."
fi

echo "Installazione dei modelli speech completata."
