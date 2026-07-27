#!/bin/bash
set -e

echo "Inizio installazione dei modelli audio (MMAudio)..."

PYTHON_BIN="venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

MMAUDIO_STATUS=$("$PYTHON_BIN" -c "import yaml; print(yaml.safe_load(open('configs/models.yaml')).get('audio', {}).get('mmaudio', {}).get('status', 'not_installed'))")
if [ "$MMAUDIO_STATUS" != "installed" ]; then
    ./scripts/download_models.sh "mmaudio" "hkchengrex/MMAudio" "./models/audio/mmaudio" "audio" "mmaudio"
else
    echo "[OK] Modello mmaudio già installato."
fi

echo "Installazione dei modelli audio completata."
