#!/bin/bash
set -e

echo "Inizio installazione dei modelli speech (Whisper)..."

WHISPER_STATUS=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/models.yaml')).get('speech', {}).get('whisper', {}).get('status', 'not_installed'))")
if [ "$WHISPER_STATUS" != "installed" ]; then
    ./scripts/download_models.sh "whisper" "openai/whisper-large-v3" "./models/speech/whisper" "speech" "whisper"
else
    echo "[OK] Modello whisper già installato."
fi

echo "Installazione dei modelli speech completata."
