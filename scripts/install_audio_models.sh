#!/bin/bash
set -e

echo "Inizio installazione dei modelli audio (MMAudio)..."

MMAUDIO_STATUS=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/models.yaml')).get('audio', {}).get('mmaudio', {}).get('status', 'not_installed'))")
if [ "$MMAUDIO_STATUS" != "installed" ]; then
    ./scripts/download_models.sh "mmaudio" "hkchengrex/MMAudio" "./models/audio/mmaudio" "audio" "mmaudio"
else
    echo "[OK] Modello mmaudio già installato."
fi

echo "Installazione dei modelli audio completata."
