#!/bin/bash
set -e

echo "Inizio installazione dei modelli audio (MMAudio)..."
./scripts/download_models.sh "mmaudio" "hkchengrex/MMAudio" "./models/audio/mmaudio"
echo "Installazione dei modelli audio completata."
