#!/bin/bash
set -e

echo "Inizio installazione dei modelli speech (Whisper)..."
./scripts/download_models.sh "whisper" "openai/whisper-large-v3" "./models/speech/whisper" "speech" "whisper"
echo "Installazione dei modelli speech completata."
