#!/bin/bash
set -e

echo "Inizio installazione dei modelli voce (Kokoro TTS)..."
./scripts/download_models.sh "kokoro_tts" "hexgrad/Kokoro-82M" "./models/voice/kokoro_tts"
echo "Installazione dei modelli voce completata."
