#!/bin/bash
set -e

echo "Inizio installazione dei modelli audio (MMAudio)..."

PYTHON_BIN="venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

MMAUDIO_DIR="./models/audio/mmaudio"
if [ ! -d "$MMAUDIO_DIR" ] || [ ! -f "$MMAUDIO_DIR/.download_complete" ]; then
    ./scripts/download_models.sh "mmaudio" "hkchengrex/MMAudio" "$MMAUDIO_DIR" "audio" "mmaudio"
else
    echo "[OK] Modello mmaudio già installato."
fi

WAN_2_2_5B_DIR="./models/video/wan_2_2_5b"
if [ ! -d "$WAN_2_2_5B_DIR" ] || [ ! -f "$WAN_2_2_5B_DIR/.download_complete" ]; then
    ./scripts/download_models.sh "wan_2_2_5b" "Wan-AI/Wan2.2-T2V-5B" "$WAN_2_2_5B_DIR" "video" "wan_2_2_5b"
else
    echo "[OK] Modello wan_2_2_5b già installato."
fi

echo "Installazione dei modelli audio e video completata."
