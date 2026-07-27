#!/bin/bash
set -e

echo "Inizio installazione dei modelli video..."

PYTHON_BIN="venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

# Verifica Wan 2.2
WAN_STATUS=$("$PYTHON_BIN" -c "import yaml; print(yaml.safe_load(open('configs/models.yaml')).get('video', {}).get('wan_2_2_5b', {}).get('status', 'not_installed'))")
if [ "$WAN_STATUS" != "installed" ]; then
    ./scripts/download_models.sh "wan_2_2_5b" "Wan-AI/Wan2.2-T2V-5B" "./models/video/wan_2_2_5b" "video" "wan_2_2_5b"
else
    echo "[OK] Modello wan_2_2_5b già installato."
fi

# Verifica LTX Video
LTX_STATUS=$("$PYTHON_BIN" -c "import yaml; print(yaml.safe_load(open('configs/models.yaml')).get('video', {}).get('ltx_video', {}).get('status', 'not_installed'))")
if [ "$LTX_STATUS" != "installed" ]; then
    ./scripts/download_models.sh "ltx_video" "Lightricks/LTX-Video" "./models/video/ltx_video" "video" "ltx_video"
else
    echo "[OK] Modello ltx_video già installato."
fi

echo "Installazione dei modelli video completata."
