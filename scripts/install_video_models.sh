#!/bin/bash
set -e

echo "Inizio installazione dei modelli video..."

# Wan 2.2 5B
./scripts/download_models.sh "wan_2_2_5b" "Wan-AI/Wan2.2-T2V-5B" "./models/video/wan_2_2_5b"

# LTX Video
./scripts/download_models.sh "ltx_video" "Lightricks/LTX-Video" "./models/video/ltx_video"

echo "Installazione dei modelli video completata."
