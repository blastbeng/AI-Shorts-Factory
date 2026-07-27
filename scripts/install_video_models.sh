#!/bin/bash
set -e

echo "Inizio installazione dei modelli video..."
./scripts/download_models.sh "wan_2_2_5b" "Wan-AI/Wan2.2-T2V-5B" "./models/video/wan_2_2_5b" "video" "wan_2_2_5b"
./scripts/download_models.sh "ltx_video" "Lightricks/LTX-Video" "./models/video/ltx_video" "video" "ltx_video"
echo "Installazione dei modelli video completata."
