import subprocess
import json
import os
from backend.services.logger import logger

class QualityScorer:
    def score(self, video_path):
        if not os.path.exists(video_path):
            return 0.0

        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            streams = data.get("streams", [])
            video_stream = next((s for s in streams if s["codec_type"] == "video"), None)

            if not video_stream:
                return 0.0

            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))
            bitrate = int(data.get("format", {}).get("bit_rate", 0))

            score = 0
            if width >= 1080 and height >= 1920:
                score += 5
            elif width >= 720 and height >= 1280:
                score += 3
            else:
                score += 1

            if bitrate > 5000000:
                score += 3
            elif bitrate > 2000000:
                score += 2
            else:
                score += 1

            return min(score / 10.0 * 10.0, 10.0)

        except Exception as e:
            logger.error(f"Errore durante il quality scoring: {e}")
            return 0.0
