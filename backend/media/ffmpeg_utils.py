import subprocess
from backend.services.logger import logger

class FFmpegUtils:
    @staticmethod
    def assemble_video(video_path: str, voice_path: str, audio_path: str, output_path: str):
        """
        Unisce un video, una traccia vocale e una traccia audio/effetti in un unico file.
        """
        logger.info(f"Assemblaggio video: {video_path} + {voice_path} + {audio_path} -> {output_path}")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", voice_path,
            "-i", audio_path,
            "-map", "0:v:0",
            "-filter_complex", "[1:a:0][2:a:0]amix=inputs=2:duration=longest[a]",
            "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac",
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Assemblaggio completato con successo.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Errore durante l'assemblaggio ffmpeg: {e.stderr}")
            return False
