import os
from backend.services.social.base_social_provider import BaseSocialProvider
from backend.services.logger import logger

class YouTubeProvider(BaseSocialProvider):
    def authenticate(self, credentials: dict = None):
        self.token = os.getenv("YOUTUBE_API_KEY")
        if not self.token:
            raise ValueError("API Key YouTube mancante. Configura YOUTUBE_API_KEY nel file .env")
        logger.info("Autenticazione YouTube riuscita (simulata).")
        return True

    def upload_video(self, video_path: str, metadata: dict):
        logger.info(f"Upload video su YouTube: {video_path} con metadata: {metadata}")
        return {"post_id": "yt_123", "status": "uploaded", "message": "Upload simulato completato"}

    def get_status(self, post_id: str):
        logger.info(f"Controllo stato post YouTube: {post_id}")
        return {"status": "published", "post_id": post_id}
