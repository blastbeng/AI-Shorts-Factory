import os
from backend.services.social.base_social_provider import BaseSocialProvider
from backend.services.logger import logger

class TikTokProvider(BaseSocialProvider):
    def authenticate(self, credentials: dict = None):
        self.token = os.getenv("TIKTOK_ACCESS_TOKEN")
        if not self.token:
            raise ValueError("Token TikTok mancante. Configura TIKTOK_ACCESS_TOKEN nel file .env")
        logger.info("Autenticazione TikTok riuscita (simulata).")
        return True

    def upload_video(self, video_path: str, metadata: dict):
        logger.info(f"Upload video su TikTok: {video_path} con metadata: {metadata}")
        return {"post_id": "tiktok_123", "status": "uploaded", "message": "Upload simulato completato"}

    def get_status(self, post_id: str):
        logger.info(f"Controllo stato post TikTok: {post_id}")
        return {"status": "published", "post_id": post_id}
