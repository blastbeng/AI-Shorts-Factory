from backend.services.social.base_social_provider import BaseSocialProvider
from backend.services.logger import logger

class FacebookProvider(BaseSocialProvider):
    def authenticate(self, credentials: dict):
        self.token = credentials.get("token")
        if not self.token:
            raise ValueError("Token Facebook mancante")
        logger.info("Autenticazione Facebook riuscita (simulata).")
        return True

    def upload_video(self, video_path: str, metadata: dict):
        logger.info(f"Upload video su Facebook: {video_path} con metadata: {metadata}")
        return {"post_id": "fb_123", "status": "uploaded", "message": "Upload simulato completato"}

    def get_status(self, post_id: str):
        logger.info(f"Controllo stato post Facebook: {post_id}")
        return {"status": "published", "post_id": post_id}
