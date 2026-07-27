from backend.services.social.base_social_provider import BaseSocialProvider

class TikTokProvider(BaseSocialProvider):
    def authenticate(self, credentials: dict):
        # TODO: Implementare OAuth2 di TikTok
        return True

    def upload_video(self, video_path: str, metadata: dict):
        # TODO: Implementare upload video
        return {"post_id": "tiktok_123", "status": "uploaded"}

    def get_status(self, post_id: str):
        # TODO: Implementare check status
        return {"status": "published"}
