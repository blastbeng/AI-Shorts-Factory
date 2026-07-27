from backend.services.social.base_social_provider import BaseSocialProvider

class YouTubeProvider(BaseSocialProvider):
    def authenticate(self, credentials: dict):
        return True

    def upload_video(self, video_path: str, metadata: dict):
        return {"post_id": "yt_123", "status": "uploaded"}

    def get_status(self, post_id: str):
        return {"status": "published"}
