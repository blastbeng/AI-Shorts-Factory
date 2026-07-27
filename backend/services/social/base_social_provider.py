from abc import ABC, abstractmethod

class BaseSocialProvider(ABC):
    @abstractmethod
    def authenticate(self, credentials: dict):
        pass

    @abstractmethod
    def upload_video(self, video_path: str, metadata: dict):
        pass

    @abstractmethod
    def get_status(self, post_id: str):
        pass
