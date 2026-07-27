import yaml
from backend.ai_providers.base_provider import BaseAIProvider

class KokoroProvider(BaseAIProvider):
    def __init__(self):
        with open("configs/models.yaml", "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("voice", {}).get("kokoro_tts", {})

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, *args, **kwargs):
        raise NotImplementedError("Kokoro TTS generation not implemented yet.")

    def get_capabilities(self):
        return {"type": "voice", "model": "kokoro_tts"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb"), "backend": self.model_info.get("backend")}
