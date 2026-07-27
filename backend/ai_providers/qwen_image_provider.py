import yaml
from backend.ai_providers.base_provider import BaseAIProvider

class QwenImageProvider(BaseAIProvider):
    def __init__(self):
        with open("configs/models.yaml", "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("image", {}).get("qwen_image", {})

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, *args, **kwargs):
        raise NotImplementedError("Qwen Image generation not implemented yet.")

    def get_capabilities(self):
        return {"type": "image", "model": "qwen_image"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb"), "backend": self.model_info.get("backend")}
