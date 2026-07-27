import os
import yaml
from openai import OpenAI
from backend.ai_providers.base_provider import BaseAIProvider
from backend.services.logger import logger

class LLMProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("text", {}).get("llm_base", {})
        
        self.provider_type = os.getenv("LLM_PROVIDER", "ollama").lower()
        
        if self.provider_type == "openai":
            self.api_base = "https://api.openai.com/v1"
            self.api_key = os.getenv("OPENAI_API_KEY", "")
            self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        elif self.provider_type == "ollama":
            # Ollama espone un'API compatibile con OpenAI all'endpoint /v1
            base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            self.api_base = f"{base}/v1"
            self.api_key = "ollama" # Ollama non richiede una vera API key, ma il client ne ha bisogno
            self.model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3")
        else:
            self.api_base = None
            self.api_key = None
            self.model_name = None
            
        if self.api_base:
            self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)

    def install_status(self):
        return "installed" if self.api_base else "not_installed"

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, prompt: str, max_length: int = 500, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("LLM non configurato. Controlla il file .env e LLM_PROVIDER.")
        
        logger.info(f"Generazione testo tramite LLM ({self.provider_type} - {self.model_name}) su {self.api_base}")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_length
            )
            generated_text = response.choices[0].message.content
            return generated_text
        except Exception as e:
            logger.error(f"Errore nella generazione LLM: {e}")
            raise

    def get_capabilities(self):
        return {"type": "text", "provider": self.provider_type, "model": self.model_name, "api_base": self.api_base}

    def get_gpu_requirements(self):
        # Basato su API, nessun requisito VRAM locale
        return {"vram_required_gb": 0, "backend": "api"}
