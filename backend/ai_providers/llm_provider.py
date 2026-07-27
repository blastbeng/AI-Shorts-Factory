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
        
        # Leggi configurazioni API da .env
        self.api_base = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
        self.api_key = os.getenv("LLM_API_KEY", "ollama") # Ollama richiede una key fittizia
        self.model_name = os.getenv("LLM_MODEL_NAME", self.model_info.get("model_name", "gpt-3.5-turbo"))
        
        self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)

    def install_status(self):
        # Considerato installato se l'API base è configurata
        return "installed" if self.api_base else "not_installed"

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, prompt: str, max_length: int = 500, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("LLM non configurato. Controlla il file .env.")
        
        logger.info(f"Generazione testo tramite LLM ({self.model_name}) su {self.api_base}")
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
        return {"type": "text", "model": self.model_name, "api_base": self.api_base}

    def get_gpu_requirements(self):
        # Basato su API, nessun requisito VRAM locale
        return {"vram_required_gb": 0, "backend": "api"}
