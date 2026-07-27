import os
from backend.services.logger import logger

class ConfigValidator:
    @staticmethod
    def validate():
        errors = []
        warnings = []

        # Verifica file .env caricato
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            errors.append("DATABASE_URL non configurata")

        # Verifica LLM
        llm_provider = os.getenv("LLM_PROVIDER", "llama_cpp")
        if llm_provider == "llama_cpp":
            model_path = os.getenv("LLAMA_CPP_MODEL_PATH")
            bin_path = os.getenv("LLAMA_CPP_BIN_PATH")
            if not model_path:
                errors.append("LLAMA_CPP_MODEL_PATH non configurata")
            elif not os.path.exists(model_path):
                warnings.append(f"Modello llama.cpp non trovato in: {model_path}. Esegui scripts/install_llama_cpp.sh")
            if not bin_path:
                errors.append("LLAMA_CPP_BIN_PATH non configurata")
            elif not os.path.exists(bin_path):
                warnings.append(f"Binario llama.cpp non trovato in: {bin_path}. Esegui scripts/install_llama_cpp.sh")
        elif llm_provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                errors.append("OPENAI_API_KEY non configurata")
        elif llm_provider == "ollama":
            if not os.getenv("OLLAMA_API_BASE"):
                warnings.append("OLLAMA_API_BASE non configurata, uso default")

        # Verifica GPU config
        gpu_config = os.getenv("GPU_CONFIG_PATH", "configs/gpu.yaml")
        if not os.path.exists(gpu_config):
            errors.append(f"File configurazione GPU non trovato: {gpu_config}")

        # Verifica models config
        models_config = os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml")
        if not os.path.exists(models_config):
            errors.append(f"File configurazione modelli non trovato: {models_config}")

        # Verifica social tokens (solo warning)
        social_vars = ["TIKTOK_ACCESS_TOKEN", "YOUTUBE_API_KEY", "FACEBOOK_ACCESS_TOKEN", "INSTAGRAM_ACCESS_TOKEN"]
        for var in social_vars:
            val = os.getenv(var)
            if not val or val.startswith("your_"):
                warnings.append(f"{var} non configurata o usa valore placeholder")

        for e in errors:
            logger.error(f"[ConfigValidator] ERRORE: {e}")
        for w in warnings:
            logger.warning(f"[ConfigValidator] AVVISO: {w}")

        return len(errors) == 0

    @staticmethod
    def validate_and_exit():
        if not ConfigValidator.validate():
            logger.error("Validazione configurazione fallita. Impossibile avviare l'applicazione.")
            raise SystemExit(1)
