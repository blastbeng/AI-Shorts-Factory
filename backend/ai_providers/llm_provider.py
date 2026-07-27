import os
import re
import json
import yaml
from openai import OpenAI
from backend.ai_providers.base_provider import BaseAIProvider
from backend.services.logger import logger

class LLMProvider(BaseAIProvider):
    _llm_instance = None
    _llm_model_path = None

    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("text", {}).get("llm_base", {})
        
        self.provider_type = os.getenv("LLM_PROVIDER", "llama_cpp").lower()
        self.client = None
        
        if self.provider_type == "openai":
            self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            self.api_key = os.getenv("OPENAI_API_KEY", "")
            self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
            self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        elif self.provider_type == "ollama":
            self.api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
            self.api_key = os.getenv("OLLAMA_API_KEY", "ollama")
            self.model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3")
            self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        elif self.provider_type == "llama_cpp":
            self.model_path = os.getenv("LLAMA_CPP_MODEL_PATH", self.model_info.get("path", "/opt/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf"))
            self.params = os.getenv("LLAMA_CPP_PARAMS", "")
            
            # Parse params
            params_list = self.params.split()
            self.n_gpu_layers = 0
            self.n_ctx = 4096
            self.n_threads = 8
            self.n_batch = 512
            self.n_ubatch = 512
            self.tensor_split = None
            self.flash_attn = False
            self.type_k = 0  # Default (F16)
            self.type_v = 0  # Default (F16)
            self.temperature = 0.8
            self.top_p = 0.95
            self.top_k = 40
            self.repeat_penalty = 1.1
            
            for i, p in enumerate(params_list):
                if p == "-ngl" and i+1 < len(params_list): self.n_gpu_layers = int(params_list[i+1])
                elif p == "-c" and i+1 < len(params_list): self.n_ctx = int(params_list[i+1])
                elif p == "-t" and i+1 < len(params_list): self.n_threads = int(params_list[i+1])
                elif p == "-b" and i+1 < len(params_list): self.n_batch = int(params_list[i+1])
                elif p == "-ub" and i+1 < len(params_list): self.n_ubatch = int(params_list[i+1])
                elif p == "--tensor-split" and i+1 < len(params_list): self.tensor_split = [float(x) for x in params_list[i+1].split(",")]
                elif p == "--flash-attn" and i+1 < len(params_list): self.flash_attn = params_list[i+1].lower() == "on"
                elif p == "--cache-type-k" and i+1 < len(params_list):
                    val = params_list[i+1].lower()
                    if val == "q4_0": self.type_k = 2
                    elif val == "q8_0": self.type_k = 8
                    elif val == "f16": self.type_k = 1
                elif p == "--cache-type-v" and i+1 < len(params_list):
                    val = params_list[i+1].lower()
                    if val == "q4_0": self.type_v = 2
                    elif val == "q8_0": self.type_v = 8
                    elif val == "f16": self.type_v = 1
                elif p == "--temp" and i+1 < len(params_list): self.temperature = float(params_list[i+1])
                elif p == "--top-p" and i+1 < len(params_list): self.top_p = float(params_list[i+1])
                elif p == "--top-k" and i+1 < len(params_list): self.top_k = int(params_list[i+1])
                elif p == "--repeat-penalty" and i+1 < len(params_list): self.repeat_penalty = float(params_list[i+1])
            
            self._load_model()
        else:
            self.api_base = None
            self.api_key = None
            self.model_name = None

    def _load_model(self):
        if LLMProvider._llm_instance is None or LLMProvider._llm_model_path != self.model_path:
            logger.info(f"Caricamento modello llama.cpp: {self.model_path}")
            from llama_cpp import Llama
            n_gpu_layers = self.n_gpu_layers
            while True:
                try:
                    LLMProvider._llm_instance = Llama(
                        model_path=self.model_path,
                        n_gpu_layers=n_gpu_layers,
                        n_ctx=self.n_ctx,
                        n_threads=self.n_threads,
                        n_batch=self.n_batch,
                        n_ubatch=self.n_ubatch,
                        tensor_split=self.tensor_split,
                        flash_attn=self.flash_attn,
                        type_k=self.type_k,
                        type_v=self.type_v,
                        verbose=False
                    )
                    break
                except Exception as e:
                    logger.exception(f"Errore nel caricamento del modello LLM con n_gpu_layers={n_gpu_layers}. Riduzione layer su GPU.")
                    if n_gpu_layers > 1:
                        n_gpu_layers = max(1, n_gpu_layers // 2)
                    else:
                        raise
            LLMProvider._llm_model_path = self.model_path
        self.llm = LLMProvider._llm_instance

    def install_status(self):
        if self.provider_type == "llama_cpp":
            return "installed" if os.path.exists(self.model_path) else "not_installed"
        return "installed" if self.api_base else "not_installed"

    def health_check(self):
        if self.provider_type == "llama_cpp":
            return os.path.exists(self.model_path)
        return self.install_status() == "installed"

    def generate(self, prompt: str, max_length: int = 500, *args, is_interrupted=None, **kwargs):
        if not self.health_check():
            raise RuntimeError("LLM non configurato. Controlla il file .env e LLM_PROVIDER.")
        
        logger.info(f"Generazione testo tramite LLM ({self.provider_type})")
        logger.info(f"Prompt inviato a LLM: {prompt}")
        try:
            if self.provider_type == "llama_cpp":
                if is_interrupted and is_interrupted():
                    return ""
                
                system_prompt = f"You are a professional scriptwriter. Follow the user's instructions exactly. Do not output any thinking process, reasoning, meta-text, prompt analysis, or step-by-step breakdowns. Output ONLY a valid JSON object with a single key 'content' containing the final text. The output MUST be in the language requested by the user. Do not include any introductory or concluding remarks. Do not output the prompt or any part of it. Never output your internal thoughts or translate the prompt. Example: {{\"content\": \"The generated text here.\"}}"
                # Aggiunge /no_think al prompt per disabilitare il processo di pensiero nei modelli Qwen3
                user_prompt = f"{prompt}\n/no_think"
                
                response = self.llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repeat_penalty=self.repeat_penalty,
                    stream=False,
                    response_format={"type": "json_object"}
                )
                generated_text = response["choices"][0]["message"]["content"].strip()
                
                # Estrazione del contenuto dal JSON
                try:
                    parsed_json = json.loads(generated_text)
                    generated_text = parsed_json.get("content", "").strip()
                except json.JSONDecodeError:
                    # Fallback: estrazione del campo content tramite regex
                    match = re.search(r'"content"\s*:\s*"(.*?)"', generated_text, flags=re.DOTALL)
                    if match:
                        generated_text = match.group(1).strip()
                    else:
                        # Fallback finale: rimozione di eventuali tag di pensiero e meta-testo
                        generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()
                        generated_text = re.sub(r'<thought>.*?</thought>', '', generated_text, flags=re.DOTALL).strip()
                        generated_text = re.sub(r'<reasoning>.*?</reasoning>', '', generated_text, flags=re.DOTALL).strip()
                        generated_text = re.sub(r'^.*?(Here\'s a thinking process:|Thinking Process:).*?\n', '', generated_text, flags=re.IGNORECASE).strip()
                        generated_text = re.sub(r'^(Sure, here is|Here is|Here\'s|This is|Il seguente è|Ecco).*?:\s*', '', generated_text, flags=re.IGNORECASE).strip()
                        generated_text = re.sub(r'\*+', '', generated_text).strip()
                        generated_text = re.sub(r'#+', '', generated_text).strip()
                
                if is_interrupted and is_interrupted():
                    return ""
                
                logger.info(f"Risposta ricevuta da LLM: {generated_text}")
                return generated_text
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_length
                )
                generated_text = response.choices[0].message.content
                
                logger.info(f"Risposta ricevuta da LLM: {generated_text}")
                return generated_text
        except Exception as e:
            logger.error(f"Errore nella generazione LLM: {e}")
            raise

    def get_capabilities(self):
        return {"type": "text", "provider": self.provider_type}

    def get_gpu_requirements(self):
        if self.provider_type == "llama_cpp":
            return {"vram_required_gb": 0, "backend": "vulkan"}
        return {"vram_required_gb": 0, "backend": "api"}

    def cleanup(self):
        if self.provider_type == "llama_cpp":
            if LLMProvider._llm_instance is not None:
                del LLMProvider._llm_instance
                LLMProvider._llm_instance = None
                LLMProvider._llm_model_path = None
            self.llm = None
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
