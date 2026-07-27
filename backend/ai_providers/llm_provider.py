import os
import yaml
import subprocess
import threading
import time
from openai import OpenAI
from backend.ai_providers.base_provider import BaseAIProvider
from backend.services.logger import logger
from backend.services.subprocess_manager import SubprocessManager

class LLMProvider(BaseAIProvider):
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
            self.bin_path = os.getenv("LLAMA_CPP_BIN_PATH", "/opt/llama.cpp/build/bin/llama-cli")
            self.model_path = os.getenv("LLAMA_CPP_MODEL_PATH", "/opt/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf")
            self.params = os.getenv("LLAMA_CPP_PARAMS", "")
        else:
            self.api_base = None
            self.api_key = None
            self.model_name = None

    def install_status(self):
        if self.provider_type == "llama_cpp":
            return "installed" if os.path.exists(self.model_path) else "not_installed"
        return "installed" if self.api_base else "not_installed"

    def health_check(self):
        if self.provider_type == "llama_cpp":
            return os.path.exists(self.model_path) and os.path.exists(self.bin_path)
        return self.install_status() == "installed"

    def generate(self, prompt: str, max_length: int = 500, *args, is_interrupted=None, **kwargs):
        if not self.health_check():
            raise RuntimeError("LLM non configurato. Controlla il file .env e LLM_PROVIDER.")
        
        logger.info(f"Generazione testo tramite LLM ({self.provider_type})")
        logger.info(f"Prompt inviato a llama.cpp:\n{prompt}")
        try:
            if self.provider_type == "llama_cpp":
                import tempfile
                # Scrive il prompt in un file temporaneo per evitare problemi di escaping
                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
                    temp_file.write(prompt)
                    temp_file_path = temp_file.name
                
                cmd = [
                    "stdbuf", "-eL", "-oL",
                    self.bin_path,
                    "-m", self.model_path,
                    "-f", temp_file_path,
                    "-n", str(max_length)
                ] + self.params.split()
                
                try:
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    SubprocessManager.add(process)
                    
                    stdout_lines = []
                    
                    def read_stream(stream, is_stderr):
                        while True:
                            line = stream.readline()
                            if line == '' and process.poll() is not None:
                                break
                            if line:
                                if is_stderr:
                                    logger.info(f"[llama.cpp] {line.strip()}")
                                else:
                                    logger.info(f"[llama.cpp output] {line.strip()}")
                                    stdout_lines.append(line)
                    
                    stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, False), daemon=True)
                    stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, True), daemon=True)
                    stdout_thread.start()
                    stderr_thread.start()
                    
                    def monitor_interruption(proc, check_func):
                        while proc.poll() is None:
                            if check_func and check_func():
                                logger.warning("Job interrotto, uccisione processo llama.cpp...")
                                proc.kill()
                                break
                            time.sleep(0.5)
                    
                    interrupt_thread = threading.Thread(target=monitor_interruption, args=(process, is_interrupted), daemon=True)
                    interrupt_thread.start()
                    
                    process.wait()
                    stdout_thread.join()
                    stderr_thread.join()
                    interrupt_thread.join(timeout=1)
                    
                    if process.returncode != 0:
                        if is_interrupted and is_interrupted():
                            return ""
                        raise RuntimeError("llama.cpp exited with non-zero status")
                    
                    generated_text = "".join(stdout_lines).strip()
                    SubprocessManager.remove(process)
                    return generated_text
                finally:
                    os.remove(temp_file_path)
            else:
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
        return {"type": "text", "provider": self.provider_type}

    def get_gpu_requirements(self):
        if self.provider_type == "llama_cpp":
            return {"vram_required_gb": 0, "backend": "vulkan"}
        return {"vram_required_gb": 0, "backend": "api"}
