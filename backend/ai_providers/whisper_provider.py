import os
import yaml
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger
import librosa

class WhisperProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("speech", {}).get("whisper", {})
        self.gm = GPUManager()
        self.model = None
        self.processor = None

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, audio_path: str, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello Whisper non installato.")
            
        preferred_backend = self.model_info.get("backend")
        gpu = self.gm.get_gpu_for_task("speech_recognition", self.get_gpu_requirements().get("vram_required_gb", 0), preferred_backend=preferred_backend)
        if not gpu:
            logger.warning("Nessuna GPU con VRAM sufficiente per Whisper. Uso GPU con offload su RAM.")
            gpu = self.gm.get_gpu_for_task_ignore_vram("speech_recognition", preferred_backend=preferred_backend)
            if not gpu:
                raise RuntimeError("Nessuna GPU assegnata per lo speech recognition.")
            use_cpu_offload = True
        else:
            use_cpu_offload = False
            
        device = self.gm.get_device_string(gpu['id'], preferred_backend=self.model_info.get("backend"))
        
        if self.model is None:
            logger.info("Caricamento modello Whisper...")
            model_path = self.model_info.get("path")
            try:
                self.processor = WhisperProcessor.from_pretrained(model_path)
                if use_cpu_offload:
                    self.model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
                else:
                    self.model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
            except Exception as e:
                logger.exception(f"Errore nel caricamento del modello su GPU. Fallback con offload su RAM.")
                if self.model is not None:
                    del self.model
                    self.model = None
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                self.model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
            
        logger.info(f"Trascrizione audio da: {audio_path}")
        audio, sampling_rate = librosa.load(audio_path, sr=16000)
        
        inputs = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        # Move inputs to the model's actual device and dtype (handles device_map="auto")
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        inputs = {k: v.to(model_device, model_dtype) for k, v in inputs.items()}
        
        with torch.no_grad():
            predicted_ids = self.model.generate(**inputs)
            
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        logger.info(f"Trascrizione completata: {transcription}")
        return transcription

    def generate_srt(self, audio_path: str, output_srt_path: str, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello Whisper non installato.")
            
        preferred_backend = self.model_info.get("backend")
        gpu = self.gm.get_gpu_for_task("speech_recognition", self.get_gpu_requirements().get("vram_required_gb", 0), preferred_backend=preferred_backend)
        if not gpu:
            logger.warning("Nessuna GPU con VRAM sufficiente per Whisper. Uso GPU con offload su RAM.")
            gpu = self.gm.get_gpu_for_task_ignore_vram("speech_recognition", preferred_backend=preferred_backend)
            if not gpu:
                raise RuntimeError("Nessuna GPU assegnata per lo speech recognition.")
            use_cpu_offload = True
        else:
            use_cpu_offload = False
            
        device = self.gm.get_device_string(gpu['id'], preferred_backend=self.model_info.get("backend"))
        
        if self.model is None:
            logger.info("Caricamento modello Whisper...")
            model_path = self.model_info.get("path")
            try:
                self.processor = WhisperProcessor.from_pretrained(model_path)
                if use_cpu_offload:
                    self.model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
                else:
                    self.model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
            except Exception as e:
                logger.exception(f"Errore nel caricamento del modello su GPU. Fallback con offload su RAM.")
                if self.model is not None:
                    del self.model
                    self.model = None
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                self.model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
            
        logger.info(f"Trascrizione audio per sottotitoli da: {audio_path}")
        audio, sampling_rate = librosa.load(audio_path, sr=16000)
        audio_duration = len(audio) / sampling_rate
        
        inputs = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        # Move inputs to the model's actual device and dtype (handles device_map="auto")
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        inputs = {k: v.to(model_device, model_dtype) for k, v in inputs.items()}
        
        with torch.no_grad():
            predicted_ids = self.model.generate(**inputs, return_timestamps=True)
            
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=False)[0]
        
        import re
        import datetime
        
        # Regex aggiornata per supportare timestamp interi o decimali (es. <|0|> o <|0.00|>)
        pattern = r'<\|(\d+(?:\.\d+)?)\|>([^<]*)'
        matches = re.findall(pattern, transcription)
        
        def format_time(t):
            td = datetime.timedelta(seconds=float(t))
            return f"{td.seconds//3600:02d}:{(td.seconds%3600)//60:02d}:{td.seconds%60:02d},{td.microseconds//1000:03d}"
        
        with open(output_srt_path, 'w') as f:
            if not matches:
                # Fallback se i timestamp non vengono rilevati
                logger.warning("Nessun timestamp rilevato nella trascrizione. Creazione SRT con testo unico.")
                f.write("1\n")
                f.write(f"{format_time(0)} --> {format_time(audio_duration)}\n")
                # Pulizia del testo da eventuali tag speciali residui
                clean_text = re.sub(r'<\|[^|]+\|>', '', transcription).strip()
                f.write(f"{clean_text}\n\n")
            else:
                for i, (start, text) in enumerate(matches):
                    start_time = float(start)
                    if i + 1 < len(matches):
                        end_time = float(matches[i+1][0])
                    else:
                        # Per l'ultimo segmento, usa la durata totale dell'audio o start + 2.0 se più corto
                        end_time = max(start_time + 2.0, audio_duration)
                    
                    f.write(f"{i+1}\n")
                    f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
                    f.write(f"{text.strip()}\n\n")
        
        logger.info(f"Sottotitoli salvati in {output_srt_path}")
        return output_srt_path

    def get_capabilities(self):
        return {"type": "speech", "model": "whisper"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb"), "backend": self.model_info.get("backend")}

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
