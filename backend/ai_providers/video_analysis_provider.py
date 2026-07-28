import os
import yaml
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger

class VideoAnalysisProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("image", {}).get("qwen_image", {})
        self.gm = GPUManager()
        self.model = None
        self.processor = None

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, video_path: str, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello Qwen2-VL non installato per l'analisi video.")
        
        if not os.path.exists(video_path):
            raise RuntimeError(f"File video non trovato: {video_path}")

        preferred_backend = self.model_info.get("backend")
        gpu = self.gm.get_gpu_for_task("image_generation", self.get_gpu_requirements().get("vram_required_gb", 0), preferred_backend=preferred_backend)
        if not gpu:
            gpu = self.gm.get_gpu_for_task_ignore_vram("image_generation", preferred_backend=preferred_backend)
            if not gpu:
                raise RuntimeError("Nessuna GPU assegnata per l'analisi video.")
        
        device = self.gm.get_device_string(gpu['id'], preferred_backend=preferred_backend)

        if self.model is None:
            logger.info("Caricamento modello Qwen2-VL per analisi video...")
            model_path = self.model_info.get("path")
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16
            ).to(device)

        # Extract key frames from the video (one frame every 4 seconds, max 8 frames)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            cap.release()
            raise RuntimeError("Impossibile leggere il video per l'analisi.")

        frame_interval = max(1, int(fps * 4))
        frames = []
        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            if len(frames) >= 8:
                break
        cap.release()

        if not frames:
            raise RuntimeError("Nessun frame estratto dal video.")

        logger.info(f"Estratti {len(frames)} frame per l'analisi video.")

        # Analyze each frame
        descriptions = []
        for i, frame_img in enumerate(frames):
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"This is frame {i+1} from a short video. Describe what you see in detail: characters, actions, setting, lighting, and mood. Be concise but specific."}
            ]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=text, images=[frame_img], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=150)
            
            # Decode only the new tokens (skip the input)
            input_len = inputs["input_ids"].shape[1]
            response = self.processor.decode(output[0][input_len:], skip_special_tokens=True).strip()
            descriptions.append(f"Frame {i+1}: {response}")
            logger.info(f"Frame {i+1} analizzato: {response[:100]}...")

        video_description = "\n".join(descriptions)
        logger.info(f"Analisi video completata. Descrizione: {video_description[:200]}...")
        return video_description

    def get_capabilities(self):
        return {"type": "video_analysis", "model": "qwen2_vl"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb", 10), "backend": self.model_info.get("backend")}

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
