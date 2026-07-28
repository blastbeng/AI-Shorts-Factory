import os
import yaml
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import LTXVideoImg2VideoPipeline
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger
import imageio

class LtxProvider(BaseAIProvider):
    def __init__(self):
        with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
            self.models_config = yaml.safe_load(f)
        self.model_info = self.models_config.get("video", {}).get("ltx_video", {})
        self.gm = GPUManager()
        self.pipeline = None

    def install_status(self):
        return self.model_info.get("status", "not_installed")

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, prompts: list, output_path: str, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Modello LTX Video non installato.")
            
        job_id = kwargs.get("job_id")
        image_path = kwargs.get("image_path")
        
        preferred_backend = self.model_info.get("backend")
        gpu = self.gm.get_gpu_for_task("video_generation", self.get_gpu_requirements().get("vram_required_gb", 0), preferred_backend=preferred_backend)
        if not gpu:
            logger.warning("Nessuna GPU con VRAM sufficiente per LTX Video. Uso GPU con offload su RAM.")
            gpu = self.gm.get_gpu_for_task_ignore_vram("video_generation", preferred_backend=preferred_backend)
            if not gpu:
                raise RuntimeError("Nessuna GPU assegnata per la video generation.")
            use_cpu_offload = True
        else:
            use_cpu_offload = False
            
        device = self.gm.get_device_string(gpu['id'], preferred_backend=self.model_info.get("backend"))
        
        if self.pipeline is None:
            logger.info("Caricamento pipeline LTX Video (Img2Video)...")
            model_path = self.model_info.get("path")
            try:
                self.pipeline = LTXVideoImg2VideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
                self.pipeline.enable_attention_slicing()
                if use_cpu_offload:
                    self.pipeline.enable_model_cpu_offload(device=device)
                else:
                    self.pipeline.to(device)
            except Exception as e:
                logger.exception(f"Errore nel caricamento del modello su GPU. Fallback con offload su RAM.")
                if self.pipeline is not None:
                    del self.pipeline
                    self.pipeline = None
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                available_ram = self.gm.get_available_system_ram_gb()
                if available_ram < 8.0:
                    raise RuntimeError(f"RAM di sistema insufficiente ({available_ram:.2f}GB) per il fallback su CPU. Operazione annullata per evitare il blocco del sistema.")
                
                logger.warning(f"RAM disponibile: {available_ram:.2f}GB. Uso sequential CPU offload per evitare OOM.")
                self.pipeline = LTXVideoImg2VideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_sequential_cpu_offload(device=device)

        import gc
        import os
        
        temp_clips = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Pulizia VRAM prima della generazione clip {i+1}/{len(prompts)}...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            logger.info(f"Generazione clip {i+1}/{len(prompts)} per prompt: {prompt}")
            
            # Determine the conditioning image for this clip
            if i == 0 and image_path and os.path.exists(image_path):
                # Load and resize the initial Flux image to match video dimensions (320x576)
                init_image = Image.open(image_path).convert("RGB")
                init_image = init_image.resize((320, 576), Image.LANCZOS)
                current_image = init_image
            elif temp_clips:
                # Extract last frame of the previous clip
                prev_cap = cv2.VideoCapture(temp_clips[-1])
                total_frames = int(prev_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                prev_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
                ret, frame_bgr = prev_cap.read()
                prev_cap.release()
                if ret:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    current_image = Image.fromarray(frame_rgb)
                else:
                    logger.warning("Impossibile estrarre l'ultimo frame. Uso immagine nera.")
                    current_image = Image.new("RGB", (320, 576), color="black")
            else:
                logger.warning("Nessuna immagine iniziale fornita. Uso immagine nera.")
                current_image = Image.new("RGB", (320, 576), color="black")

            def progress_callback(pipe, step, timestep, callback_kwargs):
                logger.info(f"LTX generation progress (clip {i+1}): step {step + 1}/25")
                if job_id:
                    from backend.services.progress_tracker import ProgressTracker
                    ProgressTracker().update(job_id, "video_generation", step + 1, 25, f"Generazione clip {i+1}/{len(prompts)}: step {step + 1}/25")
                return callback_kwargs

            video = self.pipeline(
                image=current_image,
                prompt=prompt,
                num_inference_steps=25,
                height=576,
                width=320,
                num_frames=49,
                guidance_scale=4.0,
                callback_on_step_end=progress_callback,
                callback_on_step_end_tensor_inputs=[]
            ).frames[0]

            if isinstance(video, torch.Tensor):
                video = video.cpu().numpy()

            # Converti da [0, 1] float32 a [0, 255] uint8
            video = (video * 255).round().astype("uint8")

            temp_clip_path = output_path.replace(".mp4", f"_clip_{i}.mp4")
            frame_height, frame_width = video.shape[1], video.shape[2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_clip_path, fourcc, 24.0, (frame_width, frame_height))
            for frame in video:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            video_writer.release()
            temp_clips.append(temp_clip_path)
            logger.info(f"Clip {i+1} salvata in {temp_clip_path}")

        logger.info(f"Concatenazione di {len(temp_clips)} clip in {output_path}...")
        cap = cv2.VideoCapture(temp_clips[0])
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        final_writer = cv2.VideoWriter(output_path, fourcc, 24.0, (frame_width, frame_height))
        for temp_clip_path in temp_clips:
            cap = cv2.VideoCapture(temp_clip_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                final_writer.write(frame)
            cap.release()
        final_writer.release()
        
        for temp_clip_path in temp_clips:
            if os.path.exists(temp_clip_path):
                os.remove(temp_clip_path)
                
        logger.info(f"Video finale salvato in {output_path}")
        return output_path

    def get_capabilities(self):
        return {"type": "video", "model": "ltx_video"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": self.model_info.get("vram_required_gb"), "backend": self.model_info.get("backend")}

    def cleanup(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force glibc to release unused memory back to the OS
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
