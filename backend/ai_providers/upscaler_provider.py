import os
import torch
import numpy as np
import imageio
from backend.ai_providers.base_provider import BaseAIProvider
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger

class UpscalerProvider(BaseAIProvider):
    def __init__(self):
        self.gm = GPUManager()
        self.model = None

    def install_status(self):
        return "installed"

    def health_check(self):
        return self.install_status() == "installed"

    def generate(self, video_path: str, output_path: str, *args, **kwargs):
        if not self.health_check():
            raise RuntimeError("Upscaler non installato.")
            
        gpu = self.gm.get_gpu_for_task("video_upscaling", 4)
        if not gpu:
            logger.warning("Nessuna GPU con VRAM sufficiente per Upscaler. Uso GPU con offload su RAM.")
            gpu = self.gm.get_gpu_for_task_ignore_vram("video_upscaling")
            if not gpu:
                raise RuntimeError("Nessuna GPU assegnata per l'upscaling.")
            use_cpu_offload = True
        else:
            use_cpu_offload = False
            
        device = self.gm.get_device_string(gpu['id'])
        
        if self.model is None:
            logger.info("Caricamento modello Real-ESRGAN...")
            import sys
            import torchvision.transforms.functional as F
            sys.modules['torchvision.transforms.functional_tensor'] = F
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            self.model = RealESRGANer(
                scale=2,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                model=model,
                device=device,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=not use_cpu_offload
            )
        
        logger.info(f"Upscaling video: {video_path}")
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data()['fps']
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8, macro_block_size=1)
        
        for frame in reader:
            frame = np.array(frame)
            output, _ = self.model.enhance(frame, outscale=2)
            writer.append_data(output)
            
        reader.close()
        writer.close()
        logger.info(f"Video upscalato salvato in {output_path}")
        return output_path

    def get_capabilities(self):
        return {"type": "video", "model": "realesrgan"}

    def get_gpu_requirements(self):
        return {"vram_required_gb": 4, "backend": "cuda"}

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
