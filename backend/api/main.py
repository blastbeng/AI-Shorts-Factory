import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from dotenv import load_dotenv
load_dotenv() # Carica le variabili d'ambiente dal file .env

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pydantic import BaseModel
from backend.database.session import Base, engine, get_db, SessionLocal
from backend.database.models import GenerationProfile, Job, Video, PipelineStage
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger, log_buffer
from backend.services.social.tiktok_provider import TikTokProvider
from backend.services.social.youtube_provider import YouTubeProvider
from backend.services.social.instagram_provider import InstagramProvider
from backend.services.social.facebook_provider import FacebookProvider
from backend.workers.job_worker import job_worker
from backend.services.scheduler import auto_scheduler
from backend.services.config_validator import ConfigValidator
from backend.services.subprocess_manager import SubprocessManager
import os
import yaml
import platform
import sys
import importlib.metadata

def log_environment_info():
    logger.info("=== Informazioni Ambiente ===")
    logger.info(f"Python: {sys.version.split(' ')[0]}")
    logger.info(f"OS: {platform.platform()}")
    
    packages = [
        "fastapi", "uvicorn", "sqlalchemy", "pyyaml", "torch", "torchvision", 
        "torchaudio", "transformers", "diffusers", "kokoro", "spacy", 
        "huggingface_hub", "numpy", "sentencepiece", "tiktoken", "llama-cpp-python"
    ]
    
    for pkg in packages:
        try:
            version = importlib.metadata.version(pkg)
            logger.info(f"{pkg}: {version}")
        except importlib.metadata.PackageNotFoundError:
            logger.warning(f"{pkg}: non installato")
    logger.info("===============================")

app = FastAPI(title="AI Shorts Factory")

# Configurazione CORS per permettere al frontend di comunicare con il backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permette tutte le origini per lo sviluppo locale
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

# Reset running jobs to interrupted on startup
db_startup = SessionLocal()
running_jobs = db_startup.query(Job).filter(Job.status == "running").all()
for job in running_jobs:
    job.status = "interrupted"
db_startup.commit()
db_startup.close()

logger.info("Avvio dell'applicazione AI Shorts Factory")
log_environment_info()

# Validazione configurazione
ConfigValidator.validate_and_exit()

# Inizializzazione del contesto CUDA nel thread principale per evitare deadlock con ROCm/PyTorch nei thread secondari
try:
    import torch
    if torch.cuda.is_available():
        logger.info("Inizializzazione contesto CUDA nel thread principale...")
        torch.cuda.init()
        dummy = torch.tensor([1.0], device="cuda")
        del dummy
        torch.cuda.empty_cache()
        logger.info("Contesto CUDA inizializzato correttamente.")
except Exception as e:
    logger.error(f"Errore nell'inizializzazione del contesto CUDA: {e}")

# Avvia il worker in background per processare i job avviati manualmente
job_worker.start()
logger.info("AutoScheduler non avviato automaticamente. Generazione manuale abilitata.")

# Monta la directory output per servire i video generati
os.makedirs("output", exist_ok=True)
app.mount("/output", StaticFiles(directory="output"), name="output")

class ProfileCreate(BaseModel):
    name: str
    genre: str = "random"
    custom_prompt: str = ""
    language: str = "italian"
    style: str = "default"
    duration_seconds: int = 30

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/gpus")
def gpus():
    gm = GPUManager()
    gpus_info = []
    for gpu in gm.get_gpus():
        vram_info = gm.monitor_vram(gpu["id"])
        gpus_info.append({
            "id": gpu["id"],
            "name": gpu["name"],
            "vram_total_gb": vram_info["vram_total_gb"] if vram_info else gpu["vram_gb"],
            "vram_used_gb": vram_info["vram_used_gb"] if vram_info else 0,
            "vram_free_gb": vram_info["vram_free_gb"] if vram_info else gpu["vram_gb"],
            "gpu_utilization": vram_info["gpu_utilization"] if vram_info else 0,
            "backends": gpu.get("backends", []),
            "assigned_tasks": gpu.get("assigned_tasks", [])
        })
    return gpus_info

@app.get("/monitor")
def monitor():
    gm = GPUManager()
    gpus = gm.get_gpus()
    logger.info("Richiesta monitoraggio ricevuta")
    return {
        "system_status": "operational",
        "gpus": gpus
    }

@app.post("/profiles/")
def create_profile(profile: ProfileCreate, db: Session = Depends(get_db)):
    db_profile = GenerationProfile(**profile.dict())
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    return db_profile

@app.get("/profiles/")
def get_profiles(db: Session = Depends(get_db)):
    return db.query(GenerationProfile).all()

@app.post("/jobs/{profile_id}")
def start_job(profile_id: int, db: Session = Depends(get_db)):
    profile = db.query(GenerationProfile).filter(GenerationProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    job = Job(status="pending", profile_id=profile_id)
    db.add(job)
    db.commit()
    db.refresh(job)

    return {"job_id": job.id, "status": "pending", "message": "Job aggiunto alla coda. Il worker lo processerà a breve."}

@app.post("/jobs/{job_id}/interrupt")
def interrupt_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ["running", "pending"]:
        raise HTTPException(status_code=400, detail=f"Impossibile interrompere un job con stato {job.status}")
    job.status = "interrupted"
    db.commit()
    db.refresh(job)
    return {"status": "interrupted", "job_id": job_id}

@app.get("/videos/")
def get_videos(db: Session = Depends(get_db)):
    return db.query(Video).all()

@app.get("/videos/{video_id}")
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video

@app.put("/videos/{video_id}/approve")
def approve_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    video.approved = True
    job = db.query(Job).filter(Job.id == video.job_id).first()
    if job:
        job.status = "completed"
    db.commit()
    db.refresh(video)
    return video

@app.put("/videos/{video_id}/reject")
def reject_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    video.approved = False
    job = db.query(Job).filter(Job.id == video.job_id).first()
    if job:
        job.status = "rejected"
    db.commit()
    db.refresh(video)
    return video

@app.delete("/videos/{video_id}")
def delete_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Elimina il file fisico se esiste
    if video.file_path and os.path.exists(video.file_path):
        os.remove(video.file_path)
    
    # Opzionale: elimina anche il job e i suoi stadi se non ha altri video associati
    video_count = db.query(Video).filter(Video.job_id == video.job_id).count()
    if video_count == 1:  # Se questo è l'unico video
        # Elimina gli stadi della pipeline associati al job
        db.query(PipelineStage).filter(PipelineStage.job_id == video.job_id).delete()
        job = db.query(Job).filter(Job.id == video.job_id).first()
        if job:
            db.delete(job)
    
    db.delete(video)
    db.commit()
    return {"status": "deleted", "video_id": video_id}

@app.post("/videos/{video_id}/publish/{platform}")
def publish_video(video_id: int, platform: str, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not video.approved:
        raise HTTPException(status_code=400, detail="Il video deve essere approvato prima della pubblicazione.")
    if not video.file_path or not os.path.exists(video.file_path):
        raise HTTPException(status_code=404, detail="File video non trovato sul filesystem.")
    
    metadata = {
        "title": f"AI Generated Video {video.id}",
        "description": "Generated by AI Shorts Factory",
        "tags": ["ai", "shorts", "auto"]
    }

    if platform == "tiktok":
        provider = TikTokProvider()
    elif platform == "youtube":
        provider = YouTubeProvider()
    elif platform == "instagram":
        provider = InstagramProvider()
    elif platform == "facebook":
        provider = FacebookProvider()
    else:
        raise HTTPException(status_code=400, detail="Piattaforma non supportata")
    
    try:
        provider.authenticate()
        result = provider.upload_video(video.file_path, metadata)
        video.published = True
        db.commit()
        db.refresh(video)
        return {"status": "publish_started", "platform": platform, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}")
def get_job_details(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    stages = db.query(PipelineStage).filter(PipelineStage.job_id == job_id).order_by(PipelineStage.id).all()
    return {
        "job_id": job.id,
        "status": job.status,
        "profile_id": job.profile_id,
        "stages": [{"name": s.stage_name, "status": s.status, "result": s.result, "created_at": s.created_at, "updated_at": s.updated_at} for s in stages]
    }

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    total_videos = db.query(Video).count()
    approved_videos = db.query(Video).filter(Video.approved == True).count()
    published_videos = db.query(Video).filter(Video.published == True).count()
    total_jobs = db.query(Job).count()
    return {
        "total_videos": total_videos,
        "approved_videos": approved_videos,
        "published_videos": published_videos,
        "total_jobs": total_jobs
    }

@app.get("/models/status")
def models_status():
    with open(os.getenv("MODELS_CONFIG_PATH", "configs/models.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    models = []
    # LLM
    llm_provider = os.getenv("LLM_PROVIDER", "llama_cpp")
    if llm_provider == "llama_cpp":
        models.append({"name": "LLM (llama.cpp)", "status": "installed" if os.path.exists(os.getenv("LLAMA_CPP_MODEL_PATH", "")) else "not_installed"})
    else:
        models.append({"name": f"LLM ({llm_provider})", "status": "installed" if os.getenv("OPENAI_API_KEY") or os.getenv("OLLAMA_API_BASE") else "not_installed"})

    # Video, Audio, Voice, Image, Speech
    for category in ["video", "audio", "voice", "image", "speech"]:
        for name, info in config.get(category, {}).items():
            models.append({"name": f"{category.capitalize()} ({name})", "status": info.get("status", "not_installed")})
            
    return {"models": models}

@app.get("/logs")
def get_logs():
    return {"logs": list(log_buffer)}

@app.get("/jobs/")
def get_all_jobs(db: Session = Depends(get_db)):
    jobs = db.query(Job).order_by(Job.created_at.desc()).all()
    result = []
    for j in jobs:
        stages = db.query(PipelineStage).filter(PipelineStage.job_id == j.id).all()
        total_stages = len(stages)
        # Considera 'completed' e 'waiting_for_review' come completati per la barra di progresso
        completed_stages = len([s for s in stages if s.status in ("completed", "waiting_for_review")])
        result.append({
            "id": j.id, 
            "status": j.status, 
            "profile_id": j.profile_id,
            "progress": {
                "completed": completed_stages,
                "total": total_stages
            }
        })
    return result

@app.get("/scheduler/status")
def scheduler_status():
    return {
        "worker_running": job_worker._running,
        "scheduler_running": auto_scheduler._running,
        "scheduler_interval_minutes": auto_scheduler._interval_minutes
    }

@app.post("/scheduler/start")
def start_scheduler(interval: int = 60):
    auto_scheduler.start(interval_minutes=interval)
    return {"status": "started", "interval_minutes": interval}

@app.post("/scheduler/stop")
def stop_scheduler():
    auto_scheduler.stop()
    return {"status": "stopped"}

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Arresto dell'applicazione in corso...")
    SubprocessManager.kill_all()
    job_worker.stop()
    auto_scheduler.stop()
    logger.info("Worker, Scheduler e Processi AI arrestati correttamente.")
