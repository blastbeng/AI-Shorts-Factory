from dotenv import load_dotenv
load_dotenv() # Carica le variabili d'ambiente dal file .env

from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pydantic import BaseModel
from backend.database.session import Base, engine, get_db
from backend.database.models import GenerationProfile, Job, Video, PipelineStage
from backend.gpu_manager.manager import GPUManager
from backend.services.logger import logger
from backend.services.social.tiktok_provider import TikTokProvider
from backend.services.social.youtube_provider import YouTubeProvider
from backend.services.social.instagram_provider import InstagramProvider
from backend.services.social.facebook_provider import FacebookProvider
from backend.workers.job_worker import job_worker
from backend.services.scheduler import auto_scheduler
from backend.services.config_validator import ConfigValidator
import os

app = FastAPI(title="AI Shorts Factory")
Base.metadata.create_all(bind=engine)
logger.info("Avvio dell'applicazione AI Shorts Factory")

# Validazione configurazione
ConfigValidator.validate_and_exit()

# Avvia il worker in background per processare i job avviati manualmente
job_worker.start()
logger.info("AutoScheduler non avviato automaticamente. Generazione manuale abilitata.")

# Monta la directory output per servire i video generati
os.makedirs("output", exist_ok=True)
app.mount("/output", StaticFiles(directory="output"), name="output")

class ProfileCreate(BaseModel):
    name: str
    topic: str
    style: str = "default"
    duration_seconds: int = 30

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/gpus")
def gpus():
    gm = GPUManager()
    return gm.get_gpus()

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
    db.commit()
    db.refresh(video)
    return video

@app.put("/videos/{video_id}/reject")
def reject_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    video.approved = False
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
    
    # Elimina gli stadi della pipeline associati al job
    db.query(PipelineStage).filter(PipelineStage.job_id == video.job_id).delete()
    
    # Opzionale: elimina anche il job se non ha altri video associati
    job = db.query(Job).filter(Job.id == video.job_id).first()
    if job and len(job.videos) == 1:  # Se questo è l'unico video
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
        return {"status": "publish_started", "platform": platform, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}")
def get_job_details(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    stages = db.query(PipelineStage).filter(PipelineStage.job_id == job_id).all()
    return {
        "job_id": job.id,
        "status": job.status,
        "profile_id": job.profile_id,
        "stages": [{"name": s.stage_name, "status": s.status, "result": s.result} for s in stages]
    }

@app.get("/jobs/")
def get_all_jobs(db: Session = Depends(get_db)):
    jobs = db.query(Job).all()
    return [{"id": j.id, "status": j.status, "profile_id": j.profile_id} for j in jobs]

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
    job_worker.stop()
    auto_scheduler.stop()
    logger.info("Worker e Scheduler arrestati correttamente.")
