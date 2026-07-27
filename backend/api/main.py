from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from backend.database.session import Base, engine, get_db, SessionLocal
from backend.database.models import GenerationProfile, Job
from backend.gpu_manager.manager import GPUManager
from backend.domain.pipeline import PipelineOrchestrator
from backend.services.logger import logger

app = FastAPI(title="AI Shorts Factory")
Base.metadata.create_all(bind=engine)
logger.info("Avvio dell'applicazione AI Shorts Factory")

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

def run_pipeline_job(job_id: int, profile_id: int):
    db = SessionLocal()
    try:
        profile = db.query(GenerationProfile).filter(GenerationProfile.id == profile_id).first()
        orchestrator = PipelineOrchestrator(job_id, profile, db)
        orchestrator.run()

        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "completed"
            db.commit()
    except Exception as e:
        logger.error(f"Errore nel job {job_id}: {e}")
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "failed"
            db.commit()
    finally:
        db.close()

@app.post("/jobs/{profile_id}")
def start_job(profile_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    profile = db.query(GenerationProfile).filter(GenerationProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    job = Job(status="running", profile_id=profile_id)
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(run_pipeline_job, job.id, profile.id)

    return {"job_id": job.id, "status": "running", "message": "Job avviato in background"}
