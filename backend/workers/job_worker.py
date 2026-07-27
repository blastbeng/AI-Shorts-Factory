import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import threading
import time
import multiprocessing
from backend.database.session import SessionLocal
from backend.database.models import Job, GenerationProfile
from backend.domain.pipeline import PipelineOrchestrator
from backend.services.logger import logger

def _run_job_process(job_id: int, profile_id: int):
    """Esegue un singolo job in un processo separato per isolare il contesto CUDA."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return
        
        profile = db.query(GenerationProfile).filter(
            GenerationProfile.id == profile_id
        ).first()
        
        if not profile:
            job.status = "failed"
            db.commit()
            return
        
        try:
            orchestrator = PipelineOrchestrator(job.id, profile, db)
            result = orchestrator.run()
            if result == "interrupted":
                pass
            elif result == "waiting_for_review":
                job.status = "waiting_for_review"
            elif result:
                job.status = "completed"
            else:
                job.status = "failed"
        except Exception as e:
            logger.error(f"[Worker-Process] Errore job {job.id}: {e}")
            job.status = "failed"
        db.commit()
    except Exception as e:
        logger.error(f"[Worker-Process] Errore fatale per job {job_id}: {e}")
    finally:
        db.close()

class JobWorker:
    _instance = None
    _lock = threading.Lock()
    _running = False
    _thread = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def start(self):
        if self._running:
            logger.warning("JobWorker già in esecuzione.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("JobWorker avviato.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("JobWorker arrestato.")

    def _poll_loop(self):
        while self._running:
            try:
                db = SessionLocal()
                pending_jobs = db.query(Job).filter(Job.status == "pending").all()
                for job in pending_jobs:
                    job.status = "running"
                    db.commit()
                    logger.info(f"[Worker] Avvio elaborazione job {job.id} in subprocess")
                    
                    # Avvia il job in un processo separato per isolare CUDA/PyTorch
                    p = multiprocessing.Process(
                        target=_run_job_process,
                        args=(job.id, job.profile_id)
                    )
                    p.start()
                    p.join() # Attende che il processo termini
                    
                    if p.exitcode != 0:
                        # Se il processo è crashato, assicurati che il job sia marcato come fallito
                        current_job = db.query(Job).filter(Job.id == job.id).first()
                        if current_job and current_job.status == "running":
                            current_job.status = "failed"
                            db.commit()
                            logger.error(f"[Worker] Il subprocess per il job {job.id} è terminato con codice {p.exitcode}")
                
                db.close()
                time.sleep(2)
            except Exception as e:
                logger.error(f"[Worker] Errore nel poll loop: {e}")
                time.sleep(5)

job_worker = JobWorker()
