import threading
import time
from backend.database.session import SessionLocal
from backend.database.models import Job, GenerationProfile
from backend.domain.pipeline import PipelineOrchestrator
from backend.services.logger import logger

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
                    logger.info(f"[Worker] Avvio elaborazione job {job.id}")
                    
                    profile = db.query(GenerationProfile).filter(
                        GenerationProfile.id == job.profile_id
                    ).first()
                    
                    if not profile:
                        job.status = "failed"
                        db.commit()
                        continue
                    
                    try:
                        orchestrator = PipelineOrchestrator(job.id, profile, db)
                        result = orchestrator.run()
                        if result == "interrupted":
                            # Non sovrascrivere lo stato, è già "interrupted"
                            pass
                        elif result == "waiting_for_review":
                            job.status = "waiting_for_review"
                        elif result:
                            job.status = "completed"
                        else:
                            job.status = "failed"
                    except Exception as e:
                        logger.error(f"[Worker] Errore job {job.id}: {e}")
                        job.status = "failed"
                    db.commit()
                
                db.close()
                time.sleep(2)
            except Exception as e:
                logger.error(f"[Worker] Errore nel poll loop: {e}")
                time.sleep(5)

job_worker = JobWorker()
