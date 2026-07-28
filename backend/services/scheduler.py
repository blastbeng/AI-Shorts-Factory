import threading
import time
from datetime import datetime, timedelta
from backend.database.session import SessionLocal
from backend.database.models import Job
from backend.services.logger import logger

class AutoScheduler:
    _instance = None
    _lock = threading.Lock()
    _running = False
    _thread = None
    _interval_minutes = 60

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def start(self, interval_minutes=60):
        self._interval_minutes = interval_minutes
        if self._running:
            logger.warning("Scheduler già in esecuzione.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._schedule_loop, daemon=True)
        self._thread.start()
        logger.info(f"AutoScheduler avviato (intervallo: {self._interval_minutes} min).")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("AutoScheduler arrestato.")

    def _schedule_loop(self):
        while self._running:
            try:
                db = SessionLocal()
                
                # Global check: do not create new jobs if any job is already running or pending
                active_jobs_count = db.query(Job).filter(Job.status.in_(["pending", "running"])).count()
                if active_jobs_count > 0:
                    db.close()
                    time.sleep(60)
                    continue

                jobs_created_this_cycle = 0
                for _ in range(3):
                    new_job = Job(status="pending")
                    db.add(new_job)
                    db.commit()
                    logger.info(f"[Scheduler] Nuovo job creato automaticamente (ID: {new_job.id})")
                    jobs_created_this_cycle += 1
                
                db.close()
                time.sleep(60)
            except Exception as e:
                logger.error(f"[Scheduler] Errore: {e}")
                time.sleep(60)

auto_scheduler = AutoScheduler()
