from backend.services.logger import logger

class BaseWorker:
    def __init__(self, name="base_worker"):
        self.name = name

    def run(self):
        logger.info(f"[{self.name}] Worker avviato.")

    def process_job(self, job_id):
        logger.info(f"[{self.name}] Processo job {job_id}")
        # Logica di processamento delegata ai servizi specifici
        return True
