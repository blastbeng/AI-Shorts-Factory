class BaseWorker:
    def __init__(self, name="base_worker"):
        self.name = name

    def run(self):
        print(f"[{self.name}] Worker avviato.")

    def process_job(self, job_id):
        # TODO: implementare logica di processamento job
        print(f"[{self.name}] Processo job {job_id}")
