class ProgressTracker:
    _instance = None
    _progress = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def update(self, job_id, stage, current_step, total_steps, message):
        self._progress[job_id] = {
            "stage": stage,
            "current_step": current_step,
            "total_steps": total_steps,
            "message": message
        }

    def get(self, job_id):
        return self._progress.get(job_id, None)

    def clear(self, job_id):
        if job_id in self._progress:
            del self._progress[job_id]
