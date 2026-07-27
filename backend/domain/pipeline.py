from backend.database.models import PipelineStage
from backend.services.logger import logger
from backend.services.quality_scorer import QualityScorer
import os

class PipelineOrchestrator:
    def __init__(self, job_id, profile, db):
        self.job_id = job_id
        self.profile = profile
        self.db = db
        self.scorer = QualityScorer()
        self.stages = [
            "topic_generation",
            "script_generation",
            "voice_generation",
            "storyboard_creation",
            "image_generation",
            "video_generation",
            "audio_generation",
            "video_assembly",
            "quality_scoring",
            "storage",
            "dashboard_review"
        ]

    def _update_stage(self, stage_name, status, result=None):
        stage = PipelineStage(
            job_id=self.job_id,
            stage_name=stage_name,
            status=status,
            result=result
        )
        self.db.add(stage)
        self.db.commit()
        logger.info(f"[Job {self.job_id}] Stage {stage_name}: {status}")

    def run(self):
        for stage in self.stages:
            self._update_stage(stage, "running")
            try:
                if stage == "topic_generation":
                    self._update_stage(stage, "completed", self.profile.topic)
                elif stage == "script_generation":
                    script = f"Script generato per il topic: {self.profile.topic}"
                    self._update_stage(stage, "completed", script)
                elif stage == "quality_scoring":
                    score = self.scorer.score("output/final_video.mp4")
                    self._update_stage(stage, "completed", str(score))
                else:
                    self._update_stage(stage, "completed", "Placeholder logic executed")
            except Exception as e:
                self._update_stage(stage, "failed", str(e))
                logger.error(f"[Job {self.job_id}] Fallimento stage {stage}: {e}")
                return False
        return True
