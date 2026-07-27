from backend.database.models import PipelineStage, Video
from backend.services.logger import logger
from backend.services.quality_scorer import QualityScorer
from backend.ai_providers.wan_provider import WanProvider
from backend.ai_providers.kokoro_provider import KokoroProvider
from backend.ai_providers.mmaudio_provider import MMAudioProvider
from backend.ai_providers.flux_provider import FluxProvider
import os
import subprocess

class PipelineOrchestrator:
    def __init__(self, job_id, profile, db):
        self.job_id = job_id
        self.profile = profile
        self.db = db
        self.scorer = QualityScorer()
        self.wan = WanProvider()
        self.kokoro = KokoroProvider()
        self.mmaudio = MMAudioProvider()
        self.flux = FluxProvider()
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
        os.makedirs("output", exist_ok=True)

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
        script = ""
        voice_path = ""
        image_path = ""
        video_path = ""
        audio_path = ""
        final_video_path = "output/final_video.mp4"

        for stage in self.stages:
            self._update_stage(stage, "running")
            try:
                if stage == "topic_generation":
                    self._update_stage(stage, "completed", self.profile.topic)

                elif stage == "script_generation":
                    script = f"Script generato per il topic: {self.profile.topic}. Durata target: {self.profile.duration_seconds} secondi."
                    self._update_stage(stage, "completed", script)

                elif stage == "voice_generation":
                    voice_path = f"output/voice_{self.job_id}.wav"
                    self.kokoro.generate(script, voice_path)
                    self._update_stage(stage, "completed", voice_path)

                elif stage == "storyboard_creation":
                    storyboard = f"Storyboard per: {self.profile.topic}"
                    self._update_stage(stage, "completed", storyboard)

                elif stage == "image_generation":
                    image_path = f"output/image_{self.job_id}.png"
                    self.flux.generate(self.profile.topic, image_path)
                    self._update_stage(stage, "completed", image_path)

                elif stage == "video_generation":
                    video_path = f"output/video_{self.job_id}.mp4"
                    self.wan.generate(self.profile.topic, video_path)
                    self._update_stage(stage, "completed", video_path)

                elif stage == "audio_generation":
                    audio_path = f"output/audio_{self.job_id}.wav"
                    self.mmaudio.generate(self.profile.topic, audio_path)
                    self._update_stage(stage, "completed", audio_path)

                elif stage == "video_assembly":
                    cmd = [
                        "ffmpeg", "-y", "-i", video_path, "-i", voice_path,
                        "-i", audio_path, "-c:v", "copy", "-c:a", "aac",
                        final_video_path
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    self._update_stage(stage, "completed", final_video_path)

                elif stage == "quality_scoring":
                    score = self.scorer.score(final_video_path)
                    self._update_stage(stage, "completed", str(score))

                elif stage == "storage":
                    video_record = Video(
                        job_id=self.job_id,
                        file_path=final_video_path,
                        quality_score=float(self.scorer.score(final_video_path)),
                        approved=False
                    )
                    self.db.add(video_record)
                    self.db.commit()
                    self._update_stage(stage, "completed", final_video_path)

                elif stage == "dashboard_review":
                    self._update_stage(stage, "completed", "In attesa di revisione manuale")

            except Exception as e:
                self._update_stage(stage, "failed", str(e))
                logger.error(f"[Job {self.job_id}] Fallimento stage {stage}: {e}")
                return False
        return True
