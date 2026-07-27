from backend.database.models import PipelineStage, Video
from backend.services.logger import logger
from backend.services.quality_scorer import QualityScorer
from backend.ai_providers.wan_provider import WanProvider
from backend.ai_providers.kokoro_provider import KokoroProvider
from backend.ai_providers.mmaudio_provider import MMAudioProvider
from backend.ai_providers.flux_provider import FluxProvider
from backend.ai_providers.llm_provider import LLMProvider
from backend.media.ffmpeg_utils import FFmpegUtils
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
        self.llm = LLMProvider()
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
        final_video_path = f"output/final_video_{self.job_id}.mp4"

        # Recupera gli stage già completati per supportare il resume
        completed_stages = self.db.query(PipelineStage).filter(
            PipelineStage.job_id == self.job_id,
            PipelineStage.status == "completed"
        ).all()
        completed_names = {s.stage_name for s in completed_stages}

        # Recupera i risultati degli stage completati
        for s in completed_stages:
            if s.stage_name == "script_generation":
                script = s.result or ""
            elif s.stage_name == "voice_generation":
                voice_path = s.result or ""
            elif s.stage_name == "image_generation":
                image_path = s.result or ""
            elif s.stage_name == "video_generation":
                video_path = s.result or ""
            elif s.stage_name == "audio_generation":
                audio_path = s.result or ""

        for stage in self.stages:
            if stage in completed_names:
                logger.info(f"[Job {self.job_id}] Stage {stage} già completato, salto.")
                continue
            self._update_stage(stage, "running")
            try:
                if stage == "topic_generation":
                    # Usa l'LLM per espandere il topic
                    expanded_topic = self.llm.generate(f"Espandi questo topic per un video short: {self.profile.topic}", max_length=100)
                    self._update_stage(stage, "completed", expanded_topic)

                elif stage == "script_generation":
                    # Usa l'LLM per generare lo script
                    script_prompt = f"Scrivi uno script di {self.profile.duration_seconds} secondi per un video su: {self.profile.topic}"
                    script = self.llm.generate(script_prompt, max_length=300)
                    self._update_stage(stage, "completed", script)

                elif stage == "voice_generation":
                    voice_path = f"output/voice_{self.job_id}.wav"
                    self.kokoro.generate(script, voice_path)
                    self._update_stage(stage, "completed", voice_path)

                elif stage == "storyboard_creation":
                    # Usa l'LLM per generare lo storyboard
                    storyboard_prompt = f"Crea uno storyboard di 3 scene per questo script: {script}"
                    storyboard = self.llm.generate(storyboard_prompt, max_length=200)
                    self._update_stage(stage, "completed", storyboard)

                elif stage == "image_generation":
                    image_path = f"output/image_{self.job_id}.png"
                    # Usa lo storyboard per un prompt più dettagliato
                    image_prompt = f"Immagine di alta qualità per un video su: {self.profile.topic}. Scene: {storyboard}"
                    self.flux.generate(image_prompt, image_path)
                    self._update_stage(stage, "completed", image_path)

                elif stage == "video_generation":
                    video_path = f"output/video_{self.job_id}.mp4"
                    # Usa lo script per un prompt video più contestualizzato
                    video_prompt = f"Video short verticale basato su questo script: {script}"
                    self.wan.generate(video_prompt, video_path)
                    self._update_stage(stage, "completed", video_path)

                elif stage == "audio_generation":
                    audio_path = f"output/audio_{self.job_id}.wav"
                    # Usa lo storyboard per generare effetti sonori contestualizzati
                    audio_prompt = f"Effetti sonori e musica di sottofondo per queste scene: {storyboard}"
                    self.mmaudio.generate(audio_prompt, audio_path)
                    self._update_stage(stage, "completed", audio_path)

                elif stage == "video_assembly":
                    success = FFmpegUtils.assemble_video(video_path, voice_path, audio_path, final_video_path)
                    if not success:
                        raise RuntimeError("Assemblaggio video fallito.")
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
