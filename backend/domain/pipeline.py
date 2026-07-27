from backend.database.models import PipelineStage, Video, Job
from backend.services.logger import logger
from backend.services.quality_scorer import QualityScorer
from backend.media.ffmpeg_utils import FFmpegUtils
import os
import random
import yaml

class PipelineOrchestrator:
    def __init__(self, job_id, profile, db):
        self.job_id = job_id
        self.profile = profile
        self.db = db
        self.scorer = QualityScorer()
        self.stages = [
            "topic_generation",
            "script_generation",
            "storyboard_creation",
            "voice_generation",
            "image_generation",
            "video_generation",
            "audio_generation",
            "video_assembly",
            "quality_scoring",
            "storage",
            "dashboard_review"
        ]
        os.makedirs("output", exist_ok=True)
        with open("configs/prompts_templates.yaml", "r") as f:
            self.templates = yaml.safe_load(f)

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

    def _is_interrupted(self):
        from backend.database.session import SessionLocal
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == self.job_id).first()
            return job.status == "interrupted"
        finally:
            db.close()

    def _generate_dummy_media(self, media_type, output_path):
        import subprocess
        if media_type == "audio":
            cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "1", "-q:a", "9", "-acodec", "pcm_s16le", output_path]
        elif media_type == "image":
            cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1080x1920", "-vframes", "1", output_path]
        elif media_type == "video":
            cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:r=24", "-t", "1", output_path]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.warning(f"Generato file dummy per {media_type}: {output_path}")

    def run(self):
        script = ""
        voice_path = ""
        image_path = ""
        video_path = ""
        audio_path = ""
        storyboard = ""
        expanded_topic = ""
        final_video_path = f"output/final_video_{self.job_id}.mp4"
        quality_score = 0.0

        # Recupera gli stage già completati per supportare il resume
        completed_stages = self.db.query(PipelineStage).filter(
            PipelineStage.job_id == self.job_id,
            PipelineStage.status == "completed"
        ).all()
        completed_names = {s.stage_name for s in completed_stages}

        # Recupera i risultati degli stage completati
        for s in completed_stages:
            if s.stage_name == "topic_generation":
                expanded_topic = s.result or ""
            elif s.stage_name == "script_generation":
                script = s.result or ""
            elif s.stage_name == "voice_generation":
                voice_path = s.result or ""
            elif s.stage_name == "storyboard_creation":
                storyboard = s.result or ""
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
            
            if self._is_interrupted():
                logger.info(f"[Job {self.job_id}] Job interrotto dall'utente. Arresto della pipeline.")
                return "interrupted"

            self._update_stage(stage, "running")
            try:
                if stage == "topic_generation":
                    from backend.ai_providers.llm_provider import LLMProvider
                    llm = LLMProvider()
                    
                    if self.profile.custom_prompt:
                        prompt = f"Espandi questo topic per un video short: {self.profile.custom_prompt}. Genera il testo in lingua: {self.profile.language}."
                    else:
                        genre = self.profile.genre
                        if not genre or genre == "random":
                            genre = random.choice(self.templates.get("genres", ["generale"]))
                        
                        instruction = self.templates.get("random_prompt_instruction", "Genera un'idea per un video di genere {genre}.").replace("{genre}", genre)
                        prompt = f"{instruction} Genera il testo in lingua: {self.profile.language}."
                    
                    expanded_topic = llm.generate(prompt, max_length=100, is_interrupted=self._is_interrupted)
                    if self._is_interrupted():
                        return "interrupted"
                    self._update_stage(stage, "completed", expanded_topic)

                elif stage == "script_generation":
                    from backend.ai_providers.llm_provider import LLMProvider
                    llm = LLMProvider()
                    script_prompt = f"Scrivi uno script di {self.profile.duration_seconds} secondi per un video su: {expanded_topic or self.profile.custom_prompt or self.profile.topic}. Lo script deve essere scritto in lingua: {self.profile.language}."
                    script = llm.generate(script_prompt, max_length=300, is_interrupted=self._is_interrupted)
                    if self._is_interrupted():
                        return "interrupted"
                    self._update_stage(stage, "completed", script)

                elif stage == "storyboard_creation":
                    from backend.ai_providers.llm_provider import LLMProvider
                    llm = LLMProvider()
                    storyboard_prompt = f"Crea uno storyboard di 3 scene per questo script: {script}. Lo storyboard deve essere in lingua: {self.profile.language}."
                    storyboard = llm.generate(storyboard_prompt, max_length=200, is_interrupted=self._is_interrupted)
                    if self._is_interrupted():
                        return "interrupted"
                    llm.cleanup()
                    self._update_stage(stage, "completed", storyboard)

                elif stage == "voice_generation":
                    from backend.ai_providers.kokoro_provider import KokoroProvider
                    kokoro = KokoroProvider()
                    voice_path = f"output/voice_{self.job_id}.wav"
                    if not kokoro.health_check():
                        logger.warning("Kokoro TTS non installato. Uso file audio dummy.")
                        self._generate_dummy_media("audio", voice_path)
                    else:
                        kokoro.generate(script, voice_path)
                    kokoro.cleanup()
                    self._update_stage(stage, "completed", voice_path)

                elif stage == "image_generation":
                    from backend.ai_providers.flux_provider import FluxProvider
                    flux = FluxProvider()
                    image_path = f"output/image_{self.job_id}.png"
                    image_prompt = f"Immagine di alta qualità per un video su: {expanded_topic or self.profile.custom_prompt or self.profile.topic}. Scene: {storyboard}. Eventuale testo nell'immagine deve essere in lingua: {self.profile.language}."
                    if not flux.health_check():
                        logger.warning("Flux non installato. Uso immagine dummy.")
                        self._generate_dummy_media("image", image_path)
                    else:
                        flux.generate(image_prompt, image_path)
                    flux.cleanup()
                    self._update_stage(stage, "completed", image_path)

                elif stage == "video_generation":
                    from backend.ai_providers.wan_provider import WanProvider
                    wan = WanProvider()
                    video_path = f"output/video_{self.job_id}.mp4"
                    video_prompt = f"Video short verticale basato su questo script: {script}. Il video deve essere coerente con la lingua: {self.profile.language}."
                    if not wan.health_check():
                        logger.warning("Wan 2.2 non installato. Uso video dummy.")
                        self._generate_dummy_media("video", video_path)
                    else:
                        wan.generate(video_prompt, video_path)
                    wan.cleanup()
                    self._update_stage(stage, "completed", video_path)

                elif stage == "audio_generation":
                    from backend.ai_providers.mmaudio_provider import MMAudioProvider
                    mmaudio = MMAudioProvider()
                    audio_path = f"output/audio_{self.job_id}.wav"
                    audio_prompt = f"Effetti sonori e musica di sottofondo per queste scene: {storyboard}. Eventuale voce o audio deve essere in lingua: {self.profile.language}."
                    if not mmaudio.health_check():
                        logger.warning("MMAudio non installato. Uso file audio dummy.")
                        self._generate_dummy_media("audio", audio_path)
                    else:
                        mmaudio.generate(audio_prompt, audio_path)
                    mmaudio.cleanup()
                    self._update_stage(stage, "completed", audio_path)

                elif stage == "video_assembly":
                    if not os.path.exists(video_path) or not os.path.exists(voice_path) or not os.path.exists(audio_path):
                        raise RuntimeError("File di input mancanti per l'assemblaggio video.")
                    success = FFmpegUtils.assemble_video(video_path, voice_path, audio_path, final_video_path)
                    if not success:
                        raise RuntimeError("Assemblaggio video fallito.")
                    self._update_stage(stage, "completed", final_video_path)

                elif stage == "quality_scoring":
                    quality_score = self.scorer.score(final_video_path)
                    self._update_stage(stage, "completed", str(quality_score))

                elif stage == "storage":
                    existing_video = self.db.query(Video).filter(Video.job_id == self.job_id).first()
                    if not existing_video:
                        video_record = Video(
                            job_id=self.job_id,
                            file_path=final_video_path,
                            quality_score=float(quality_score),
                            approved=False
                        )
                        self.db.add(video_record)
                        self.db.commit()
                    self._update_stage(stage, "completed", final_video_path)

                elif stage == "dashboard_review":
                    self._update_stage(stage, "waiting_for_review", "In attesa di revisione manuale")

            except Exception as e:
                self._update_stage(stage, "failed", str(e))
                logger.error(f"[Job {self.job_id}] Fallimento stage {stage}: {e}")
                return False
        return "waiting_for_review"
