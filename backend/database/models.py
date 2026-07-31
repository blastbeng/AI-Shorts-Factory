import os
import uuid
from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database.session import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String, default="pending")  # pending, running, completed, failed
    genre = Column(String, default="random")
    custom_prompt = Column(String, nullable=True)
    language = Column(String, default="italian")
    style = Column(String, default="default")
    duration_seconds = Column(Integer, default=16)
    gen_width = Column(Integer, default=lambda: int(os.getenv("GEN_WIDTH", 480)))
    gen_height = Column(Integer, default=lambda: int(os.getenv("GEN_HEIGHT", 832)))
    width = Column(Integer, default=353)
    height = Column(Integer, default=640)
    gen_frames = Column(Integer, default=lambda: int(os.getenv("GEN_FRAMES", 49)))
    gen_flux_steps = Column(Integer, default=lambda: int(os.getenv("GEN_FLUX_STEPS", 4)))
    gen_wan_steps = Column(Integer, default=lambda: int(os.getenv("GEN_WAN_STEPS", 30)))
    gen_ltx_steps = Column(Integer, default=lambda: int(os.getenv("GEN_LTX_STEPS", 50)))
    video_provider = Column(String, default="wan")
    generate_subtitles = Column(Boolean, default=True)
    input_image = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    videos = relationship("Video", backref="job")

class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, ForeignKey("jobs.id"))
    file_path = Column(String, nullable=False)
    quality_score = Column(Float, nullable=True)
    approved = Column(Boolean, default=False)
    published = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelInfo(Base):
    __tablename__ = "model_infos"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # video, audio, voice, image, speech
    version = Column(String, nullable=True)
    path = Column(String, nullable=False)
    vram_required = Column(Integer, nullable=True)
    backend = Column(String, nullable=True)
    status = Column(String, default="not_installed")

class PipelineStage(Base):
    __tablename__ = "pipeline_stages"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, ForeignKey("jobs.id"))
    stage_name = Column(String, nullable=False)
    status = Column(String, default="pending") # pending, running, completed, failed
    result = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
