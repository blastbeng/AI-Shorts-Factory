from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, ForeignKey
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
    id = Column(Integer, primary_key=True, index=True)
    status = Column(String, default="pending")  # pending, running, completed, failed
    profile_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"))
    file_path = Column(String, nullable=False)
    quality_score = Column(Float, nullable=True)
    approved = Column(Boolean, default=False)
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

class GenerationProfile(Base):
    __tablename__ = "generation_profiles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    topic = Column(String, nullable=False)
    style = Column(String, default="default")
    duration_seconds = Column(Integer, default=30)
    created_at = Column(DateTime, default=datetime.utcnow)

class PipelineStage(Base):
    __tablename__ = "pipeline_stages"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"))
    stage_name = Column(String, nullable=False)
    status = Column(String, default="pending") # pending, running, completed, failed
    result = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
