import os
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import uuid

# Crea la directory del database se non esiste
os.makedirs("backend", exist_ok=True)
SQLALCHEMY_DATABASE_URL = "sqlite:///./backend/app.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Video(Base):
    __tablename__ = "videos"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, index=True)
    prompt = Column(String)
    script_text = Column(String)
    status = Column(String, default="Pending Review") # Pending Review, Approved, Rejected, Posting, Posted, Failed
    video_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Crea le tabelle nel database
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
