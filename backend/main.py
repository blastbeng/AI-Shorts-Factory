from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from backend.database import get_db, Video

app = FastAPI(title="Short-Form Video Generator API")

class VideoOut(BaseModel):
    id: str
    title: str
    prompt: str
    script_text: str
    status: str
    video_path: str | None = None
    created_at: str

    class Config:
        from_attributes = True

@app.get("/api/videos", response_model=List[VideoOut])
def get_videos(db: Session = Depends(get_db)):
    videos = db.query(Video).order_by(Video.created_at.desc()).all()
    return videos
