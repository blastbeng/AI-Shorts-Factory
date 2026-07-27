from fastapi import FastAPI
from backend.database.session import Base, engine
from backend.gpu_manager.manager import GPUManager

app = FastAPI(title="AI Shorts Factory")

Base.metadata.create_all(bind=engine)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/gpus")
def gpus():
    gm = GPUManager()
    return gm.get_gpus()
