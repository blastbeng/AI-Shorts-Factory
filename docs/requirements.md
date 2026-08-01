# AI Shorts Factory - Requisiti

## Requisiti Hardware

| Componente | Specifica |
|---|---|
| CPU | AMD Ryzen 7 7800X3D |
| RAM | 64GB |
| GPU 1 | NVIDIA RTX 3060 12GB |

## Supporto GPU

- CUDA (NVIDIA)
- Vulkan (cross-vendor, per llama.cpp)

## Modelli AI Richiesti

| Modello | Tipo | Stato |
|---|---|---|
| Wan 2.2 5B | Video generation | Richiesto |
| MMAudio | Audio generation | Richiesto |
| Kokoro TTS | Voice generation | Richiesto |
| LTX Video | Video generation | Futuro |
| Flux | Image generation | Futuro |
| Qwen Image | Image generation | Futuro |
| Whisper | Speech-to-text | Futuro |

## Stack Tecnologico

| Componente | Tecnologia |
|---|---|
| Frontend | Next.js |
| Backend | FastAPI |
| Database | SQLite + SQLAlchemy |
| Storage | Filesystem locale |

## Vincoli

- Niente ComfyUI in produzione.
- Solo inferenza Python nativa.
- Nessuna dipendenza da cloud o Docker.
