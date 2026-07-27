# AI Shorts Factory

Piattaforma locale autonoma per la generazione di video short-form (TikTok, YouTube Shorts, ecc.) utilizzando AI nativa su workstation Linux multi-GPU.

## Requisiti

- Linux (Ubuntu/Debian raccomandato)
- Python 3.10+
- Node.js 18+ e npm
- GPU con driver ROCm (AMD) o CUDA (NVIDIA)

## Installazione

1. Clona il repository.
2. Esegui lo script di setup completo:
   ```bash
   chmod +x scripts/*.sh
   ./scripts/setup_project.sh
   ```
   Questo script installerà le dipendenze di sistema, lo stack GPU, l'ambiente Python, le dipendenze frontend e scaricherà i modelli AI.
3. Configura il file `.env`:
   ```bash
   cp .env.example .env
   nano .env
   ```
   Modifica le variabili d'ambiente per LLM, Social Media e Host secondo le tue esigenze.

## Avvio

Per avviare sia il backend che il frontend contemporaneamente:
```bash
./scripts/start.sh
```
- Backend: `http://0.0.0.0:8000`
- Frontend: `http://0.0.0.0:3000`

## Test

Per verificare che l'installazione e la pipeline funzionino correttamente:
```bash
./scripts/run_mvp_test.sh
```

## Architettura

- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: Next.js, TypeScript, Tailwind CSS
- **AI Providers**: Wan 2.2, MMAudio, Kokoro TTS, Flux, Whisper, LLM (Ollama/OpenAI)
- **GPU Manager**: Supporto multi-GPU (ROCm/CUDA) con scheduling configurabile
