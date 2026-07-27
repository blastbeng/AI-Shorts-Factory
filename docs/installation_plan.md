# AI Shorts Factory - Piano di Installazione

## Script di Installazione

Gli script risiedono in `scripts/` e devono essere eseguiti nell'ordine indicato:

| # | Script | Descrizione |
|---|---|---|
| 1 | `install_system_dependencies.sh` | Pacchetti di sistema (ffmpeg, build-essential, ecc.) |
| 2 | `install_gpu_stack.sh` | Driver GPU, ROCm, CUDA toolkit |
| 3 | `install_python_environment.sh` | Python venv, pip, dipendenze Python |
| 4 | `install_video_models.sh` | Setup provider video (Wan 2.2, LTX Video) |
| 5 | `install_audio_models.sh` | Setup provider audio (MMAudio) |
| 6 | `install_voice_models.sh` | Setup provider voce (Kokoro TTS) |
| 7 | `install_image_models.sh` | Setup provider immagini (Flux, Qwen Image) |
| 8 | `install_speech_models.sh` | Setup provider speech (Whisper) |
| 9 | `download_models.sh` | Download di tutti i pesi dei modelli |
| 10 | `verify_ai_environment.sh` | Verifica completa dell'ambiente AI |

## Struttura Cartella `models/`

```
models/
├── video/     # Modelli video (Wan 2.2, LTX Video)
├── audio/     # Modelli audio (MMAudio)
├── voice/     # Modelli voce (Kokoro TTS)
├── image/     # Modelli immagini (Flux, Qwen Image)
└── speech/    # Modelli speech (Whisper)
```

## Struttura Cartella `configs/`

```
configs/
├── models.yaml   # Configurazione modelli (percorsi, parametri, backend)
├── gpu.yaml       # Configurazione GPU (assegnazione, priorità, limiti VRAM)
└── paths.yaml     # Percorsi filesystem (models dir, output dir, temp dir)
```
