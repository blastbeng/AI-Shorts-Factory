# AI Shorts Factory - Piano di Installazione

## Script di Installazione

Gli script risiedono in `scripts/` e devono essere eseguiti nell'ordine indicato:

| # | Script | Descrizione |
|---|---|---|
| 1 | `install_system_dependencies.sh` | Pacchetti di sistema (ffmpeg, build-essential, ecc.) |
| 2 | `verify_gpu_stack.sh` | Verifica la presenza di driver GPU, ROCm, CUDA e Vulkan (non installa) |
| 2.5 | `install_llama_cpp.sh` | Compila llama.cpp da sorgente con backend Vulkan, scarica modello Qwen GGUF |
| 3 | `install_python_environment.sh` | Python venv, pip, dipendenze Python |
| 4 | `install_video_models.sh` | Setup provider video (Wan 2.2, LTX Video) |
| 5 | `install_audio_models.sh` | Setup provider audio (MMAudio) |
| 6 | `install_voice_models.sh` | Setup provider voce (Kokoro TTS) |
| 7 | `install_image_models.sh` | Setup provider immagini (Flux, Qwen Image) |
| 8 | `install_speech_models.sh` | Setup provider speech (Whisper) |
| 9 | `download_models.sh` | Download di tutti i pesi dei modelli |
| 10 | `verify_ai_environment.sh` | Verifica completa dell'ambiente AI |
| 11 | `start.sh` | Avvia e monitora backend e frontend contemporaneamente (Ctrl+C per fermare) |

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
