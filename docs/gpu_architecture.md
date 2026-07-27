# AI Shorts Factory - Architettura GPU

## Gestione Multi-GPU

Il modulo `backend/gpu_manager/` è responsabile di:

- **Rilevamento GPU**: identificazione automatica delle GPU disponibili (AMD via ROCm, NVIDIA via CUDA).
- **Monitoraggio VRAM**: controllo in tempo reale della memoria VRAM occupata/libera per ogni GPU.
- **Scheduling**: decisione su quale GPU assegnare a un dato carico di lavoro.
- **Assegnazione carichi**: associazione di un provider AI a una GPU specifica in base ai requisiti.

## Configurabilità

- Nessuna selezione GPU hardcoded nel codice.
- L'assegnazione è guidata da `configs/gpu.yaml`.
- Il `gpu_manager` espone API per richiedere una GPU libera con determinati requisiti.

## Esempio di Assegnazione

| GPU | Modello | Carichi Tipici |
|---|---|---|
| GPU 1 | AMD RX 7800 XT 16GB | Video generation, Image generation |
| GPU 2 | NVIDIA RTX 3060 12GB | TTS, Whisper, Audio generation |

> L'assegnazione è configurabile e non fissa. Il `gpu_manager` può riassegnare carichi dinamicamente in base alla disponibilità.
