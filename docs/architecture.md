# AI Shorts Factory - Architettura

## Struttura del Progetto

```
ai-shorts-factory/
├── frontend/          # Applicazione Next.js (dashboard web)
├── backend/
│   ├── api/           # Endpoint REST API (FastAPI)
│   ├── domain/        # Modelli di dominio e logica business
│   ├── services/      # Servizi applicativi (orchestrazione pipeline)
│   ├── workers/       # Worker per l'esecuzione degli stadi della pipeline
│   ├── ai_providers/  # Provider per ogni modello AI
│   ├── gpu_manager/   # Gestione e scheduling multi-GPU
│   ├── media/         # Utility per manipolazione media (ffmpeg, ecc.)
│   ├── database/      # Modelli SQLAlchemy, migrazioni, sessioni
│   └── storage/       # Gestione filesystem locale (asset, output, temporanei)
├── scripts/           # Script di installazione e setup
├── configs/           # File di configurazione YAML
├── models/            # Modelli AI scaricati
└── docs/              # Documentazione
```

## Architettura AI Provider

Ogni modello AI ha un provider dedicato in `backend/ai_providers/`. Ogni provider implementa un'interfaccia standard con i seguenti metodi:

- `install_status()` — Verifica se il modello è installato e pronto.
- `health_check()` — Verifica lo stato di salute del provider (GPU disponibile, VRAM sufficiente, ecc.).
- `generate(params)` — Esegue l'inferenza e restituisce il risultato.
- `get_capabilities()` — Restituisce le capacità del modello (formati supportati, risoluzioni, ecc.).
- `get_gpu_requirements()` — Restituisce i requisiti GPU (VRAM minima, backend preferito, ecc.).

## Architettura Social Media

Interfacce pulite per l'integrazione con le piattaforme social, in `backend/services/social/`:

- TikTok
- YouTube Shorts
- Instagram Reels
- Facebook Reels

Nessuna implementazione iniziale: solo interfacce e contratti definiti.
