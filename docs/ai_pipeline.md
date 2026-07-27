# AI Shorts Factory - Pipeline AI

## Gli 11 Stadi della Pipeline

| # | Stadio | Descrizione |
|---|---|---|
| 1 | Topic generation | Generazione del topic/argomento del video |
| 2 | Script generation | Generazione della sceneggiatura/testo |
| 3 | Voice generation | Sintesi vocale (TTS) della narrazione |
| 4 | Storyboard creation | Creazione dello storyboard (scene, shot, timing) |
| 5 | Image generation | Generazione delle immagini per le scene |
| 6 | Video generation | Generazione dei clip video dalle immagini/storyboard |
| 7 | Audio generation | Generazione di audio/effetti sonori (MMAudio) |
| 8 | Video assembly | Assemblaggio finale (video + voce + audio + sottotitoli) |
| 9 | Quality scoring | Valutazione automatica della qualità del video |
| 10 | Storage | Salvataggio su filesystem locale e catalogazione |
| 11 | Dashboard review | Revisione manuale tramite dashboard web |

## Principi della Pipeline

- **Indipendenza**: ogni stadio è autonomo e non dipende dallo stato in memoria degli altri.
- **Riavviabilità**: ogni stadio può essere riavviato singolarmente senza ripetere quelli precedenti.
- **Tracciabilità**: ogni stadio registra il proprio stato nel database (pending, running, completed, failed).
- **Logging**: ogni stadio produce log strutturati per debug e monitoraggio.
- **Gestione errori**: ogni stadio gestisce le proprie eccezioni e notifica il failure alla pipeline orchestrator.
