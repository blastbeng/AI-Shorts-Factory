# AI Shorts Factory - Strategie di Ottimizzazione

## Ottimizzazione VRAM

- **Quantizzazione**: utilizzo di formati a bassa precisione (es. FP8, INT8) per i modelli video e immagini al fine di ridurre l'occupazione di memoria VRAM e permettere l'esecuzione di modelli più grandi su GPU con memoria limitata.
- **Offloading**: spostamento dinamico dei layer del modello tra CPU e GPU per gestire modelli che superano la capacità VRAM singola.

## Pipeline Asincrona

- **Task Queue**: migrazione da esecuzione sincrona a task asincroni utilizzando broker come Celery o RQ. Questo eviterà di bloccare le richieste API e permetterà una migliore gestione della concorrenza.
- **Worker Dedicati**: deploy di worker separati per stadi pesanti (es. generazione video) per isolare le risorse.

## Caching

- **Risultati Intermedi**: caching dei risultati intermedi della pipeline (es. script generato, storyboard) per evitare ricalcoli in caso di fallimento di uno stadio successivo, permettendo il riavvio rapido della pipeline dal punto di rottura.
- **Caching Modelli**: mantenimento dei modelli caricati in memoria per riutilizzo rapido tra job consecutivi, riducendo il tempo di inizializzazione.
