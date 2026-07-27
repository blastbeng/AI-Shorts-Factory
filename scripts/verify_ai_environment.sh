#!/bin/bash
set -e

echo "Verifica dell'ambiente AI Shorts Factory..."

ERRORS=0

# Verifica ambiente virtuale
if [ -d "venv" ]; then
    echo "[OK] Ambiente virtuale Python 'venv' trovato."
else
    echo "[ERRORE] Ambiente virtuale Python 'venv' non trovato."
    ERRORS=$((ERRORS + 1))
fi

# Verifica directory dei modelli
for dir in models/video models/audio models/voice models/image models/speech; do
    if [ -d "$dir" ]; then
        echo "[OK] Directory '$dir' trovata."
    else
        echo "[ERRORE] Directory '$dir' non trovata."
        ERRORS=$((ERRORS + 1))
    fi
done

# Verifica file di configurazione
for file in configs/paths.yaml configs/gpu.yaml configs/models.yaml; do
    if [ -f "$file" ]; then
        echo "[OK] File di configurazione '$file' trovato."
    else
        echo "[ERRORE] File di configurazione '$file' non trovato."
        ERRORS=$((ERRORS + 1))
    fi
done

echo "----------------------------------------"
if [ "$ERRORS" -eq 0 ]; then
    echo "Verifica completata: Tutti i componenti sono presenti."
else
    echo "Verifica completata con $ERRORS errori."
    exit 1
fi
