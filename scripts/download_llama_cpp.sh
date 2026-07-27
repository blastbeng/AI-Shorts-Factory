#!/bin/bash
set -e

echo "=== Download del modello LLM (Qwen GGUF) ==="

# Crea la directory per i modelli se non esiste
sudo mkdir -p /opt/models

# Scarica il modello Qwen
MODEL_URL="https://huggingface.co/DavidAU/Llama-3.2-8X4B-MOE-V2-Dark-Champion-Instruct-uncensored-abliterated-21B-GGUF/resolve/main/L3.2-8X4B-MOE-V2-Dark-Champion-Inst-21B-uncen-ablit-D_AU-Q4_k_m.gguf"
MODEL_PATH="/opt/models/L3.2-8X4B-MOE-V2-Dark-Champion-Inst-21B-uncen-ablit-D_AU-Q4_k_m.gguf"

# Verifica se il file esiste e ha una dimensione maggiore di 0
if [ -f "$MODEL_PATH" ] && [ -s "$MODEL_PATH" ]; then
    echo "[OK] Modello Qwen già presente. Salto il download."
else
    echo "Modello Qwen non trovato, vuoto o corrotto."
    if [ -f "$MODEL_PATH" ]; then
        echo "Cancellazione del file corrotto..."
        sudo rm -f "$MODEL_PATH"
    fi
    echo "Download del modello Qwen in $MODEL_PATH..."
    sudo wget -O "$MODEL_PATH" "$MODEL_URL"
fi

echo "=== Download del modello LLM completato ==="
