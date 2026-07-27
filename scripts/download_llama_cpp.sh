#!/bin/bash
set -e

echo "=== Download del modello LLM (Qwen GGUF) ==="

# Crea la directory per i modelli se non esiste
sudo mkdir -p /opt/models

# Scarica il modello Qwen
MODEL_URL="https://huggingface.co/mradermacher/Huihui-Qwen3-Omni-30B-A3B-Instruct-abliterated-i1-GGUF/resolve/main/Huihui-Qwen3-Omni-30B-A3B-Instruct-abliterated.i1-Q6_K.gguf"
MODEL_PATH="/opt/models/Huihui-Qwen3-Omni-30B-A3B-Instruct-abliterated.i1-Q6_K.gguf"

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
