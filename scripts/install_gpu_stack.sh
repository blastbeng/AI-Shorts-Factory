#!/bin/bash
set -e

echo "Installazione dello stack GPU..."

# Installa dipendenze per ROCm (AMD)
echo "Installazione dipendenze ROCm (AMD)..."
sudo apt-get install -y rocm-core rocm-smi-lib

# Installa dipendenze per CUDA (NVIDIA)
echo "Installazione dipendenze CUDA (NVIDIA)..."
sudo apt-get install -y nvidia-cuda-toolkit

# Verifica driver
if ! command -v nvidia-smi &> /dev/null && ! command -v rocm-smi &> /dev/null; then
    echo "Attenzione: Nessun driver GPU rilevato. Assicurati di aver installato i driver corretti."
fi

echo "Stack GPU installato."
