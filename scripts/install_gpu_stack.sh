#!/bin/bash
set -e

echo "Installazione dello stack GPU..."

if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Impossibile rilevare il sistema operativo."
    exit 1
fi

case $OS in
    ubuntu|debian)
        echo "Installazione dipendenze ROCm e CUDA (Ubuntu/Debian)..."
        sudo apt-get update
        sudo apt-get install -y rocm-core rocm-smi-lib nvidia-cuda-toolkit
        ;;
    fedora)
        echo "Installazione dipendenze ROCm e CUDA (Fedora)..."
        # Fedora richiede spesso l'uso di rpmfusion per i driver NVIDIA/CUDA
        sudo dnf install -y rocm-smi nvidia-cuda-toolkit
        ;;
    arch)
        echo "Installazione dipendenze ROCm e CUDA (Arch)..."
        # Su Arch, rocm e cuda sono disponibili in AUR o extra
        sudo pacman -Sy --noconfirm rocm cuda
        ;;
    *)
        echo "Sistema operativo non supportato per l'installazione automatica dello stack GPU: $OS"
        exit 1
        ;;
esac

# Verifica driver
if ! command -v nvidia-smi &> /dev/null && ! command -v rocm-smi &> /dev/null; then
    echo "Attenzione: Nessun driver GPU rilevato. Assicurati di aver installato i driver corretti."
fi

echo "Stack GPU installato."
