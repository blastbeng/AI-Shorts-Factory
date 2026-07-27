#!/bin/bash
set -e

echo "=== Installazione llama.cpp con backend Vulkan ==="

if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Impossibile rilevare il sistema operativo."
    exit 1
fi

echo "Installazione dipendenze per la compilazione su $OS..."
case $OS in
    ubuntu|debian)
        sudo apt-get update
        sudo apt-get install -y build-essential git cmake libcurl4-openssl-dev
        ;;
    fedora)
        sudo dnf install -y gcc gcc-c++ make git cmake libcurl-devel
        ;;
    arch)
        sudo pacman -Sy --noconfirm base-devel git cmake curl
        ;;
    *)
        echo "Sistema operativo non supportato: $OS"
        exit 1
        ;;
esac

# Verifica presenza di Vulkan prima di compilare
if ! command -v vulkaninfo &> /dev/null; then
    echo "[ERRORE] Vulkan non rilevato sul sistema."
    echo "Impossibile compilare llama.cpp con backend Vulkan."
    echo "Installare Vulkan SDK o i pacchetti di sistema (es. vulkan-tools, libvulkan-dev) manualmente."
    exit 1
fi

# Clona llama.cpp
if [ ! -d "/opt/llama.cpp" ]; then
    sudo git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
fi

cd /opt/llama.cpp

# Compila con backend Vulkan
echo "Compilazione di llama.cpp con GGML_VULKAN=1..."
sudo cmake -B build-vulkan -DGGML_VULKAN=1
sudo cmake --build build-vulkan --config Release -j

# Crea la directory per i modelli se non esiste
sudo mkdir -p /opt/models

# Scarica il modello Qwen
MODEL_URL="https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/resolve/main/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf"
MODEL_PATH="/opt/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Download del modello Qwen in $MODEL_PATH..."
    sudo wget -O "$MODEL_PATH" "$MODEL_URL"
else
    echo "Modello Qwen già presente."
fi

echo "=== Installazione llama.cpp completata ==="
