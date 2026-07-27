#!/bin/bash
set -e

echo "=== Installazione/Aggiornamento llama.cpp con backend Vulkan ==="

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

REPO_DIR="/opt/services/llama.cpp"

# Gestione del repository: clona se non esiste, aggiorna se esiste
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Clonazione di llama.cpp in $REPO_DIR..."
    sudo mkdir -p /opt/services
    sudo git clone https://github.com/ggerganov/llama.cpp "$REPO_DIR"
else
    echo "Repository llama.cpp trovato in $REPO_DIR. Aggiornamento..."
    cd "$REPO_DIR"
    sudo git pull
fi

cd "$REPO_DIR"

# Verifica la presenza di CMakeLists.txt
if [ ! -f "CMakeLists.txt" ]; then
    echo "[ERRORE] CMakeLists.txt non trovato in $REPO_DIR. Il repository potrebbe essere corrotto."
    exit 1
fi

# Compila con backend Vulkan
echo "Compilazione di llama.cpp con GGML_VULKAN=1..."
sudo cmake -B build-vulkan -DGGML_VULKAN=1
sudo cmake --build build-vulkan --config Release -j

# Crea la directory per i modelli se non esiste
sudo mkdir -p /opt/models

# Scarica il modello Qwen
MODEL_URL="https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/resolve/main/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf"
MODEL_PATH="/opt/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf"

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

echo "=== Installazione llama.cpp completata ==="
