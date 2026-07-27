#!/bin/bash
set -e

echo "Rilevamento del sistema operativo..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Impossibile rilevare il sistema operativo. Assicurati di installare manualmente le dipendenze."
    exit 1
fi

echo "Sistema operativo rilevato: $OS"

install_packages() {
    case $OS in
        ubuntu|debian)
            sudo apt-get update
            sudo apt-get install -y "$@"
            ;;
        fedora)
            sudo dnf install -y "$@"
            ;;
        arch)
            sudo pacman -Sy --noconfirm "$@"
            ;;
        *)
            echo "Sistema operativo non supportato: $OS"
            exit 1
            ;;
    esac
}

echo "Installazione delle dipendenze di sistema essenziali..."
install_packages python3 python3-pip python3-venv git curl wget ffmpeg build-essential

echo "Dipendenze di sistema installate con successo."
