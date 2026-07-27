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

echo "Installazione delle dipendenze di sistema essenziali..."

case $OS in
    ubuntu|debian)
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-venv git curl wget ffmpeg build-essential
        ;;
    fedora)
        # Su Fedora, python3-venv è incluso in python3, build-essential è sostituito da gcc/gcc-c++/make.
        # Usiamo --allowerasing per risolvere il conflitto tra ffmpeg-free e ffmpeg (se rpmfusion è abilitato).
        sudo dnf install -y python3 python3-pip python3-devel git curl wget ffmpeg gcc gcc-c++ make --allowerasing
        ;;
    arch)
        sudo pacman -Sy --noconfirm python python-pip git curl wget ffmpeg base-devel
        ;;
    *)
        echo "Sistema operativo non supportato: $OS"
        echo "Assicurati di installare manualmente: python3, pip, venv, git, curl, wget, ffmpeg, build-essential (o equivalenti)."
        exit 1
        ;;
esac

echo "Dipendenze di sistema installate con successo."
