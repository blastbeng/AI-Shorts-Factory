#!/bin/bash
set -e

echo "Aggiornamento dei pacchetti apt..."
sudo apt-get update

echo "Installazione delle dipendenze di sistema essenziali..."
sudo apt-get install -y python3 python3-pip python3-venv git curl wget ffmpeg build-essential

echo "Dipendenze di sistema installate con successo."
