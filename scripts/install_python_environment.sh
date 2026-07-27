#!/bin/bash
set -e

echo "Creazione dell'ambiente virtuale Python..."
python3 -m venv venv

echo "Attivazione dell'ambiente virtuale..."
source venv/bin/activate

echo "Aggiornamento di pip..."
pip install --upgrade pip

echo "Installazione delle librerie Python di base..."
pip install fastapi uvicorn sqlalchemy pyyaml torch torchvision torchaudio

echo "Ambiente Python configurato con successo."
deactivate
