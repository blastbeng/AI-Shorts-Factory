#!/bin/bash
set -e

echo "Creazione dell'ambiente virtuale Python..."
python3 -m venv venv

echo "Attivazione dell'ambiente virtuale..."
source venv/bin/activate

echo "Aggiornamento di pip..."
pip install --upgrade pip

echo "Installazione delle librerie Python di base..."
pip install -r requirements.txt

echo "Installazione di llama-cpp-python con backend Vulkan..."
CMAKE_ARGS="-DGGML_VULKAN=on" CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

echo "Ambiente Python configurato con successo."
deactivate
