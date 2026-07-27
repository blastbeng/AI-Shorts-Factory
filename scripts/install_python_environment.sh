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

echo "Download dei modelli linguistici SpaCy per Kokoro TTS..."
python -m spacy download en_core_web_sm
python -m spacy download it_core_news_sm
python -m spacy download es_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download de_core_news_sm

echo "Installazione di llama-cpp-python con backend Vulkan..."
CMAKE_ARGS="-DGGML_VULKAN=on" CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) pip install llama-cpp-python "numpy<2.5" --upgrade --force-reinstall --no-cache-dir -v

echo "Ambiente Python configurato con successo."
deactivate
