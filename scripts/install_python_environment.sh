#!/bin/bash
set -e

if [ ! -d "venv" ]; then
    echo "Creazione dell'ambiente virtuale Python..."
    python3 -m venv venv
else
    echo "[OK] Ambiente virtuale Python già esistente."
fi

echo "Attivazione dell'ambiente virtuale..."
source venv/bin/activate

echo "Aggiornamento di pip..."
pip install --upgrade pip

echo "Installazione di MMAudio e dipendenze (numpy<2.1) prima di PyTorch..."
pip install git+https://github.com/hkchengrex/MMAudio.git

echo "Installazione di PyTorch con supporto ROCm per GPU AMD (sovrascrive torch di mmaudio)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4 --force-reinstall

echo "Installazione delle librerie Python di base..."
pip install -r requirements.txt

echo "Reinstallazione di PyTorch con supporto ROCm per GPU AMD (dopo requirements.txt)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4 --force-reinstall

echo "Installazione di torchcodec 0.9.1 (CPU) per compatibilità ROCm..."
pip install torchcodec==0.9.1 --index-url https://download.pytorch.org/whl/cpu --force-reinstall

echo "Verifica dei modelli linguistici SpaCy per Kokoro TTS..."
for model in en_core_web_sm it_core_news_sm es_core_news_sm fr_core_news_sm de_core_news_sm; do
    if ! python -c "import spacy; spacy.load('$model')" 2>/dev/null; then
        echo "Download del modello SpaCy: $model..."
        python -m spacy download $model
    else
        echo "[OK] Modello SpaCy $model già installato."
    fi
done

echo "Installazione dipendenze (numpy<2.1 pillow<12.0)..."
pip install "numpy<2.1,>=1.21" "pillow<12.0"

echo "Verifica di llama-cpp-python con backend Vulkan..."
if ! python -c "import llama_cpp" 2>/dev/null; then
    echo "Installazione di llama-cpp-python con backend Vulkan..."
    CMAKE_ARGS="-DGGML_VULKAN=on" CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir -v
else
    echo "[OK] llama-cpp-python già installato."
fi

echo "Ambiente Python configurato con successo."
deactivate
