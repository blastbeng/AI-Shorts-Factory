#!/bin/bash
set -e

if [ ! -d "venv" ]; then
    echo "Creazione dell'ambiente virtuale Python..."
    python3 -m venv venv
else
    echo "[OK] Ambiente virtuale Python già esistente."
fi

# Ensure CUDA toolkit is found
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST="8.6"

echo "Attivazione dell'ambiente virtuale..."
source venv/bin/activate

echo "Aggiornamento di pip..."
pip install --upgrade pip

echo "Installazione ninja per velocizzare le build..."
pip install ninja

echo "Installazione di MMAudio e dipendenze (numpy<2.1) prima di PyTorch..."
pip install git+https://github.com/hkchengrex/MMAudio.git

echo "Installazione delle librerie Python di base..."
pip install -r requirements.txt

echo "Installazione di torchcodec 0.9.1 (CPU) per compatibilità CUDA..."
pip install torchcodec==0.9.1 --index-url https://download.pytorch.org/whl/cpu --force-reinstall

echo "Rimozione di flash-attn incompatible..."
pip uninstall -y flash-attn

echo "Installazione di flash-attn 2.7.4.post1 (compatibile con PyTorch 2.6)..."
if command -v nvcc &> /dev/null; then
    pip install flash-attn==2.7.4.post1 --no-build-isolation || echo "[WARN] flash-attn non installato (richiede CUDA toolkit)."
else
    echo "[WARN] nvcc non trovato. flash-attn non può essere compilato. Verrà usato xformers come fallback."
fi

echo "Installazione di xformers compatibile con PyTorch 2.6 (dal index cu124)..."
pip install "xformers<0.0.30" --index-url https://download.pytorch.org/whl/cu124 --no-deps || echo "[WARN] xformers non installato."

echo "Reinstallazione finale di PyTorch 2.6.0 con supporto CUDA per GPU NVIDIA..."
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

echo "Rimozione delle vecchie librerie NCCL e cuDNN per evitare conflitti di file..."
pip uninstall -y nvidia-nccl-cu12 nvidia-cudnn-cu12

echo "Installazione forzata di nvidia-nccl-cu12==2.23.4 e nvidia-cudnn-cu12==9.1.0.70 per risolvere ncclCommResume e compatibilità PyTorch 2.6..."
pip install nvidia-nccl-cu12==2.23.4 nvidia-cudnn-cu12==9.1.0.70 --no-deps

# Re-export library path before importing torch (SpaCy may trigger torch import)
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SITE_PACKAGES/torch/lib:$SITE_PACKAGES/nvidia/nccl/lib:$LD_LIBRARY_PATH"

# Force loading of the correct NCCL library (required by PyTorch 2.6+)
NCCL_LIB=$(find "$SITE_PACKAGES/nvidia/nccl/lib" -name "libnccl.so.2" 2>/dev/null | head -1)
if [ -n "$NCCL_LIB" ]; then
    export LD_PRELOAD="$NCCL_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "[OK] Preloading NCCL from: $NCCL_LIB"
else
    echo "[WARN] NCCL library not found in nvidia-nccl-cu12 package. Attempting to continue..."
fi

echo "Verifica dei modelli linguistici SpaCy per Kokoro TTS..."
for model in en_core_web_sm it_core_news_sm es_core_news_sm fr_core_news_sm de_core_news_sm; do
    if ! python -c "import spacy; spacy.load('$model')" 2>/dev/null; then
        echo "Download del modello SpaCy: $model..."
        python -m spacy download $model
    else
        echo "[OK] Modello SpaCy $model già installato."
    fi
done

echo "Verifica di llama-cpp-python con backend CUDA o Vulkan..."
if ! python -c "import llama_cpp" 2>/dev/null; then
    if command -v nvcc &> /dev/null; then
        echo "Installazione di llama-cpp-python con backend CUDA..."
        CMAKE_ARGS="-DGGML_CUDA=on" CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir -v
    #elif command -v vulkaninfo &> /dev/null; then
    #    echo "[WARN] nvcc non trovato. Installazione di llama-cpp-python con backend Vulkan..."
    #    CMAKE_ARGS="-DGGML_VULKAN=on" CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir -v
    else
        echo "[ERRORE] Nessun backend GPU (CUDA o Vulkan) trovato. Impossibile installare llama-cpp-python con accelerazione GPU."
        exit 1
    fi
else
    echo "[OK] llama-cpp-python già installato."
fi

echo "Installazione dipendenze (numpy<2.1 pillow<12.0)..."
pip install "numpy<2.1,>=1.21" "pillow<12.0"

echo "Ambiente Python configurato con successo."
deactivate
