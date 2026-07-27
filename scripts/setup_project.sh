#!/bin/bash
set -e

echo "=== AI Shorts Factory - Setup Completo ==="

# 1. Copia il file .env se non esiste
if [ ! -f ".env" ]; then
    echo "Copia del file .env.example in .env..."
    cp .env.example .env
    echo "[OK] File .env creato. Modificalo con le tue configurazioni (LLM, Social, ecc.)."
else
    echo "[SALTA] File .env già esistente."
fi

# 2. Installa le dipendenze di sistema
echo "=== Fase 1: Dipendenze di Sistema ==="
chmod +x scripts/*.sh
./scripts/install_system_dependencies.sh

# 3. Installa lo stack GPU
echo "=== Fase 2: Stack GPU ==="
./scripts/verify_gpu_stack.sh

echo "=== Fase 2.5: Installazione llama.cpp ==="
./scripts/install_llama_cpp.sh

# 4. Installa l'ambiente Python
echo "=== Fase 3: Ambiente Python ==="
./scripts/install_python_environment.sh

# 5. Installa le dipendenze del frontend
echo "=== Fase 4: Dipendenze Frontend ==="
if command -v npm &> /dev/null; then
    cd frontend
    npm install
    cd ..
    echo "[OK] Dipendenze frontend installate."
else
    echo "[ERRORE] npm non trovato. Installa Node.js e npm."
fi

# 6. Download dei modelli AI
echo "=== Fase 5: Download Modelli AI ==="
./scripts/install_video_models.sh
./scripts/install_audio_models.sh
./scripts/install_voice_models.sh
./scripts/install_image_models.sh
./scripts/install_speech_models.sh

# 7. Verifica dell'ambiente
echo "=== Fase 6: Verifica Ambiente ==="
./scripts/verify_ai_environment.sh

echo "=== Setup Completato! ==="
echo "Per avviare il backend: source venv/bin/activate && uvicorn backend.api.main:app --reload"
echo "Per avviare il frontend: cd frontend && npm run dev"
